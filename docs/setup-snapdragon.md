# Setting up on a Snapdragon X laptop

This is the part that actually decides whether the app uses your NPU. Work through it in
order, and reopen setup after each step to see what changed.

## 1. Build native, not emulated

This matters more than anything else here. Under x64 emulation the QNN provider cannot load at
all, and the failure looks exactly like a missing driver.

```powershell
dotnet build -c Release -r win-arm64
```

Check it worked by launching the app. Setup lists `Native arm64 process` first, and it must
not report running under emulation. If it does, nothing below will help until that is fixed.

## 2. Install the Hexagon NPU runtime driver

The driver Windows ships with is not enough. You need the separate Hexagon NPU Runtime Driver,
which comes from Qualcomm Software Center and needs a free developer account.

This is the single most common reason a Snapdragon app quietly runs on the CPU. Everything
still works, so nothing complains — it is just slower and the machine feels busier than it
should.

After installing, restart the app. Setup should show `Hexagon NPU runtime driver` as present.

## 3. Get Whisper model assets for your exact chip

Setup does this for you: launch the app and choose **Download what's missing**.

It queries Hugging Face for a build matching your chipset, downloads the files, and writes a
small `localscribe-model.json` recording which file is the encoder, the decoder, and the
vocabulary. Files keep their published names rather than being renamed, because ONNX models
above 2 GB reference their weight sidecars by name from inside the graph.

The rest of this section is for doing it by hand, or for when the automatic search comes up
empty.

Precompiled QNN binaries are **chipset-specific**. A build for Snapdragon X Elite will not load
on X Plus or X2. Setup reports which chipset folder it searched.

Qualcomm publishes pre-exported Whisper assets on Hugging Face under the `qualcomm`
organisation (`Whisper-Tiny`, `Whisper-Base`, `Whisper-Small`). Look for the
`PRECOMPILED_QNN_ONNX` asset for your chipset.

To export a different size yourself, use Qualcomm AI Hub:

```bash
python -m qai_hub_models.models.whisper_base_en.export \
  --chipset qualcomm-snapdragon-x-elite \
  --target-runtime precompiled_qnn_onnx \
  --components HfWhisperEncoder HfWhisperDecoder
```

Lay the files out like this, where the folder name matches your chip:

```
models/
  snapdragon-x-elite/
    base.en/
      encoder.onnx
      decoder.onnx
      vocab.json
```

`vocab.json` comes from the matching Hugging Face Whisper repo, not from AI Hub.

### Version pinning

Precompiled context binaries are tied to the QAIRT and ONNX Runtime versions they were built
with. The model card lists both. If a model fails to load with a context-binary error, a
version mismatch is the first thing to check.

## 4. Optional: Foundry Local for the cleanup pass

Transcription works without this. Punctuation repair, glossary correction, and summaries do not.
`--install` sets it up; by hand it is:

```powershell
winget install Microsoft.FoundryLocal
foundry service start
foundry model run qwen2.5-1.5b-instruct
```

Pass a model **alias** rather than a fully qualified id. Foundry then picks the QNN NPU build on
Snapdragon and falls back to CPU elsewhere, which is exactly the behaviour we want.

### The port is dynamic

Foundry Local binds to a **dynamic** loopback port, so do not hard-code one. Ask the CLI:

```powershell
foundry service status
```

The app does this automatically. It matters because a hard-coded port makes a perfectly healthy
service look absent, and the app then silently skips the cleanup pass.

Note that Foundry Local has had open bugs affecting NPU generation on Snapdragon X Elite. If
generation crashes, drop back to a CPU variant — the cleanup model is small enough that the
Oryon cores handle it comfortably.

## Verifying the NPU is genuinely in use

Setup's checks confirm the pieces are installed. To prove work actually lands on the NPU:

1. Open setup and confirm it reports the encoder running on the NPU.
2. Open Task Manager, Performance tab, and watch the NPU graph during a transcription.
3. If the NPU stays flat while the CPU climbs, you are seeing silent fallback.

The strict check exists for exactly this case. It sets `session.disable_cpu_ep_fallback`, which
makes ONNX Runtime throw rather than quietly relocate unsupported nodes to the CPU. A loud
failure is far easier to diagnose than a quiet one.

## Why the decoder stays on the CPU

Only the encoder is offloaded, even when the NPU is available and healthy. The decoder emits one
short step at a time, and per-call dispatch overhead dominates its runtime — roughly 50 to 100
microseconds per call, several hundred calls per token. The Oryon cores beat the NPU at that
shape of work.

This is also why the llama.cpp Hexagon backend is not used for the cleanup model. It requires
disabling Secure Boot to load test-signed driver libraries, and still tends to lose to the CPU
on small models. The Adreno OpenCL backend is the sane GPU option if you want one.
