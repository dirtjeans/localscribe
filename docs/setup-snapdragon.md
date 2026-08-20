# Setting up on a Snapdragon X laptop

This is the part that actually decides whether the app uses your NPU. Work through it in
order, and run the doctor after each step.

## 1. Build native, not emulated

This matters more than anything else here. Under x64 emulation the QNN provider cannot load at
all, and the failure looks exactly like a missing driver.

```powershell
dotnet build -c Release -r win-arm64
```

Check it worked:

```powershell
dotnet run --project src/LocalScribe.Doctor -c Release -r win-arm64
```

The `Process architecture` line must say `Arm64`. If it says `X64`, nothing below will help
until that is fixed.

## 2. Install the Hexagon NPU runtime driver

The driver Windows ships with is not enough. You need the separate Hexagon NPU Runtime Driver,
which comes from Qualcomm Software Center and needs a free developer account.

This is the single most common reason a Snapdragon app quietly runs on the CPU. Everything
still works, so nothing complains — it is just slower and the machine feels busier than it
should.

After installing, re-run the doctor. `Hexagon NPU runtime driver` should read `ok`.

## 3. Get Whisper model assets

There are two kinds of weights here and they are not interchangeable.

**Portable ONNX exports** run on the CPU and through DirectML on the Adreno GPU. The doctor
downloads them for you:

```powershell
localscribe-doctor --fetch-models
```

It picks the size the plan chose; pass `--model base.en` to override. These land in
`models/cpu/<size>/` and are enough to make the app work end to end. They cannot run on the
NPU.

**Precompiled QNN context binaries** are the NPU path, and they are chipset-specific: a build
for Snapdragon X Elite will not load on X Plus or X2. Nothing downloads these for you. They are
not published as files anywhere — Qualcomm's Hugging Face repositories under the `qualcomm`
organisation are deprecated and now contain only a pointer to AI Hub. You export your own, which
needs a free AI Hub account:

```bash
pip install qai-hub-models
python -m qai_hub_models.models.whisper_base_en.export \
  --chipset qualcomm-snapdragon-x-elite \
  --target-runtime precompiled_qnn_onnx \
  --components HfWhisperEncoder HfWhisperDecoder
```

Lay the files out like this, where the folder name matches your chip:

```
models/
  cpu/                     <- portable, from --fetch-models
    small.en/
      encoder.onnx
      decoder.onnx
      vocab.json
  snapdragon-x-elite/      <- precompiled QNN, from AI Hub
    base.en/
      encoder.onnx
      decoder.onnx
      vocab.json
```

`vocab.json` is the same file either way and comes from the Hugging Face Whisper repo, not from
AI Hub. `--fetch-models` fetches one; copy it across.

Keeping the two apart is not tidiness. The doctor treats the presence of chipset weights as part
of deciding the NPU is usable, so a portable export dropped into the chipset folder would send
the encoder to the NPU, where it would either refuse to load or quietly relocate to the CPU.

### Version pinning

Precompiled context binaries are tied to the QAIRT and ONNX Runtime versions they were built
with. The model card lists both. If a model fails to load with a context-binary error, a
version mismatch is the first thing to check.

## 4. Optional: a local language model for the cleanup pass

Transcription works without this. Punctuation repair, glossary correction, and summaries do not.

Either backend below works; the app probes both and uses whichever answers, so you only need
one. The doctor names the one it found.

### GenieX (preferred on Snapdragon)

Qualcomm's on-device generative runtime. It runs LLMs across the Hexagon NPU, the Adreno GPU and
the Oryon cores, and serves an OpenAI-compatible API on `127.0.0.1:18181`, which is all this app
needs. See [github.com/qualcomm/GenieX](https://github.com/qualcomm/GenieX).

It is preferred here because Foundry Local has had open bugs affecting NPU generation on
Snapdragon X Elite, and a cleanup stage that crashes is worse than one that is merely slower.

Note that GenieX covers this stage only. It runs language and vision-language models and has no
speech-to-text path, so Whisper still goes through ONNX Runtime.

### Foundry Local

```powershell
winget install Microsoft.FoundryLocal
foundry service start
foundry model run qwen2.5-1.5b-instruct
```

Pass a model **alias** rather than a fully qualified id. Foundry then picks the QNN NPU build on
Snapdragon and falls back to CPU elsewhere, which is exactly the behaviour we want.

If generation crashes on the NPU, drop back to a CPU variant — the cleanup model is small enough
that the Oryon cores handle it comfortably.

## Verifying the NPU is genuinely in use

The doctor's checks confirm the pieces are installed. To prove work actually lands on the NPU:

1. Run the doctor and confirm the plan says `Encoder  Npu`.
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
