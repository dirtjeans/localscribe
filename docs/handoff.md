# Handoff notes

Written at the end of a cloud session, for whoever picks this up on the actual Snapdragon.

## Where things stand

Three commits are on `main`. The core library builds and its 158 tests pass. The doctor tool runs
end to end. That is the whole of what has been proven.

**Nothing has ever run on Snapdragon hardware.** The cloud session that wrote this was on Linux
x64, where Windows targets cannot compile and no NPU exists. So `LocalScribe.App` has never been
compiled at all, and `LocalScribe.Onnx` has compiled but never executed a model.

Treat the unverified parts as a first draft that happens to typecheck.

## First things to run

```powershell
dotnet build -c Release -r win-arm64
dotnet test
dotnet run --project src/LocalScribe.Doctor -c Release -r win-arm64
```

The first command is the one most likely to fail, because the WinUI project has never been
built. Expect XAML and package problems there. The core library and doctor should be clean.

Then, to fetch models:

```powershell
dotnet run --project src/LocalScribe.Doctor -c Release -r win-arm64 -- --install
```

## What is most likely to be wrong

Ranked by how likely it is to bite, worst first.

### 1. The Hugging Face repository names are guesses

`HuggingFaceCatalog.RepositoriesFor` lists repositories like `qualcomm/Whisper-Base-En`. Only
`qualcomm/Whisper-Tiny` was confirmed to exist. The rest were inferred from a naming pattern and
never checked, because Hugging Face was blocked from the cloud container.

If `--install` reports "not found or empty" for every repository, this is why. Fix it by
browsing `https://huggingface.co/qualcomm` and correcting the list. The selection logic
underneath is tested and should be fine once the names are right.

### 2. The decoder's input signature

`WhisperOnnxTranscriber.DecodeGreedily` assumes the decoder takes `input_ids` and
`encoder_hidden_states` and returns logits shaped (batch, sequence, vocabulary). Input *names*
are discovered from the model metadata, but the *shape contract* is assumed.

Qualcomm AI Hub exports may want fixed-length inputs or explicit key/value cache tensors. If so,
this needs a different binding. Print `session.InputMetadata` first and work from what is
actually there.

### 3. The Hexagon driver check is weak

`DeviceProbe.HasHexagonRuntime` just looks for `QnnHtp.dll` beside the binary or in System32.
That is a proxy for "the QNN package is present", not proof the driver is installed and working.

The honest test is to load a model with `StrictProviderCheck` on and see whether it throws.

### 4. Silent CPU fallback

This is the failure this whole project is arranged against, and it is invisible by construction:
everything works, just slowly.

To check properly, open Task Manager, Performance tab, and watch the NPU graph during a
transcription. Flat NPU with busy CPU means fallback. `ExecutionPlan.StrictProviderCheck` sets
`session.disable_cpu_ep_fallback`, which converts the silence into an exception.

### 5. Two ONNX Runtimes in one process

sherpa-onnx bundles its own native ONNX Runtime; Whisper uses the QNN build. Nothing has ever
loaded both at once. If the app crashes at startup on real hardware and the doctor does not,
this is the first suspect — the doctor only touches sherpa when installing models.

### 6. Performance cores are counted wrong

`DeviceProbe` sets `PerformanceCoreCount` from `Environment.ProcessorCount`, which is *all*
logical processors. On Snapdragon X Elite every core is an Oryon performance core, so the number
happens to be right. On anything heterogeneous it is wrong and the CPU budget will be too
generous.

## Decisions worth not undoing

These look like things to tidy up. They are not.

**The decoder stays on the CPU even when the NPU has the encoder.** The decoder issues many
small steps, and per-call dispatch overhead beats the compute saved. Moving it to an accelerator
will probably make things slower.

**Downloaded model files keep their published names.** ONNX models above 2 GB reference their
weight sidecars by name from inside the graph. Renaming to tidy fixed names breaks them.
`localscribe-model.json` records which file is which instead.

**The Foundry Local port is never hard-coded.** It binds a dynamic loopback port. A fixed port
makes a healthy service look absent, and the cleanup pass then gets skipped in silence.

**CPU threads are capped rather than maximised.** Two-thirds of cores on mains, less on battery,
two when the models are offloaded. This is the "do not make the machine feel slow" requirement,
not an oversight.

**The doctor never installs the Hexagon driver.** It is a signed kernel driver behind an account
wall. Automating it would produce a tool that reports success and leaves the app silently
CPU-bound.

**`LocalScribe.Core` has no external dependencies.** That is what lets the policy and the audio
maths be tested anywhere. Adding a package reference there gives up more than it looks like.

## Evaluated and not taken

### WhisperX

Rejected on hardware grounds, not quality. It transcribes through faster-whisper on CTranslate2,
which has no Windows ARM64 wheels; the one person who hand-compiled it reports cpuinfo failing
to detect the X Elite and the result running very slowly. The library exists to be fast, so a
slow build of it is pointless. Its wav2vec2 forced alignment is still worth stealing later —
see `docs/diarization.md`.

### VibeVoice-ASR — worth a spike, first thing

This one may well be better than what is built here, and it deserves a real look on hardware.

Microsoft ships [VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp), an official MIT-licensed
C++/ggml runtime for VibeVoice-ASR-BitNet. The 7B language model is replaced with a 1.5B one and
quantized hard, taking the package from 4.62 GB to 1.58 GB. Reported real-time factor is
0.42–1.98 depending on threads, and 1.6–2.3× faster than whisper.cpp at comparable size. It has
been tested on Apple M4 through ARM NEON, which is the best available evidence that Oryon cores
would work.

The full 7B VibeVoice-ASR does transcription, diarization, and timestamps jointly in one model.
If the BitNet runtime kept that, it replaces both Whisper and pyannote here and removes the
boundary-reconciliation seam entirely, which is this design's weakest point.

Three questions decide it, and all three need the actual laptop:

1. **Does it build for ARM64 Windows?** The documented Windows path is MinGW-w64, and MSVC is
   explicitly unsupported. Windows on ARM is not listed as tested.
2. **What is the real RTF on Oryon cores?** Under 1.0 makes the NPU largely irrelevant for batch
   work.
3. **Does the C++ runtime emit speaker labels and timestamps?** The repository does not advertise
   them. If it does not, it replaces Whisper but not diarization.

Answer those before writing any integration code. A MinGW-built DLL called from .NET through
P/Invoke is the likely shape if it goes ahead, and that is awkward enough that it should not be
started on a guess.

## Things deliberately not attempted

- **The llama.cpp Hexagon backend.** Requires disabling Secure Boot to load test-signed driver
  libraries, and still tends to lose to the CPU on small models. Adreno OpenCL is the sane GPU
  option if you want one.
- **Word-level timestamps.** wav2vec2 forced alignment, exported to ONNX. This is the single
  biggest quality win still on the table: it is what would let a segment be split at a speaker
  change instead of flagged as uncertain. See `docs/diarization.md`.
- **Beam search.** Greedy decoding only, so accuracy sits a little below reference Whisper.
- **A key/value cache in the decode loop.** The decoder re-runs over the whole prefix each step.
  Correct against every export, slower than necessary against any particular one. Worth adding
  once you know which export you are actually targeting.

## Where the logic lives

| Question | File |
| --- | --- |
| Why did it pick the NPU, GPU, or CPU? | `Core/Hardware/AcceleratorPlanner.cs` |
| How does audio become model input? | `Core/Audio/LogMelSpectrogram.cs` |
| Why is this word duplicated or missing? | `Core/Transcription/TranscriptStitcher.cs` |
| Why does live text keep changing? | `Core/Pipeline/LiveTranscriptionSession.cs` |
| What gets installed, and what does not? | `Core/Provisioning/Provisioner.cs` |
| Why is this line attributed to the wrong person? | `Core/Diarization/SpeakerAssigner.cs` |
| How does a provider actually get registered? | `Onnx/OnnxSessionFactory.cs` |

## A note on the tests

The 158 tests cover decisions, not plumbing. The FFT is checked against the definition of the
DFT rather than against a recorded snapshot of its own output, and the mel filterbank is checked
by confirming that rising frequencies land in rising bands.

That matters because signal-processing bugs here do not crash. A wrong filterbank produces a
model that runs happily and emits fluent nonsense, which is a far harder thing to notice than an
exception. If you change anything in `LogMelSpectrogram`, keep those tests honest.
