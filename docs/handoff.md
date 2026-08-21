# Handoff notes

Written across cloud sessions, for whoever picks this up on the actual Snapdragon.

## Where things stand

The core library builds and its 144 tests pass. `LocalScribe.App` compiles for win-arm64 on a
Windows CI runner, clean under `TreatWarningsAsErrors`. That is the whole of what has been
proven.

**Nothing has ever run on Snapdragon hardware.** The sessions that wrote this were on Linux x64,
where no NPU exists and Windows binaries cannot be executed. `LocalScribe.Onnx` has never run a
model, and the app has never been launched. Compiling is not evidence of working — the runtime
bugs in this codebase are the kind that only appear under a real start/stop cycle.

Treat the unverified parts as a first draft that happens to compile.

## First things to run

```powershell
dotnet build -c Release -r win-arm64
dotnet test
dotnet build src/LocalScribe.App/LocalScribe.App.csproj -c Release -r win-arm64
```

Then launch the app. Setup opens by itself when anything is missing, lists what it found, and
offers to download the rest:

```powershell
dotnet run --project src/LocalScribe.App -c Release -r win-arm64
```

**There is no headless path any more.** `LocalScribe.Doctor` was removed when setup moved into
the app, so hardware reporting and provisioning now require launching the GUI. That is a real
loss for CI, for support requests where you want someone to paste terminal output, and for
diagnosing a machine over SSH. If you want it back, the logic all lives in `SetupViewModel` and
`Core/Provisioning`, and a console front end over them is a small job.

## What is most likely to be wrong

Ranked by how likely it is to bite, worst first.

### 1. The Hugging Face repository names are guesses

`HuggingFaceCatalog.RepositoriesFor` lists repositories like `qualcomm/Whisper-Base-En`. Only
`qualcomm/Whisper-Tiny` was confirmed to exist. The rest were inferred from a naming pattern and
never checked, because Hugging Face was blocked from the cloud container.

If setup reports "no such repository" for every entry, this is why. Fix it by browsing
`https://huggingface.co/qualcomm` and correcting the list. The selection logic underneath is
tested and should be fine once the names are right.

Note the wording. A repository that does not exist and a network that cannot be reached used to
produce the same "not found or empty" message, which sent you to rewrite a repository list when
your connection was the problem. They are now separated: "no such repository" means Hugging Face
answered, and a message about your internet connection means it did not. Believe the one you
get — on a work network, a proxy blocking `huggingface.co` produces the second.

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

### 5. Performance cores are counted wrong

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

**Setup never installs the Hexagon driver.** It is a signed kernel driver behind an account
wall. Automating it would produce a tool that reports success and leaves the app silently
CPU-bound.

**`LocalScribe.Core` has no external dependencies.** That is what lets the policy and the audio
maths be tested anywhere. Adding a package reference there gives up more than it looks like.

## Things deliberately not attempted

- **The llama.cpp Hexagon backend.** Requires disabling Secure Boot to load test-signed driver
  libraries, and still tends to lose to the CPU on small models. Adreno OpenCL is the sane GPU
  option if you want one.
- **Speaker labels.** Needs a separate diarization model; pyannote is the usual choice.
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
| Why does setup say a repository is missing? | `Core/Provisioning/HuggingFaceCatalog.cs` |
| What does the app do on first launch? | `App/SetupViewModel.cs` |
| How does a provider actually get registered? | `Onnx/OnnxSessionFactory.cs` |

## A note on the tests

The 144 tests cover decisions, not plumbing. The FFT is checked against the definition of the
DFT rather than against a recorded snapshot of its own output, and the mel filterbank is checked
by confirming that rising frequencies land in rising bands.

That matters because signal-processing bugs here do not crash. A wrong filterbank produces a
model that runs happily and emits fluent nonsense, which is a far harder thing to notice than an
exception. If you change anything in `LogMelSpectrogram`, keep those tests honest.
