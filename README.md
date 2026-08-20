# LocalScribe

Offline transcription for Snapdragon X Windows laptops. Whisper runs on the Hexagon NPU, a
local language model cleans up the text, and nothing leaves the machine.

## Why this exists

Transcription tools that use the cloud are easy to build and awkward to trust. Transcription
tools that run locally are trustworthy and usually slow, because almost none of them touch the
NPU sitting idle in the laptop.

The guiding rule here is that work belongs on the NPU or the GPU not because those are always
faster, but because work placed there does not compete with everything else you are doing. A
transcription that finishes in four minutes without you noticing beats one that finishes in
three while the machine stutters.

## What it does

- Transcribes audio files, or listens to the microphone live
- Puts the Whisper encoder on the Hexagon NPU, falling back to the Adreno GPU, then the CPU
- Caps CPU usage so the rest of Windows stays responsive
- Repairs punctuation and casing with a local language model
- Corrects names and jargon against a glossary you supply
- Writes a summary and pulls out action items

## Getting started

```powershell
# Build native arm64. This is not optional: under x64 emulation the NPU is unreachable.
dotnet build -c Release -r win-arm64

# See what your machine has, and what is missing
dotnet run --project src/LocalScribe.Doctor -c Release -r win-arm64

# Download the missing models and install the local inference engine
dotnet run --project src/LocalScribe.Doctor -c Release -r win-arm64 -- --install
```

Without `--install` the doctor changes nothing and downloads nothing — it only reports. With it,
it fetches Whisper weights matched to your chipset and installs Foundry Local through winget.

One thing it deliberately will not do is install the Hexagon NPU driver. That is a signed kernel
driver behind a Qualcomm account, so it needs you; the doctor names it and links you to it.
Everything else on the list it handles.

Setup reaches the network, to Hugging Face and Microsoft. Audio never does.

## Layout

| Project | What it is |
| --- | --- |
| `LocalScribe.Core` | Hardware policy, audio maths, stitching, orchestration. No dependencies, runs anywhere. |
| `LocalScribe.Onnx` | ONNX Runtime sessions, execution-provider wiring, hardware probing. |
| `LocalScribe.Doctor` | Console tool that reports what the machine can do and what the app will do with it. |
| `LocalScribe.App` | The WinUI 3 window. |

The split is deliberate. Everything that involves a decision lives in `Core`, which has no
dependencies and therefore builds and tests on any machine — including CI runners with no
Qualcomm hardware anywhere near them. The parts that genuinely need Snapdragon silicon are thin
by design, so there is as little untestable code as possible.

```powershell
dotnet test
```

`LocalScribe.App` is intentionally left out of `LocalScribe.sln`. WinUI targets cannot be
restored or built on Linux, so including it would break `dotnet build` for anyone not on
Windows. CI builds it as a separate job; build it directly when you need it:

```powershell
dotnet build src/LocalScribe.App/LocalScribe.App.csproj -c Release -r win-arm64
```

## How the placement decision works

`AcceleratorPlanner` takes a set of observed capabilities and returns a plan: which processor
runs each stage, how many CPU threads we may take, and which Whisper size to load. It touches
no hardware, so the whole policy is unit tested.

The rules it encodes:

- **Encoder to the NPU when possible.** Fixed-shape, compute-heavy, and it is the expensive half.
- **Decoder stays on the CPU unless the NPU took the encoder.** It emits many small steps, and
  dispatch overhead outweighs any GPU gain.
- **CPU threads are capped, not maximised.** Two-thirds of the performance cores on mains power,
  less on battery, and only two threads when the models are offloaded.
- **Model size follows the placement.** Offloaded work is nearly free from the user's point of
  view, so the NPU path can afford a larger model than the CPU path would tolerate. Live
  transcription steps down again, because falling behind the speaker is unrecoverable.

## Status

The core library and its 135 tests are verified. The doctor tool runs.

**The WinUI app and the ONNX layer have not been run on real Snapdragon hardware.** They were
written on Linux, where Windows targets cannot be compiled and no NPU exists. Expect to fix
things on first run. The most likely rough edge is the decoder's input signature: exports differ
between Hugging Face Optimum and Qualcomm AI Hub, and the binding here discovers input names but
assumes a shape contract. The doctor prints the discovered signature so a mismatch is obvious.

## Known limitations

- **No speaker labels.** Diarization needs a separate model; pyannote is the usual choice.
- **Greedy decoding only.** No beam search, so accuracy is a little below reference Whisper.
- **No key/value cache in the decode loop.** The decoder re-runs over the whole prefix each
  step. Correct against every export, slower than it needs to be against any particular one.
- **English-focused.** The tokenizer handles multilingual models, but model selection assumes
  the `.en` variants.
