# macOS port — handoff notes

Written on the Windows machine, for whoever builds this on an Apple-silicon Mac. Nothing here
has ever run on a Mac. Everything below is either portable by construction, verified only on
Windows, or a recommendation with the reasoning attached — each is marked as what it is.

The goal is the same app: offline transcription, encoder on the machine's NPU, everything else
on a capped CPU, nothing leaving the machine. The Mac translation of "Hexagon NPU" is the Apple
Neural Engine, and the happy accident of this codebase is that the invariants were written
against the *shape* of that hardware split, not against Qualcomm.

## What ports untouched

This is most of the program, and it is not luck — `LocalScribe.Core` taking no external
dependencies was the design decision that makes a second platform cheap.

- **`LocalScribe.Core`** — all of the policy: the global banded alignment
  (`GlobalCtcAlignment`), anchor pruning (`CredibleAnchors`), text likeness, word-level speaker
  attribution, unfinished-sentence repair, repeat trimming, the `.scrb` archive format (a zip;
  byte-portable), formatting, crowding reports. Its 578 tests should pass on the Mac on day
  one, and running them is the first thing to do.
- **`LocalScribe.Onnx`** — the scan windows and margins, the two-probe stride measurement, the
  twin-trial support, `AlignAll`. ONNX Runtime ships osx-arm64 in the same NuGet package. The
  **aligner (MMS CTC) ran on the CPU on Windows too**, so word timing needs no Apple-specific
  model at all.
- **`LocalScribe.Diarization`** — sherpa-onnx publishes osx-arm64 natives in the same package.
  The models (pyannote segmentation, WeSpeaker embedding) are platform-neutral ONNX. The
  "diarization runs on the CPU" invariant carries unchanged, for the same reason: sherpa's
  bundled runtime has no NPU provider on any platform.
- **`LocalScribe.Doctor`** — a console app with no Windows dependencies worth keeping. It is
  also the acceptance harness for the whole port: `--align`, `--check-words`, and `--replay`
  are how every timing claim on Windows was proven, and they prove the Mac the same way.

## What must be replaced

Everything Windows-specific lives in `LocalScribe.App`, on purpose.

| Windows | macOS replacement | Confidence |
| --- | --- | --- |
| WinUI 3 window | Avalonia (see below) | recommendation |
| Whisper via ONNX Runtime QNN | whisper.cpp via the whisper.net binding, Metal + Core ML encoder | recommendation |
| NAudio playback/capture | miniaudio (one C file, P/Invoke) | recommendation |
| Media Foundation decode | miniaudio for wav/mp3/flac; `afconvert` for m4a/aac | recommendation |
| Foundry Local (Windows) | Foundry Local (macOS, Homebrew) | verified to exist, never run here |
| HKCU file association script | `.app` bundle `CFBundleDocumentTypes` + UTI | standard, unverified |

## The transcriber: recommendation and reasoning

**Use whisper.cpp through the whisper.net .NET binding, with the Metal backend and the Core ML
encoder enabled, running Whisper large-v3-turbo.** The reasoning:

- whisper.cpp's Core ML path runs **the encoder on the Apple Neural Engine and keeps the
  decoder on CPU/Metal** — which is this app's architecture invariant, arrived at
  independently, for the same reason (per-step dispatch overhead outweighs the decode compute).
  The port keeps the invariant by adopting the tool rather than by reimplementing it.
- whisper.net is a maintained .NET binding with runtime packages that carry the Core ML and
  Metal builds, so the app stays a .NET program end to end.
- The alternative with the *least code churn* — keeping `WhisperOnnxTranscriber` and swapping
  QNN for ONNX Runtime's Core ML execution provider — is also the least likely to perform:
  Core ML EP coverage of Whisper's graphs is historically partial, and what falls off the EP
  runs on CPU silently. Try it only if curious, and judge it with the doctor, never by feel.
- The *fastest* option on Apple hardware is WhisperKit (Core ML models tuned by ex-Apple ANE
  engineers), but it is a Swift package; bridging it into a .NET app costs an interop layer.
  Revisit only if whisper.net's numbers disappoint.

Whatever engine wins, it must produce the same `TranscriptSegment` stream (text, stamps,
probabilities) the pipeline already consumes — and its stamps get the same distrust. Assume the
sawtooth drift and final-window inflation are Whisper properties, not Windows properties: the
anchor pruning and the 36-second corridor band exist because stamps measured up to 17.5 seconds
late. Do not narrow anything because a first Mac recording looks clean.

Model files: whisper.cpp uses GGML/Core ML model files, not the AI Hub ONNX layout. Extend
`localscribe-model.json` with the new roles rather than renaming anything — the never-rename
invariant is about sidecar references and applies to Core ML `.mlmodelc` bundles just as much.

## The other models

- **Aligner**: the same MMS CTC ONNX model, CPU, unchanged. Copy the fetch logic; it has no
  Qualcomm in it.
- **Diarization**: the same pyannote + WeSpeaker ONNX files, CPU, via sherpa-onnx osx-arm64.
  The 'voices' clustering path is the active one; `active-diarizer.txt` carries over.
- **Cleanup**: Foundry Local ships for Apple silicon (`brew tap microsoft/foundrylocal &&
  brew install foundrylocal`) with GPU acceleration. The dynamic-port invariant is unchanged:
  ask `foundry service status`, never hard-code. If Foundry disappoints on the Mac, Ollama
  exposes the same OpenAI-shaped HTTP surface and the refiner should treat that as a
  configuration difference, not a code path.

  The glossary dialog's "Download and start it" button drives
  `FoundryLocalManager` — install if missing, start the service, download the default model,
  reconnect, and rerun cleanup on the open transcript. The manager already branches to
  Homebrew on macOS (`OperatingSystem.IsMacOS()` in `InstallAsync`), but that branch has
  never executed; verify it the first time the button is pressed on a Mac.

## The UI: recommendation and reasoning

**Avalonia.** Three reasons:

- `MainViewModel` deliberately holds no WinUI types — it was built to be exercised without a
  UI host. Avalonia is the framework closest to the XAML model the window code already speaks,
  so the port is a translation, not a rewrite.
- It runs on Windows too. If the Mac window reaches parity, there is a future where one window
  serves both platforms and WinUI retires.
- The alternatives are worse fits: MAUI's macOS story is Catalyst (an iPad app in a window),
  and a native SwiftUI shell over a .NET core means two languages forever.

Port the window *behaviors*, not just the layout — several of them are paid-for lessons:

- **Playback position updates are coalesced, latest-wins** (`OnPlaybackPosition`): at most one
  update in flight, reading the newest position when it runs. Twenty queued updates a second
  against slow repaints made the marker replay the past.
- **The marker rules**: a word covering the instant beats paragraph bounds; else the most
  recently begun word; repaint every realised paragraph rather than tracking which to clear.
  Each clause exists because its absence was a visible bug.
- **The close gate**: refuse the close over unsaved work, ask Save / Discard / Cancel; a
  cancelled save picker is not consent to lose anything. On macOS this is
  `NSWindowDelegate.windowShouldClose` / Avalonia's `Closing` event with the same shape.
- **The diagnostics**: the temp-folder dumps (`localscribe-input.txt`, `localscribe-spans.txt`,
  `localscribe-alignment.txt`, `localscribe-clock.txt`) are the reason the sync campaign ended.
  Keep them; they cost nothing and they turn "it feels off" into a file.
- The app accepts a file path as a launch argument; that is how debugging drives it headlessly.

## Audio in and out

- **Playback and capture**: miniaudio — a single C file, compiled once for arm64, P/Invoked.
  It replaces both NAudio playback (`TranscriptPlayer`) and WASAPI capture
  (`MicrophoneCapture`). Keep the player's design: play the decoded sample array the
  transcriber was given, never re-decode; report the device's position, not the read position.
- **Decode**: miniaudio decodes wav/mp3/flac natively but **not m4a/aac**, which podcasts
  often are. Rather than bundling ffmpeg, shell out to `afconvert` — it ships with macOS —
  to produce a 16 kHz mono WAV, then read that. Zero dependencies, and the pipeline only ever
  sees PCM it decoded itself.
- **Microphone permission**: the `.app` bundle needs `NSMicrophoneUsageDescription` in its
  Info.plist or capture fails silently — the exact class of failure this project hates, so
  make the doctor check it.

## Invariants, translated

Do not relax these; each was earned.

- **Build native arm64** (`-r osx-arm64`). An x64 build under Rosetta cannot reach the ANE and
  the symptoms will look like a missing framework — the same trap as x64-under-emulation
  hiding the QNN provider, wearing a different coat.
- **The decoder stays on CPU/Metal; only the encoder goes to the ANE.** Adopted for free with
  whisper.cpp's Core ML split.
- **Diarization on the CPU.** Same runtime, same reason.
- **CPU threads are capped so the machine stays usable.** On macOS, also set QoS (utility) on
  the worker threads; Apple silicon schedules efficiency cores by QoS, and this is the
  product requirement expressed in the platform's own vocabulary.
- **Never rename downloaded model files. Never hard-code the Foundry port. Never auto-install
  anything** — no driver on macOS to resist, but the same restraint applies to models and
  helper tools: report, do not install.

## Packaging and the .scrb association

- `dotnet publish -c Release -r osx-arm64` with `SelfContained=true` (carry the Windows
  decision: the app is a folder someone is handed and must not care what .NET the machine
  has), wrapped into a `LocalScribe.app` bundle.
- Declare `.scrb` (and legacy `.lscribe`) in `CFBundleDocumentTypes` with an exported UTI —
  the bundle-native equivalent of `tools/register-scrb.ps1`, and on macOS it is legitimate
  for the bundle to declare it (the user installs the app; the declaration ships with it).
- The icon masters in `tools/make-icon.py` and `tools/make-scrb-icon.py` already draw at
  1024; add an `--iconset` mode emitting the sizes `iconutil` wants and produce `.icns` for
  both the app and the document type on the Mac.

## Build order on the Mac

Prove things in the order they were proven here — smallest first, measurement before UI.

```bash
dotnet build -c Release          # the solution: Core, Onnx, Diarization, Doctor
dotnet test                      # 578 tests; all should pass before anything else is touched
dotnet run --project src/LocalScribe.Doctor -c Release -r osx-arm64
```

Then, in order:

1. **Doctor + aligner on CPU**: fetch the alignment model, `--align` a known WAV, confirm the
   20.00 ms stride and a clean read-back. This validates ONNX Runtime on osx-arm64 with zero
   new code.
2. **Diarization**: `--diarize` the same WAV; sherpa-onnx natives are the risk here.
3. **Transcription**: whisper.net + Metal + Core ML encoder behind the transcriber seam;
   `--transcribe`, then `--check-words` on a saved archive from the Windows machine — the
   archives are portable, and a Windows-made `.scrb` checking out on the Mac is the port's
   first end-to-end proof.
4. **Cleanup**: Foundry Local via brew; the refiner over the dynamic port.
5. **The window**: Avalonia, last, once every number under it is already proven.

## Most likely to be wrong, ranked

1. **whisper.net's Core ML runtime on large-v3-turbo.** Binding versions, Core ML model
   generation, and quantization choices all have sharp edges. If the encoder quietly falls
   back to CPU the app still works and merely feels busy — which is why step 3 must measure
   (encoder time per 30 s window) rather than trust.
2. **Whisper stamp behavior differing in *degree*.** The corridor band and anchor pruning were
   sized against stamps measured 17.5 s late. A different engine build may lie differently.
   `--check-words` plus a fresh-slice `--align` read-back (the non-circular check) is the
   verdict, per `docs/handoff.md`'s closing note on tests measuring decisions, not snapshots.
3. **miniaudio capture defaults** (sample rate, channel negotiation) differing from WASAPI's;
   the live path assumes 16 kHz mono reaches it.
4. **Bundle/notarization friction** — unsigned apps and mic permission dialogs interact
   badly; a `.app` that was never notarized warns on first launch. Cosmetic, but decide
   early whether this Mac build is personal-use (skip notarization) or distributable.
5. **Foundry Local model availability on macOS** differing from Windows; the cleanup already
   degrades gracefully when no model is found, so this costs polish, not correctness.
