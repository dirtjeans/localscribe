# macOS — handoff notes

Originally written on the Windows machine as a plan, when nothing had run on a Mac. That era
is over: the port ran, on an Apple-silicon Mac (M2, 16 GB), on real recordings, and the plan's
"most likely to be wrong" list turned out to be right about what went wrong. This version says
what is proven, how it was proven, and what is still open — the same order as
[handoff.md](handoff.md), because the *how* is the part a newcomer needs most.

## Where things stand

`LocalScribe.Desktop` (Avalonia, in the solution — it restores everywhere, unlike the WinUI
app) drives the same `MainViewModel` the Windows window compiles, from the same file. It
transcribes with whisper.cpp through the whisper.net binding — encoder on the Apple Neural
Engine via Core ML, decoder on CPU/Metal — times words with the same MMS aligner on the CPU,
attributes speakers with the same models, plays back through miniaudio with the word-level
highlight, records from the microphone, saves and opens the same byte-portable `.scrb`
archives, and provisions its own models on first launch (~2.8 GiB, narrated, into
`~/Library/Application Support/LocalScribe/models` when bundled). `tools/make-macos-app.sh`
builds `build/LocalScribe.app` (`--install` copies it to /Applications); the icons are the
Windows masters rendered through the scripts' `--iconset` mode.

## How it was proven

The Windows instruments carried over unchanged, and they were the acceptance harness:

- **`dotnet test`** — the core suite passes on osx-arm64; it was the first thing run.
- **`--align`** measured the 20.00 ms stride with a clean letter read-back — ONNX Runtime on
  osx-arm64 validated with zero new code.
- **`--diarize`** produced stable turns at ~20× real time, and its turn list is byte-identical
  across the ONNX Runtime 1.22 → 1.29 bump, which is how that bump was judged safe.
- **`--check-words` on a Windows-made archive** — the debate fixture: 39 of 39 segments timed,
  drift +0.00 in every fifth. The archives are byte-portable, and this was the port's first
  end-to-end proof.
- **The app's own diagnostics** write to the temp dir exactly as on Windows —
  `localscribe-input/spans/alignment/clock.txt`, plus a new one, `localscribe-errors.txt`,
  where failures keep their stacks (the status line only has room for a verdict). It exists
  because a phantom failure was blamed on the user as "Cancelled." and the evidence was gone.

## What the plan got right, and where it was wrong

- **whisper.cpp + Core ML was the right engine** — large-v3-turbo at 7.4× real time on
  battery with three capped threads — but the plan's #1 fear (silent CPU fallback) happened
  immediately, twice over: whisper.net only reports segment probabilities when asked
  (`WithProbabilities()`, or every segment reads as guesswork and the transcript blanks), and
  **Whisper.net.Runtime.CoreML ships libwhisper.dylib with only its CI machine's rpaths**
  (1.8.1 and 1.9.0 both), so the Core ML runtime can never load as published.
  `Directory.Build.targets` patches `@loader_path` onto the output copy; the bundle script
  patches what it ships. The permanent instrument: the loaded runtime's name is in every
  engine description — "(CoreML runtime)" or "(Cpu runtime)" — so the fallback can never be
  silent again. Measure, never trust: this is the doctor's `--engine whispercpp --transcribe`.
- **Diarization needed no sherpa** — the repo's diarizer runs pyannote + WeSpeaker through
  ONNX Runtime directly, and osx-arm64 ships in the same package.
- **miniaudio, afconvert and avconvert did their jobs** — one C file compiled on demand
  (`src/LocalScribe.Desktop/native/`), the player reporting one period behind the callback
  (the clock diagnostic's gap column is the judge), capture at 16 kHz with the permission
  string in the bundle's Info.plist.
- **The engine seam is a constructor argument**: `MainViewModel(modelRoot, openTranscriber)`.
  Left null, the ONNX engine loads as it always has — the WinUI app is untouched.

## Paid for on the Mac, owed to both platforms

- **`ResilientTranscriber`** wraps whatever engine opens: an engine failure mid-recording
  rebuilds it and retries the window it failed on — the pipeline still holds that window, so
  "continue where it left off" is literal. A real cancellation passes through; a window that
  fails a fresh engine too fails honestly.
- **A cleanup model slower than HttpClient's 180 s reported itself as a cancellation**, and
  the refiner's stop-button guard rethrew it, taking the finished transcript down as
  "Cancelled." The guard now checks whether the token was actually cancelled; a timeout is
  one failed window, kept raw and said so. Windows has the same exposure with a slow GenieX.
- **Automatic speaker labels renumber by first appearance** (`SpeakerLabels`) — cluster order
  is not speaking order, and a transcript that opens with "Speaker 2" reads as a bug even
  when the separation is right. Runs before the aligned-times table is keyed, the same law
  the crosstalk marks obey; never touches a label once a user has renamed anyone.
- The microphone start is guarded in the shared view model — on macOS that is where a denied
  permission surfaces, and unguarded it took the process down from a click.

## Invariants, as they landed

Build native arm64 (`-r osx-arm64`; Rosetta hides the ANE the way x64 emulation hid QNN, and
the doctor warns). Encoder on the ANE, decoder on CPU/Metal — adopted with whisper.cpp rather
than reimplemented. Diarization on the CPU. Threads capped by the same plan (the battery share
applies; QoS remains untried). Model files keep their published names — whisper.cpp derives
the Core ML bundle's path from the GGML file's name, so the invariant now guards that pairing
too. The Foundry port stays dynamic. The app installs nothing uninvited: model weights
download themselves with the cost announced, but Foundry Local waits for its button.

## Still open, ranked by likelihood of mattering

1. **The Windows build has not compiled the shared-file edits.** They are additive and
   C# 12-clean, but the WinUI app could not be built from the Mac. First session on the
   laptop: `dotnet build`, then normalize the line-ending churn this sync left behind.
2. **Notarization is undecided.** The bundle is ad-hoc signed — fine for this machine,
   warning-laden for anyone else's. Decide when there is a second Mac in the picture.
3. **Finder's open-with does not reach the window.** macOS delivers opened documents as Apple
   events, not argv; the `.scrb` association is declared and the icon shows, but
   double-clicking an archive needs Avalonia's activation wiring. The launch-argument path
   works and is how debugging drives the app headlessly.
4. **The whisper.net defects deserve upstream reports** — the rpath packaging bug and the
   probabilities default. Until then the workarounds are load-bearing and commented.
5. **Glossary and summary UI** exist on Windows only; the refiner underneath is shared and
   already runs (this Mac answers with an OpenAI-compatible local model).
