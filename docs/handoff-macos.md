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
- **The aligner scan overlaps transcription.** The scan needs only the audio, and the
  transcriber's heavy half runs off-CPU on both platforms — yet the scan used to wait for the
  transcript. Measured on a 63-second file: 41 s of scan and 12 s of transcription, formerly
  spent in sequence. Verified on the Snapdragon since: 294 s → 271 s wall on the reference
  podcast, with the two runs' span dumps identical in every word time — smaller than the Mac's
  win because there the scan was the long pole, while the NPU's transcription dominates the
  Snapdragon run and the scan already overlapped the finish stages; starting it earlier trims
  the tail that used to outlast diarization.

## Invariants, as they landed

Build native arm64 (`-r osx-arm64`; Rosetta hides the ANE the way x64 emulation hid QNN, and
the doctor warns). Encoder on the ANE, decoder on CPU/Metal — adopted with whisper.cpp rather
than reimplemented. Diarization on the CPU. Threads capped by the same plan (the battery share
applies; QoS remains untried). Model files keep their published names — whisper.cpp derives
the Core ML bundle's path from the GGML file's name, so the invariant now guards that pairing
too. The Foundry port stays dynamic. The app installs nothing uninvited: model weights
download themselves with the cost announced, but Foundry Local waits for its button.

## Next Mac build: Nemotron and the pace slider

Two changes landed on Windows in commit `bf2acc2`, and both reach this Mac through the shared
`MainViewModel` whether or not anything here is touched. Neither has run on a Mac.

**Nemotron 3 Diarization is now the only diarizer on ARM64**, and Apple silicon is ARM64.
`SpeakerEngines.For` picks by processor, so the next build diarizes with NVIDIA's end-to-end
model instead of the pyannote pipeline. On the Snapdragon it found the same speakers wherever
the pipeline was right and far fewer phantom ones where it was not (sixteen to three on the
hour-long panel, twenty-five to six on Karl's phone call), at four to eleven times the speed;
word by word it won the lines the two disputed. The evidence, the model's provenance and its
limits — eight speakers at most, no threshold to turn — are in [diarization.md](diarization.md).

**A pace slider** moves the CPU budget the plan hands every model session: Light, Balanced (the
planner's own answer, unchanged), Fast, Fastest. `MainViewModel.Pace` and `PaceDescription`
carry it and the choice persists by itself; the Avalonia window has no control for it yet.

### What happens on its own

- First launch fetches the model into `models/diarization-nemotron/` beside the others: about
  105 MB, five files from a community ONNX export, each pinned to a commit and a SHA-256
  (`SortformerModelSource`). A file that fails its pin is refused before it reaches the name
  the app loads. `--fetch-models` fetches it too.
- The pyannote models are still fetched. Renaming a speaker by voice listens with the WeSpeaker
  model, on every platform.
- A user-given speaker count is met by folding Nemotron's extra speakers, smallest first, into
  whoever they sound most like. It never splits; if fewer people were heard than the user
  names, the status line says so.
- Crosstalk badges come straight from the model: frames where two of its speakers are active.

### Verify, in this order

1. **`dotnet test`** — 674 on the Snapdragon, all passing.
2. **`--diarize-trial`** on the two fixtures that live on both machines:
   ```bash
   localscribe-doctor --models <models> --diarize-trial "Peterson debate.scrb" --transcripts
   localscribe-doctor --models <models> --diarize-trial "MM Aug 10_v1-new.scrb"
   ```
   It runs both engines, times them, and compares them frame by frame and then word by word
   through the app's own attribution. The Snapdragon's answers, to hold the Mac's against:

   | Recording | Speakers, pipeline / Nemotron | Same person, single-speaker time | Words given to the same person |
   |---|---|---|---|
   | Debate | 2 / 2 | 97.4% | 358 of 388 |
   | Podcast, Aug 10 | 5 / 5 | 100% | 1,178 of 1,178 |

   The Mac should match these closely. Exactly is not guaranteed: the int8 arithmetic can take
   a different kernel path on a different chip, which moves probabilities by tiny amounts.
3. **If the int8 graph will not load.** It should: it needs only `MatMulInteger` and
   `DynamicQuantizeLinear`, which loaded on Windows ARM64 through the same CPU kernels, and the
   `ConvInteger` gap that ruled out the aligner's quantised builds does not apply. If it does
   fail, try the full-precision graph with the doctor's `--float32`. The app does not download
   that one: fetch `step.onnx` by hand from the same commit into `models/diarization-nemotron/`
   (396 MB, SHA-256 `cd7fa5b062e0cbcd8c43e27627eb7fb305e5bce3ec4faa89d8ff8d8879bae82f`). If
   neither works, returning macOS to the pipeline is one line in `SpeakerEngines.For`.
4. **Optional: parity with the reference.** The Windows port was checked against the
   exporter's JavaScript (`diar.js` in nealcaren/local-interview-transcriber) and matched on
   every frame. To repeat it here: run `diar.js` under Node with `onnxruntime-node`, then run
   the trial with `LOCALSCRIBE_DUMP_ACTIVITY=<file>` set, which writes the raw probabilities
   as float32, eight per 10 ms frame, and compare the two files.

### Threads on Apple silicon: the one thing expected to go wrong

`DeviceProbe` reports `Environment.ProcessorCount` as the performance-core count, which on the
M2 is eight: four performance cores and four efficiency cores, counted as equals. On the
Snapdragon, where all twelve cores are alike, twelve threads already measured slower than ten —
every parallel step waits for its slowest thread — so the slider's top stop leaves two cores
free (`WorkPaces.MostThreads`). Here the slow threads are built in: anything past four lands on
an efficiency core. On battery, expect Fast (five threads) and Fastest (six) to be no faster
than Balanced (three), and possibly slower. Plugged in it is worse: the planner's own share is
two-thirds of eight, so Balanced itself is five threads and may already be past the knee.

Measure before changing anything — the doctor's `--threads` takes an exact count:

```bash
unzip -p "Peterson debate.scrb" audio.wav > debate.wav
for t in 2 3 4 5 6 8; do localscribe-doctor --models <models> --threads $t --align debate.wav | grep Took; done
for t in 2 3 4 5 6 8; do localscribe-doctor --models <models> --diarize-trial debate.wav --only nemotron --threads $t | grep Took; done
```

If four wins, the likely fix is to read the performance-core count from
`sysctl -n hw.perflevel0.physicalcpu` in `DeviceProbe` on macOS. Mind what else reads that
number: the planner's own budget is a share of it, so the change would move Balanced too — from
three threads to two on battery — and whisper.cpp's 7.4× real time was measured at three. Either
re-measure whisper.cpp at the new default, or apply the performance-core count only where the
slider's stops are computed.

The NPU power setting does nothing here: whisper.cpp reaches the Neural Engine through Core ML,
not QNN, so on the Mac Light only halves the threads. `PaceDescription` already leaves the NPU
out when the plan has no NPU stage.

### Window work in `LocalScribe.Desktop`

- **The pace control.** Bind a four-stop slider to `MainViewModel.Pace` (`WorkPace`, values
  0 to 3) and show `PaceDescription` under it, with a note that it applies from the next
  transcription or recording. Two lessons from the WinUI version: put it where a narrow window
  cannot clip it — on Windows it moved from the far right to the end of the
  make-a-transcript group for that reason — and do not take the control's first value change
  as a choice, or every launch resets the user's setting to the slider's starting position.
  The choice persists by itself under the user's local application data folder
  (`LocalScribe/work-pace.txt`).
- **The speakers flyout.** `SpeakerCountBox` allows up to 12. Bind its `Maximum` to
  `MainViewModel.MostSpeakers`, which is eight under Nemotron and ten for the pipeline.
- Nothing else in the window needs to change for Nemotron.

### What to expect

Estimates from the Snapdragon's measurements and the pipeline's ~20× real time measured here,
not measurements:

| Recording | Diarization on this Mac, battery budget |
|---|---|
| Podcast, 7 min | 5–8 s |
| Karl, 21 min | 20–25 s |
| Panel, 62 min | about a minute |

Diarization will be short enough that the fanless chassis never heats up on its account. On a
long recording the word aligner is the stage that will, as it is the long pole on Windows too.

## Still open, ranked by likelihood of mattering

0. **Nemotron and the pace slider reach the Mac on its next build, unmeasured there.** See the
   section above; it is the first thing to do.

1. ~~The Windows build has not compiled the shared-file edits.~~ Done: solution, WinUI app and
   all 605 tests pass on the laptop; the scan overlap measured 294 s → 271 s there with
   identical spans, and the diarizer cap's turn diff came back byte-identical on the podcast
   fixture, so Windows adopted the cap too.
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
