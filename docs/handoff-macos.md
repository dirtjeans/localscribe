# macOS — handoff notes

Originally written on the Windows machine as a plan, when nothing had run on a Mac. That era
is over: the port ran, on an Apple-silicon Mac (M2, 16 GB), on real recordings, and the plan's
"most likely to be wrong" list turned out to be right about what went wrong. This version says
what is proven, how it was proven, and what is still open — the same order as
[handoff.md](handoff.md), because the *how* is the part a newcomer needs most.

Every model measured for a stage of the pipeline — what was kept, what was rejected, and the
numbers that decided it — is in [model-trials.md](model-trials.md).

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

Measured on the M2, plugged in (2026-10-02), on the debate fixture — and the prediction was
half right, which is why the measurement is the law here. Nemotron's knee is exactly the four
performance cores: 1.9 s at four threads, 2.5 s at six, 3.1 s at eight. But the aligner —
twenty times the wall clock — kept improving through six (61.5 s at two, 48.7 s at four,
45.1 s at six, 47.9 s at eight): its long uniform matmuls take net help from the efficiency
cores. The slider's top stop, `MostThreads(8)` = 6, is already the aligner's optimum, and
computing the stops from the performance-core count would have traded 0.6 s of diarization
for 3.6 s of alignment. **No change was made**: the stops stand as the flat count computes
them. The one gap is battery — this sweep ran on mains, and with Balanced at three threads
and the cluster declocked, the knee could sit lower there. If unplugged runs ever feel wrong,
repeat the sweep on battery before touching anything; the fix, if it is ever earned, goes
only where the slider's stops are computed, since the planner's own budget underwrote
whisper.cpp's measured 7.4×.

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

## Next Windows build: what changed on the Mac (2026-10-03)

Everything below landed in shared code — `MainViewModel`, `LocalScribe.Core`, `LocalScribe.Onnx` —
so the next Windows build gets it whether or not anything there is touched. All of it was
measured on the M2 (16 GB), where the bottleneck was memory and the CPU did the alignment
work. **Results may differ on the Snapdragon.** Its NPU carries the whole transcription, it has
12 alike cores rather than 4+4, and if it has 32 GB the memory pressure that drove most of
this may never arise. Re-measure there before trusting any number in this section; the
instruments to do it with are named against each item.

### Fixes that apply everywhere

- **Cleanup could write a window twice, and that broke word sync for the whole file.**
  `CleanedTextAlignment` matched the cleaned reply back onto segments with a four-word
  lookahead; when a model dropped or reworded more than four leading words, the walk never
  found its place, poured the whole cleaned window into one segment, and the rest kept their
  raw text. The duplicate made the global alignment pass either fail outright (podcast: 0 of
  109 segments placed) or squeeze words onto the wrong audio while still counting them as
  "measured" (an hour-long file: sync lost at 22:25 and never recovered). The walk now
  resynchronises on a run of three matching words, and `TryApply` refuses any mapping that
  keeps more raw text than a filler segment's worth; the refiner counts that window as kept
  raw, so the cleanup notice reports it. The 7B model triggered it often; the smaller models
  Windows has used may have triggered it rarely, but nothing prevented it. Tests:
  `CleanedTextAlignmentTests`. **Verify:** `--check-words` on a fresh Windows transcription
  with cleanup on; and scan `localscribe-input.txt` for any segment holding far more words
  than its stamps could carry.
- **The aligner's model is released the moment the scan ends.** Everything after — placing
  words, read-backs, the duplicate and unheard-line trials — works from the scores alone, but
  the session stayed loaded through cleanup: ONNX Runtime's fp32 repack of the weights
  (1.1 GB) plus a working arena that grows and never shrinks (1.6 GB), found by
  `malloc_history`. App memory during cleanup went from 2.8 GB to about 136 MB.
  **Verify:** Task Manager's memory column through a run; it should drop when the scan ends.
- **Speaker labels renumber by first appearance; HttpClient timeouts no longer pass as
  cancellations; a failed engine restarts and retries its window** (all earlier this month,
  listed in the section above).
- **"Measured N of N" is not proof of sync.** Duplicated text squeezes placement rather than
  stopping it, so a broken transcript can report every segment measured. Only a re-listen —
  `--check-words` — or a scan for duplicated text settles it.

### Behaviour that now differs by platform

- **Cleanup is optional** (`MainViewModel.CleanupEnabled`, persisted in `cleanup.txt` beside
  `work-pace.txt`). Measured on the Mac with Whisper large-v3-turbo, cleanup changed about one
  token in eleven — mostly a comma made a colon or a missing capital — for roughly half the
  recording's length in time and 7 GB of memory with a 7B model. **Off by default on macOS,
  and since 2026-10-04 on Windows too** (see the Snapdragon results below). With it off,
  Foundry is never started and no model is loaded. The WinUI window has no control for it
  yet; it reads the same preference file. On the Snapdragon, where cleanup runs on a small
  NPU model, the time it costs may be small enough that on stays the right default — measure
  with the stage log before deciding.
- **Cleanup waits for the scan on machines with 16 GB or less** (`MemoryIsTight`). On the M2 a
  7B cleanup model (7.2 GB in Foundry) and the scan could not both fit; the scan crawled from
  swap for nearly two hours and every word fell back to estimates. A 16 GB Snapdragon
  matches the condition and will sequence too; a 32 GB one keeps the overlap.
- **The word aligner is the 4-bit build on macOS only** (`AlignmentModelSource.PreferQuantised`).
  `model_q4.onnx` quantises only the MatMuls (MatMulNBits, which has ARM64 kernels — unlike the
  int8 builds' ConvInteger), so it loads where the int8 builds fail. `--aligner-trial` graded
  it against fp16 on the debate and podcast fixtures: 99% and 92% of words within 0.1 s,
  every word within 0.5 s, no drift — from a 241 MB download instead of 632 MB, holding about
  1.6 GB during the scan instead of 3.0 GB. Speed: 19–30% faster scanning alone; in the full
  pipeline on the podcast at Balanced, twice each, alternated, fp16 took 4:26 and 4:09 from
  start to finished scan, q4 3:19 and 2:37. At Light the same comparison was noise — 3:40 to
  7:06 for either build — because below-normal threads go wherever macOS puts them, and an
  efficiency core can double a run. Compare builds at Balanced, never at Light.
  **Windows keeps fp16 until the same trial runs there.** The NPU plays no part in
  alignment on either platform, but the Snapdragon's CPU kernels for MatMulNBits are a
  different code path from Apple's and may not speed up the same way. To trial it:
  ```powershell
  # fetch the 4-bit build beside the fp16 one, under its published name
  curl.exe -L -o models\alignment-q4\model_q4.onnx https://huggingface.co/onnx-community/mms-300m-1130-forced-aligner-ONNX/resolve/main/onnx/model_q4.onnx
  copy models\alignment\vocab.json models\alignment-q4\
  localscribe-doctor --models models --aligner-trial "Peterson debate.scrb"
  localscribe-doctor --models models --aligner-trial "MM Aug 10_v1-new.scrb"
  ```
  If it matches the Mac's numbers, flipping `PreferQuantised` to include Windows is the whole
  change; setup will then fetch the 4-bit build on its own.

### Whisper's own word timing: clickable at once, refined by the scan (macOS)

whisper.cpp times each word from its cross-attention (DTW) as it transcribes. Graded against
the fp16 aligner by `--aligner-trial`: a steady lag of +0.12 to +0.18 s in every fifth of both
recordings — no drift, because each 30-second window is timed on its own and errors cannot
accumulate — and, with the lag removed, a median of 0.06 s, 67–72% of words within 0.1 s and
87–93% within 0.25 s. Not good enough to replace the scan (one word in ten lands more than a
quarter-second off, and the scan-based trials that catch duplicated lines would go with it),
but good enough to click on. So both run:

- `WhisperCppTranscriber` turns DTW on when the model has a published alignment-heads preset
  (`HeadsFor`), takes 0.15 s off for the measured lag, and exposes the words through
  `ITranscriber.HeardWords` — recording-absolute, cumulative, kept across an engine restart by
  `ResilientTranscriber`.
- `HeardWordPlacement` (Core, tested) pairs a displayed segment's words with heard words by
  text inside the segment's span, interpolates the unpaired, and refuses below half paired:
  wrongly timed is worse than grey.
- The view model consults the scan's words first and heard words second, so measured always
  wins and the scan replaces heard times segment by segment as its progressive pass reaches
  them. A segment the scan could not place keeps heard times instead of a loudness estimate.
- Podcast at Balanced: every streamed line clickable the moment its window landed (18 of 18),
  final transcript still 96 of 96 measured. Cost: transcription 68–70 s against 60 s, because
  the DTW decode ends segments differently and the chunker takes 18 windows instead of 16 —
  per-window time is unchanged. `LOCALSCRIBE_HEARD_WORDS=0` turns it off to re-measure.
- Speakers no longer wait for the scan. When the diarizer answers, whatever is on screen is
  divided between speakers on the best words to hand (scan where it has reached, heard
  elsewhere) and every later interim publish keeps the labels (`PublishInterim`); the final
  assembly ends the interim and redoes the division on measured words. Podcast at Light:
  five speakers on screen at 11:17:06, scan finished 11:20:14, same count at the end. A
  segment with no word times stays unlabelled rather than going whole to the loudest voice,
  so on Windows, whose stream has no heard words, labels still arrive with the scan's head.
- On macOS speakers are found from the moment the audio loads and the scan waits for
  transcription to end (`DiarizeEarly`, `ScanAfterTranscription`), because speakers are what
  make a transcript usable and sync comes second. Podcast at Balanced, twice each, seconds
  from the start of transcription to labels on screen / to done: old order 120 / 205 and
  92 / 155; everything at load 80 / 223 and 39 / 167; speakers early, scan after 18 / 219 and
  11 / 180. Labels arrive on the first windows while they stream. The scan alone was measured
  too: overlapped, it slows transcription by a fifth (Whisper's decoder shares the CPU) but
  saves 30–60 s overall, which is why holding it back is a choice for labels, not for speed.
  Windows keeps the overlap and late speakers until the same trial runs there:
  `LOCALSCRIBE_DIARIZE_EARLY=1` and `LOCALSCRIBE_SCAN_AFTER=1` (or `=0`) choose either way.
- Speakers can be renamed from the moment their labels appear ("this part" and
  "everywhere"; "by voice" waits for the finished transcript). `SpeakerNames` keeps an
  everywhere-name against the diarizer's own label for the voice and a part-name against its
  stretch of time, and every rebuild — new windows, the scan's re-division, the final
  assembly, a cleanup retry — puts them back after renumbering. Checked headless with
  `LOCALSCRIBE_HEADLESS_RENAME=Ada`, which renames Speaker 1 when labels first appear: in all
  six runs the final transcript still had Ada, including those where only two of the five
  speakers had appeared at the time of the rename.

Windows does not get this as it stands: the QNN transcriber is not whisper.cpp and reports no
heard words, so its transcript waits for the scan exactly as before. An ONNX equivalent would
need the decoder's cross-attention weights as an output; whether the current graphs offer
them has not been checked.

### Measured on the Snapdragon (2026-10-04), and what Windows adopted

The shared changes built and ran on Windows unchanged; 693 tests pass. Each trial above was run
on the Snapdragon X Elite (12 cores, 64 GB) with the podcast and debate fixtures, at Balanced.

- **The 4-bit aligner: adopted on Windows ARM64.** `--aligner-trial` against fp16: podcast 99%
  of words within 0.1 s and all within 0.25 s, debate 89% and 94%, no drift in any fifth; 62.5 s
  against 119.8 s and 37.4 s against 69.9 s. Graded against the audio by `--check-words`, the
  two builds left the same 12 of 71 words adrift on the debate, and 6 and 8 of 290 on the
  podcast. `PreferQuantised` now covers Windows on ARM64; x64 keeps fp16 until measured.
- **Speakers at load: adopted wherever Nemotron diarizes.** Seconds from the start of
  transcription to labels on screen, three runs each: 68, 68, 67 with the old order; 40, 32,
  32 with speakers found at load — and the first clickable words no later (33–41 against
  32–40), the run done no later. Labels still wait for the scan's timed head, because the QNN
  transcriber hears no words of its own; they now arrive with it instead of after
  transcription ends.
- **Scan after transcription: rejected for Windows.** Labels and the first clickable words came
  at 89 s instead of 32–40, and the run finished at 204 s instead of about 150. On the Mac the
  scan competes with Whisper's decoder for the CPU; here the transcriber is on the NPU and the
  scan is the only thing that makes the transcript clickable, so holding it back only delays.
- **Heard words: not possible with the current Snapdragon export.** The QNN decoder's declared
  outputs are `logits` and the self-attention caches; its cross-attention keys and values are
  inputs, and no attention weights come out. DTW timing would need a re-export from AI Hub
  with cross-attention outputs. Until then the scan's speed is what decides how soon a Windows
  transcript is clickable, which is why the 4-bit aligner matters more here than on the Mac.
- **Memory is not tight here** (64 GB), so cleanup keeps running beside the scan.
- **Cleanup is off by default on Windows too**, at the owner's call after this measurement: on
  the podcast, about 150 s to done without it and 236 s with it — roughly a minute and a half
  past the scan. The WinUI window has the switch under the processing-speed slider, worded as
  on the Mac, and its glossary dialog says cleanup is off, with a button to turn it on, rather
  than offering to download a model.
- **A Windows-only wait found on the way.** The finish stages began by checking the cleanup
  backend still answered, and when the cached client did not, asking Foundry's CLI where it
  was took about twelve seconds, during which the timed preview — and the speakers, when
  found late — had not started. The check now starts with transcription and is collected at
  the finish (`_cleanupCheck`); the gap is gone in both orders.
- **The roster summary broke the WinUI layout.** Naming the speaker and cleanup models made
  the hardware line long enough that, in an automatic column with a wrapping caption style, it
  took the status bar's whole width and wrapped to five lines, squeezing the transcript to a
  strip and hiding the status. The WinUI status bar now shares its width 2:1 and trims both
  lines; the full hardware line is a tooltip. Worth checking the Avalonia window for the same.
- **Shared-code change the Mac inherits:** `RenameSpeakerByVoiceAsync` now refuses while a run
  is in progress. It takes the busy state and hands it back when done, which mid-run would
  declare the run finished while it was still going. The Mac's menu already hides "by voice"
  until the transcript is finished; the WinUI dialog now does too.
- **Text is selectable on Windows from the moment it appears**, the Mac's click-or-drag rule
  (`OnBodyReleased`): a press and release that barely moved and selected nothing plays the word,
  a drag selects. Words are plain runs rather than links, because a link swallowed a drag that
  started on it. Copy copies the selection, or the whole transcript so far when there is none;
  a press in another paragraph lets go of the last selection, since WinUI stops drawing an
  unfocused block's highlight but keeps its selection.

- **The models chip is on Windows too, and its parsing is now shared.** The Mac window's
  roster reading — splitting `HardwareSummary` on " · ", adding the aligner, the short names
  and panel descriptions — moved into `src/LocalScribe.App/ModelRoster.cs` so the two windows
  cannot drift. The Avalonia window still has its own copy: link the file in
  `LocalScribe.Desktop.csproj` beside `MainViewModel.cs` and draw from `ModelRoster.From`. On
  Windows the chip replaced the status bar's hardware sentence, lit names carry
  "working now" in their accessible name, and the transcriber's device reads "on the Hexagon
  NPU" rather than the plan's "on Npu".

### Reaching the Mac from the Windows session (2026-10-04, evening)

Later changes to shared code. Each arrives with the next Mac build whether or not anything in
`LocalScribe.Desktop` changes; none has run on the Mac.

- **Lines one person spoke are labelled as soon as the diarizer answers** (`SoleVoice`, used in
  `WithInterimSpeakers`). An untimed line used to stay unlabelled; now, if exactly one speaker
  is heard across its span (others under half a second, at least a second of speech), it takes
  that label at once. On Windows, where no line has word times until the scan reaches it,
  this moved the first labels on the podcast from about 32 s into transcription to about 11 s,
  covering about a third of the windows on the podcast and on Karl's recording and none on the
  debate. On the Mac, heard words time nearly every streamed line already, so it should rarely
  fire — only where `HeardWordPlacement` refuses a line.
- **Interim labels keep Nemotron's own numbering** rather than renumbering by appearance among
  the lines labelled so far. With lines labelled ahead of the timed head, renumbering would
  have shifted every label as the gaps filled in; Nemotron already numbers voices in order of
  first arrival, and the final assembly still renumbers by appearance, which almost always
  gives the same numbers. Tracked through a whole podcast run on Windows, none of 32 paragraph
  labels changed after first appearing. Worth repeating on the Mac with the headless rename
  check (`LOCALSCRIBE_HEADLESS_RENAME`): names are keyed by the diarizer's labels, which this
  does not change, but it is the code path they ride on.
- **The progressive timing pass looks every two seconds until its first head is timed**, then
  every eight. It measured no earlier on Windows, where the margin behind the scan gates the
  first clickable words; harmless on the Mac, where the scan runs after transcription.
- **Cleanup is off by default everywhere now**, so `ReadCleanupPreference` no longer
  distinguishes platforms. Nothing changes on the Mac, where it was already off.
- **The cleanup-backend check starts with transcription**, not at the finish (`_cleanupCheck`),
  and only when cleanup is on. On Windows a stale Foundry endpoint cost twelve seconds at the
  start of the finish stages.
- **Rename by voice is refused mid-run** in the view model (listed above); the Mac menu already
  hides it until the transcript is finished.
- **The word aligner can scan on the Snapdragon's NPU** (2026-10-05; `NpuAligner`, see
  `docs/handoff.md`). Gated on the QNN provider specifically, so a Mac never compiles or opens
  it — the plan's "NPU" there means Core ML. What does reach the Mac's shared code: the
  progressive pass now runs until transcription ends rather than stopping when the scan does
  (`_streamingText`), and `ForcedAligner.Scan` takes an optional per-window callback the NPU
  path uses to take turns with Whisper. With the scan after transcription on the Mac, the
  flag is already down when the scan starts, so neither should change anything there.
  `ModelRoster` names the aligner "fp16 on the NPU" only when the compiled graph is on disk.
- **The stage log has new marks**: "transcription started" with the pace and order switches,
  "first words clickable", "labels on screen", "transcription ended" and "done". They are what
  the Windows ordering trial was timed with, and the Mac's headless runs write them too.

### New instruments

- `localscribe-doctor --aligner-trial <file.scrb>` — times each aligner build present and the
  Whisper DTW path, and grades each against the fp16 reference word by word, with drift by fifth.
- `LocalScribe.Desktop --headless <file> [--save <out.scrb>]` — the Mac window's whole pipeline
  with no window, for runs while the display sleeps. Windows has no equivalent yet.
- `localscribe-stages.txt` (temp folder) — one line as each finishing stage starts and ends,
  written by the shared view model on both platforms.
- `localscribe-errors.txt` — failure stacks, including an alignment state that vanished before
  placement, named with who discarded it and when.

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
