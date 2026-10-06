# Handoff notes

Originally written at the end of the cloud session that drafted this codebase, when nothing had
ever run on Snapdragon hardware. That era is over: everything below has run, on the real
laptop, on real recordings, and most of it has been broken and fixed at least once. This
version says what is now proven, how it was proven, and what is still known to be imperfect —
in that order, because the *how* is the part a newcomer needs most.

The macOS port has run; its own handoff is [handoff-macos.md](handoff-macos.md).
Every model tried for a stage of the pipeline, and the numbers that kept or rejected it, is in
[model-trials.md](model-trials.md).

## Where things stand

The app transcribes on the Hexagon NPU (Whisper large-v3-turbo through a cached QNN export,
encoder and decoder both), times every word with an MMS CTC aligner on the CPU, attributes
speakers with NVIDIA's Nemotron 3 Diarization on ARM64 (pyannote segmentation plus WeSpeaker
embeddings elsewhere), cleans up
with a local language model through Foundry Local or GenieX, and plays back with a word-level
highlight that tracks the voice. Transcripts save as `.scrb` archives — a zip of the audio, the
segments, and a readable text copy — that reopen instantly and are byte-portable across
machines. The core library's 707 tests pass; the published app is self-contained and carries
its own .NET runtime.

The reference recordings are a seven-minute studio podcast with five speakers, an interview
wrap, an ad read over music, and one self-interruption; and a debate with heavy crosstalk.
Current state on the podcast: 6 of 290 checkable words adrift (a single bounded three-word
wobble mid-file plus two odd words), drift +0.00 in every fifth, one transcriber-duplicated
line detected and dropped, crosstalk marked from the segmentation model's own overlap classes,
and the highlight in sync to the last word.

## When a transcript becomes usable, on the Snapdragon

Three lanes start the moment a recording loads: transcription on the NPU, the word-timing scan
on the CPU, and — since 2026-10-04 — speaker finding with Nemotron, also on the CPU. Whisper
here reports no word times of its own, so a line becomes clickable when the scan's timed head
reaches it, and speaker labels attach to lines as they become timed. Measured on the 7-minute
podcast at Balanced, from the start of transcription: labels and the first clickable words at
about 32 s, transcription done at about 45 s, the scan and the run done at about 150 s, or
about 236 s with cleanup on. Cleanup is off by default, as on the Mac; the switch is under the
processing-speed slider. Windows where one person speaks throughout are labelled as soon as
Nemotron answers, about 11 s in on the podcast, ahead of their word times. Text is selectable and copyable as soon as it appears; a click
plays the word, a drag selects.

The order was chosen by measurement against the Mac's (see the macOS handoff's "Next Windows
build" section): the Mac holds the scan back until transcription ends because there it
competes with Whisper's decoder, and its whisper.cpp words make lines clickable without it.
Neither is true here, and holding the scan back measured slower on every count. Overrides for
re-measuring: `LOCALSCRIBE_DIARIZE_EARLY` and `LOCALSCRIBE_SCAN_AFTER` (`1` or `0`); the
stage log (`localscribe-stages.txt` in the temp folder) records "labels on screen", "first
words clickable" and "done" for each run.

The word aligner on Windows ARM64 is the 4-bit MMS build: about half the scan time of fp16
for the same sync against the audio. It is the lever that matters most here, because the scan
is what makes a Windows transcript clickable; the pace slider is the other.

## How anything here gets believed

This project's history is a sequence of confident wrong theories corrected by measurement, and
the instruments that ended each one are permanent residents. Learn them before changing
anything; they are how a claim about this codebase gets to be true.

- **`dotnet test`** — 589 tests on the policy and the maths, runnable anywhere.
- **`localscribe-doctor --check-words <file.scrb>`** — re-times a saved transcript against its
  own audio and grades every findable word: drift by fifths, adrift words with what the audio
  actually says at their claimed seconds, unheard-line and doubled-line trials.
- **`localscribe-doctor --align <wav> [--window a-b]`** — scans audio with the alignment model
  and reads a window back as letters. **On a slice cut from the raw samples this is the one
  non-circular instrument in the box**: every in-scan check (drift, Locate, read-back on the
  full scan) measures the scan against itself and once agreed unanimously on a timebase that
  was wrong. A fresh scan of a sample slice has its own clock and owes the big scan nothing.
- **`localscribe-doctor --replay <input.txt> --audio <wav>`** — re-runs the aligner on the
  exact input an app run dumped. Built because the app and the checker disagreed for days on
  the same audio; with it, any "the app is wrong" reproduces offline in seconds.
- **`localscribe-doctor --diarize <wav>`** — turns, speech spans, and the contested (crosstalk)
  stretches. Diff its turn list before and after any change near diarization; the turns are
  the tuning, and the tuning is deliberately frozen (see below).
- **The app's own diagnostics**, written to `%TEMP%` on every run: `localscribe-input.txt`
  (the transcriber's raw stamps, exactly as the aligner will anchor on them),
  `localscribe-spans.txt` (what the marker actually follows, measured or estimate, per
  segment), `localscribe-alignment.txt` (boundary crowding), `localscribe-clock.txt` (playback
  clock against a stopwatch). They cost nothing and they turn "it feels off" into a file.
- The app accepts a WAV path as a launch argument, so a debug transcription can be driven
  end-to-end without touching the UI.

The discipline that goes with the instruments: measure before theorising, prefer evidence the
system cannot fake to evidence it produces about itself, and when two components disagree,
reproduce the disagreement outside the app before changing either.

## The word-timing architecture, and why it is shaped this way

Whisper's stamps cannot be trusted: they drift in a sawtooth that reached **17.5 seconds late
by the sixth minute** of the reference podcast, and the final padded window stamps its lines up
to 18 seconds past the end of the audio. Every windowed per-segment aligner tried here — 
stamp-anchored, chained, bias-corrected, widened on evidence — fixed one failure by creating
another, because a window is a local decision about a global constraint.

What survived is one global pass (`GlobalCtcAlignment`): the whole transcript's letters against
the whole recording's frames, a banded blank-extended Viterbi whose corridor makes text↔time
monotonicity unrepresentable to violate. Around it, three guards, each earned by a specific
failure:

- **`CredibleAnchors`** prunes stamps that could not be spoken — an anchor leaving more text
  than the remaining seconds could carry at a sprint is a lie, and clamped end-of-file lies
  once got seven real outro lines convicted as never spoken.
- **The corridor band is 36 seconds of letters** either side of the spine, sized as roughly
  double the worst *residual* stamp lie. The accumulated part is removed first: the drift is
  roughly linear in time (17.5 s over seven minutes; 73 s over twenty-one), so when the stamps
  describe a longer timeline than the audio, every anchor is scaled by real-end over
  claimed-end (`CredibleAnchors.Scale`) before the spine is built. Without that, a long
  recording's drift outruns any affordable band — a twenty-one minute recording placed its
  eighteenth minute twenty-eight seconds late; with it, the same minute landed within a
  second of fresh-slice ground truth. If a long recording still lags, measure with
  `--replay` and fresh slices before touching either constant.
- **Two trials after placement**: lines stamped past the end of the audio must prove their
  words exist (span and read-back, both failing before conviction — dropping real speech is
  the worse lie, and this trial has told it once), and a line whose folded text is a verbatim
  copy of a neighbour's and whose placed audio does not read as itself is a window-seam
  duplicate — the stitcher cannot see a copy buried a few words inside a segment, and under a
  global path a surviving twin steals real frames and drags whole sentences off their sound.
  One conviction per round, then the pass runs again so displaced real lines are re-placed
  rather than condemned.

Downstream, order is law: the decoder's emission order is authoritative and is never re-derived
from placed times — sorting by placement turned bounded time errors into unbounded order
errors, twice.

## Diarization: frozen tuning, honest marks

**On ARM64 the diarizer is Nemotron 3 Diarization, not the pipeline this section goes on to
describe.** `SpeakerEngines.For` picks by processor; the reasoning, the five-recording
measurement and the model's provenance are in `diarization.md`. The pipeline below still runs
everywhere else, still frozen, and its WeSpeaker model still serves rename-by-voice on ARM64.

Everything downstream of the diarizer — word-level attribution, `UnfinishedSentences`,
crosstalk marks — was tuned while pyannote supplied the turns, and two of its rules exist
because of pyannote's habits: turns under half a second are ignored as window-vote flicker, and
a word no turn covers is resolved by list order. `--diarize-trial <file.scrb>` runs a saved
transcript's words through that attribution with both engines and counts what those rules do.
On the debate and both podcasts neither rule fired on Nemotron's turns beyond a single word
per recording, so nothing downstream was retuned. (The hour-long panel's word-level run did not
finish inside the session's time limit; its segment-level comparison agreed on 99.1% of
single-speaker time.) Where the two engines
attribute words differently, Nemotron was right more often: on the second podcast it was right
on all four disputed runs (host lines, and where the guest's answer begins) and the pipeline on
none; on the debate it separated "How is it irrelevant?", "In what way?" and "Because, again--"
from the paragraphs around them. The debate's remaining errors — a sentence split mid-way, the
tangle around 1:49 where the transcriber wrote crosstalk down twice — appear under both engines
and sit downstream of the diarizer.

The pipeline's active method is 'voices' (clustering), selected by `active-diarizer.txt` beside the
models; 'tracking' remains available and each has recordings it wins on. Attribution is
word-level: segments are cut at the word where the voice changed, judged on each word's ending,
with grammar repairs for sentences split across turns (`UnfinishedSentences`) and a
tiling-count guard so no cut can ever lose a word.

**The tuning is frozen on purpose.** It is imperfect and it is the best this app has had; the
owner has said not to touch it. The enforcement is measurement: `--diarize` turn output on the
reference podcast is the fixture, and a change that alters any boundary by a hundredth of a
second is a tuning change whatever it was called.

Crosstalk is *marked*, not resolved. The segmentation model's powerset classes can say "two
speakers at once"; those frames are collected (`PowersetDecoder.OverlappedFrames` →
`SpeakerDiarizer.LastOverlaps`) and a line is badged when three quarters of a second of its
measured words fall on contested time. Two earlier lessons are baked in: the transcriber
usually writes down only the louder stream, so crosstalk cannot be detected from the text — 
and the global aligner is one monotone path, so it can never place two words on the same
instant, which is why word-time collision is structurally dead as evidence. The badge exists to
set the reader's expectations exactly where the labels are least trustworthy.

## The app layer's paid-for lessons

- **Playback position updates are coalesced, latest-wins**: at most one UI update in flight,
  reading the newest position when it runs. Twenty queued updates a second against slow
  repaints made the marker replay the past.
- **The marker rules** (word covering the instant beats paragraph bounds; else the most
  recently begun word; repaint all realised paragraphs) each exist because their absence was a
  visible bug. So does the playback clock reporting the device's position rather than the read
  position.
- **The aligned-times table (`_alignedFor`) is keyed by segment value.** Anything that rewrites
  segment records — speaker marks, crosstalk flags — must run *before* the table is keyed, or
  the rewritten lines silently fall back to loudness estimates. This has been a live bug once
  and a latent one twice.
- **Closing asks before it destroys**: unsaved work (a fresh transcription, a live recording,
  an edited transcript) gets Save / Discard / Cancel, and a cancelled save picker is not
  consent to lose anything.
- The cleanup model can be provisioned from the glossary dialog's notice (install Foundry
  Local, start it, download the default model, reconnect, rerun cleanup) — all user-initiated;
  the app never installs anything uninvited.

## Decisions worth not undoing

These look like things to tidy up. They are not.

**Build native arm64.** Under x64 emulation the QNN provider cannot load and the symptoms are
identical to a missing driver.

**The decoder goes to an accelerator only when the encoder did.** Per-step dispatch overhead
beats the compute saved otherwise. (On the current QNN export both run on the NPU; the point
stands for any configuration where they would split.)

**Diarization runs on the CPU.** The weights are pyannote's; the runtime is sherpa-onnx's own
ONNX Runtime, which has no QNN provider. There is no "pyannote-NPU". See `diarization.md`.

**Downloaded model files keep their published names.** Large ONNX graphs reference weight
sidecars by name from inside the graph. `localscribe-model.json` records roles instead.

**The Foundry Local port is never hard-coded.** It binds a dynamic loopback port; ask
`foundry service status`.

**CPU threads are capped rather than maximised.** The machine staying responsive is the product
requirement, not a limitation to optimise away. The pace slider (`WorkPaces`) lets the user
move the budget; its Balanced stop is exactly the planner's answer and must stay so. Its top
stop leaves two cores free, and that is measured, not cautious: on the 12-core Snapdragon the
aligner took 18.1 s on ten threads and 25.4 s on twelve, and during a burst of Windows Recall's
background indexing twelve threads took Nemotron from 36 s to 130 s. Every parallel step waits
for its slowest thread, and a thread sharing its core with anything else is that thread.

| Pace | Threads here | Aligner, 7-min podcast | Speech model |
|---|---|---|---|
| Light | 1, NPU in balanced power | 409 s | 17.3 s on the 2-min debate |
| Balanced | 2 | 217 s | 16.7 s |
| Fast | 6 | 85 s | as Balanced |
| Fastest | 10 | about 70 s, from the debate's scaling | as Balanced |

The NPU's power mode barely moves the speech model on this export (17.3 s against 16.7 s); the
aligner is where the slider earns its place.

**Nothing installs the Hexagon driver.** Signed kernel driver behind an account wall; report
it, never automate it.

**`LocalScribe.Core` has no external dependencies.** That is what makes the policy testable on
any machine — and what made the macOS port plan mostly a list of things that port untouched.

**The app publishes self-contained.** It is a folder someone is handed; published
framework-dependent it stops with an "install .NET" dialog the day an update removes the
runtime it happened to rely on.

**`LocalScribe.App` stays out of `LocalScribe.sln`.** WinUI cannot restore on non-Windows and
its presence would break `dotnet build` for every other contributor. Build it by path.

## The word aligner on the NPU (2026-10-05)

Where the planner put Whisper on the Hexagon NPU (QNN, not Core ML), the MMS aligner scans there
too, once compiled — `NpuAligner`, `ForcedAligner.LoadNpu`. What it took, and why:

- **`Erf`.** The QNN provider in ONNX Runtime 1.22 cannot run it, and the export uses it in all
  32 GELUs, which split the graph so the NPU would not finalize it (`LOCALSCRIBE_QNN_VERBOSE=1`
  shows why). `GeluRewrite` (Core, no dependencies, protobuf wire format) turns each
  `Div → Erf → Add → Mul → Mul` chain into one `com.microsoft` `Gelu`. Its output is
  byte-identical to the Python/onnx rewrite that was graded; `--gelu-rewrite <model>` writes one.
  ONNX Runtime's own fusion does not run on the fp16 model.
- **One window length.** The NPU compiles for exact shapes: 14 s (10 s kept, 2 s margins, ends
  slid inward, never padded). 34 s grew to 33 GB of memory and never finished.
- **Compiled once, in the background.** About 4 minutes and 8 GB at the peak, so only where
  memory is not tight, at the lowest priority, from launch; until it exists the CPU scans as
  before. It compiles into `models/alignment/npu/` and is trusted only once the `compiled` marker
  is written — not compiled elsewhere and moved, because under OneDrive the folder move is what
  failed. A graph the NPU later refuses (driver update) is forgotten and recompiled next launch.
  Needs the fp16 build; fetched first if only the 4-bit one is installed. ~750 MB on disk.
- **Taking turns.** The NPU runs one graph at a time and the scan's windows queue back to back:
  unpaced, Whisper wrote nothing for the scan's whole minute. So the scan waits whenever it is
  more than 60 s ahead of the streamed text (`NpuScanLeadSeconds`), and the progressive pass now
  keeps timing streamed text after the scan finishes, until transcription ends.
- **Measured** (podcast, 443 s): first clickable 18 s, done 101 s, against 32 s and 160 s with
  the 4-bit CPU scan; the text itself takes 98 s to finish streaming instead of 45. Unpaced:
  no text until 62 s, done 115 s. Word placement against the CPU fp16 reference
  (`--aligner-npu <file.scrb> --app`, the app's own path): podcast 99% within 0.1 s, debate 93%
  (97% within 0.25 s), no drift in any fifth — the same as the trial that preceded it. Scan
  alone: podcast 58.5 s against 191 s for fp16 on the CPU.
- **Switches.** `LOCALSCRIBE_NPU_ALIGNER=0` keeps the scan on the CPU. Recordings under 14 s
  always use the CPU. The stage log says which ran (`word scan ended (NPU)`).

No NPU model needs unloading for this: idle sessions cost memory, not NPU time, and the aligner's
session is released the moment its scan ends.

## Evaluated and not taken

- **WhisperX** — faster-whisper/CTranslate2 has no usable Windows ARM64 build. Its wav2vec2
  alignment idea was, in effect, adopted: the MMS CTC aligner is that design, done here.
- **VibeVoice-ASR** — spiked on the real machine: the GenieX SDK on this laptop exposes no ASR
  API, so there is nothing to integrate against. Re-evaluate only if that changes.
- **Parakeet** — deferred by choice; revisit only if asked.
- **`--install` as the app's provisioning path** — evaluated, kept unwired; the doctor's
  `--fetch-models` and the in-app cleanup provisioning cover the real flows.

## Known blemishes, honestly

- Early clicks during a run are unpredictable, with odd pauses. Three suspects, unmeasured:
  playback buffer underruns while the scan and transcription saturate the machine; the
  progressive passes re-timing the head every few strides so the same word can land slightly
  differently twice; and row rebuilds landing mid-click. Measure before fixing — the clock
  diagnostic already distinguishes underruns from timing shifts.

- The "vendor-neutral zero-trust certification" phrase repeats in the podcast transcript and
  survives the twin trial: the copy is embedded mid-segment, below the trial's
  whole-segment granularity. Cost: a bounded three-word timing wobble, the file's worst.
- One seam fragment ("So, and that's") remains inside a real segment for the same reason.
- The crosstalk badge threshold (0.75 s) has been proven on the debate and the podcast only;
  other recordings may want the constant moved, and `--diarize` measures before anyone tunes.
- Greedy decoding, no KV cache in the decode loop — accuracy and speed both leave a little on
  the table, deliberately, until a specific export justifies binding to it.

## Where the logic lives

| Question | File |
| --- | --- |
| Why did it pick the NPU, GPU, or CPU? | `Core/Hardware/AcceleratorPlanner.cs` |
| How does audio become model input? | `Core/Audio/LogMelSpectrogram.cs` |
| Why is this word at this second? | `Core/Alignment/GlobalCtcAlignment.cs`, `Onnx/ForcedAligner.cs` |
| Why was this stamp ignored? | `Core/Alignment/CredibleAnchors.cs` |
| Why was this line dropped? | the trials in `App/MainViewModel.AlignWordsAsync` and `Doctor/CheckWordsCommand.cs` |
| Why is this line attributed to this person? | `Core/Diarization/WordLevelAttribution.cs` |
| Why is this line badged as crosstalk? | `Core/Diarization/CrosstalkMarks.cs`, `Core/Diarization/PowersetDecoder.cs` |
| Why is this word duplicated or missing? | `Core/Transcription/RepeatedPhrase.cs`, `Core/Transcription/TranscriptStitcher.cs` |
| Why does live text keep changing? | `Core/Pipeline/LiveTranscriptionSession.cs` |
| How does a provider actually get registered? | `Onnx/OnnxSessionFactory.cs` |
| What does saving actually write? | `Core/Archive/TranscriptArchive.cs` |

## A note on the tests

The 589 tests cover decisions, not snapshots. The FFT is checked against the definition of the
DFT; the corridor is checked by proving a repeated phrase cannot swap its occurrences; the
anchor pruning is checked by proving a stamp that leaves more text than the clock allows is
discarded and a quiet recording's one dense sentence is not.

That matters because the failures here do not crash. A wrong filterbank, a corridor an inch too
narrow, a trial an inch too eager — each produces a program that runs happily and is wrong in a
way only a person listening would notice. Every constant in this codebase with a comment
explaining a measurement got that comment because someone listened. Keep the tests aimed at the
decisions, and keep the instruments (above) the arbiter of any claim the tests cannot reach.
