# Handoff notes

Originally written at the end of the cloud session that drafted this codebase, when nothing had
ever run on Snapdragon hardware. That era is over: everything below has run, on the real
laptop, on real recordings, and most of it has been broken and fixed at least once. This
version says what is now proven, how it was proven, and what is still known to be imperfect —
in that order, because the *how* is the part a newcomer needs most.

The macOS port has run; its own handoff is [handoff-macos.md](handoff-macos.md).

## Where things stand

The app transcribes on the Hexagon NPU (Whisper large-v3-turbo through a cached QNN export,
encoder and decoder both), times every word with an MMS CTC aligner on the CPU, attributes
speakers with pyannote segmentation plus WeSpeaker embeddings through sherpa-onnx, cleans up
with a local language model through Foundry Local or GenieX, and plays back with a word-level
highlight that tracks the voice. Transcripts save as `.scrb` archives — a zip of the audio, the
segments, and a readable text copy — that reopen instantly and are byte-portable across
machines. The core library's 589 tests pass; the published app is self-contained and carries
its own .NET runtime.

The reference recordings are a seven-minute studio podcast with five speakers, an interview
wrap, an ad read over music, and one self-interruption; and a debate with heavy crosstalk.
Current state on the podcast: 6 of 290 checkable words adrift (a single bounded three-word
wobble mid-file plus two odd words), drift +0.00 in every fifth, one transcriber-duplicated
line detected and dropped, crosstalk marked from the segmentation model's own overlap classes,
and the highlight in sync to the last word.

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
  double the worst stamp lie ever measured. The band bounds compute, not placement — the audio
  decides inside it — so its only honest size is wider than the worst lie. A recording much
  longer than ten minutes may drift proportionally further; if a long recording's tail lags,
  suspect this constant first and measure with `--check-words` before touching it.
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

The active method is 'voices' (clustering), selected by `active-diarizer.txt` beside the
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
requirement, not a limitation to optimise away.

**Nothing installs the Hexagon driver.** Signed kernel driver behind an account wall; report
it, never automate it.

**`LocalScribe.Core` has no external dependencies.** That is what makes the policy testable on
any machine — and what made the macOS port plan mostly a list of things that port untouched.

**The app publishes self-contained.** It is a folder someone is handed; published
framework-dependent it stops with an "install .NET" dialog the day an update removes the
runtime it happened to rely on.

**`LocalScribe.App` stays out of `LocalScribe.sln`.** WinUI cannot restore on non-Windows and
its presence would break `dotnet build` for every other contributor. Build it by path.

## Evaluated and not taken

- **WhisperX** — faster-whisper/CTranslate2 has no usable Windows ARM64 build. Its wav2vec2
  alignment idea was, in effect, adopted: the MMS CTC aligner is that design, done here.
- **VibeVoice-ASR** — spiked on the real machine: the GenieX SDK on this laptop exposes no ASR
  API, so there is nothing to integrate against. Re-evaluate only if that changes.
- **Parakeet** — deferred by choice; revisit only if asked.
- **`--install` as the app's provisioning path** — evaluated, kept unwired; the doctor's
  `--fetch-models` and the in-app cleanup provisioning cover the real flows.

## Known blemishes, honestly

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
