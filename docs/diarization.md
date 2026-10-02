# Speaker diarization

## Two engines, chosen by processor

**On ARM64 — the Snapdragon laptops and Apple silicon — speakers come from NVIDIA's
[Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization).** Everywhere
else they come from the pyannote pipeline described in the rest of this document, whose tuning
stays frozen. `SpeakerEngines.For` is the one switch.

Nemotron is a 100M-parameter streaming Sortformer: one transformer that emits eight speaker
activity channels every 10 ms, trained to keep each person on one channel. There is no
clustering and no threshold, so nothing to tune and nothing to drift — and a hard ceiling of
eight people. Speakers are numbered by first arrival; identity across a long recording is kept
by an arrival-order speaker cache (`ArrivalOrderSpeakerCache`) that the model is shown ahead of
every 27-second chunk.

Measured on the Snapdragon X Elite against the pipeline it replaced (`--diarize-trial`):

| Recording | Length | Pipeline speakers | Nemotron speakers | Same person, single-speaker time |
|---|---|---|---|---|
| Debate | 1:56 | 2 | 2 | 97.4% |
| Podcast, Aug 10 | 7:23 | 5 | 5 | 100% |
| Podcast, Aug 17 | 7:36 | 5 | 5 | 99.9% |
| Zero Trust panel | 61:42 | 16 | 3 | 99.1% |
| Karl (phone, poor) | 21:02 | 25 | 6 | 92.8% |

Where the pipeline was right the two agree; where it over-counted, the extra speakers were
fragments of one or two seconds. Nemotron was four to eleven times faster at every thread budget
(Karl: 25 s against 92 s at two threads), and peaks about 160 MB higher in memory.

**Provenance and verification.** NVIDIA publishes PyTorch weights only. The app downloads a
community ONNX export (`NealCaren/Nemotron-3-Diarization-ONNX`), pinned to a commit and to the
SHA-256 of every file (`SortformerModelSource`). The C# driver (`SortformerDiarizer`, with
`PreEmphasisLogMel` and `ArrivalOrderSpeakerCache` in Core) was checked against the exporter's
JavaScript reference on the debate and on all of Karl: 137,837 frames, every probability
bit-identical. The exporter checked that reference against NVIDIA's transformers code.

**A user-given speaker count** is met by folding extra speakers, smallest first, into whoever
they sound most like, using the pipeline's WeSpeaker voice model (`SpeakerMerging`). Nemotron
is never asked to split: if it heard fewer people than the user names, the status line says so.

**Crosstalk** is read straight off the model: frames where two channels are above 0.5.

The pipeline's models are still downloaded on ARM64, because renaming a speaker by voice
listens with the WeSpeaker model.

## The pyannote pipeline

Speaker labels on non-ARM64 machines come from pyannote's segmentation model, run through
[sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html).

## What runs where

**Diarization runs on the CPU. The NPU is not involved.** This surprises people, so it is worth
being explicit about why.

The *weights* are pyannote's. The *runtime* is sherpa-onnx's own bundled ONNX Runtime, which
does not carry the QNN execution provider — its `Provider` field accepts `cpu`, `cuda`,
`directml` and similar, but not `qnn`. And pyannote segmentation is a variable-length graph,
which the Hexagon backend would not accept as-is; reaching the NPU would mean a real conversion
through Qualcomm AI Hub, not a configuration flag.

So the NPU stays reserved for the Whisper encoder, and diarization takes its thread count from
`ExecutionPlan.CpuBudget` like everything else on the CPU.

There is no such thing as "pyannote-NPU". Nothing public ports pyannote to Hexagon.

## Getting the models

```powershell
dotnet run --project src/LocalScribe.Doctor -c Release -r win-arm64 -- --install
```

Two models land in `models/diarization/`:

| File | What it is |
| --- | --- |
| `segmentation.onnx` | pyannote segmentation 3.0 — finds speech and speaker changes |
| `embedding.onnx` | speaker embedding extractor — turns each turn into a vector to cluster |

Both are fetched from sherpa-onnx GitHub release assets. The exact asset names are **discovered
at runtime** rather than hard-coded, because those names change as models are revised. If the
discovery fails, `DiarizationModelInstaller.EmbeddingPreference` is the list to correct.

The embedding preference is ordered English-first on purpose. Several published extractors are
trained on Mandarin speakers and separate English voices noticeably less well.

Diarization is chipset-independent, so unlike Whisper the models are not stored per-chipset.

## Knowing the speaker count helps a lot

```csharp
var options = new DiarizationOptions { SpeakerCount = 2 };
```

Clustering is much better at splitting a *known* number of speakers than at guessing how many
there are. Guessing wrong is the most common way diarization output goes bad. If the recording
is a two-person interview, say so.

## The seam this design cannot hide

Transcription and diarization are separate models producing boundaries that do not line up.
`SpeakerAssigner` reconciles them by overlap: each segment takes the speaker it shares the most
time with.

That works until a Whisper segment straddles a speaker change. The text cannot be split at the
right word, because Whisper's timestamps are per-segment and loose. So the segment keeps its
dominant speaker and sets `SpeakerOverlapFraction` below 1, and `SpeakerIsUncertain()` reports
it. A UI should show those differently: the words are right, the attribution may not be.

Two things would fix this properly, in increasing order of effort:

1. **wav2vec2 forced alignment** for word-level timestamps, exported to ONNX. This is what
   WhisperX uses, and it is the reason WhisperX's speaker labels land better than ours will.
2. **A jointly trained model** that does transcription and diarization at once, which never has
   the seam in the first place. See the VibeVoice note in `docs/handoff.md`.

## Where the accuracy actually goes

If speaker labelling looks wrong, the models are the last place to look, not the first.

Measured on the debate recording: of 31 speaker changes the diarizer found, **5 landed on a
transcript segment edge and 26 landed inside one**. A nine-second segment held two changes; an
eleven-second one held three. The turns were right. Attribution was throwing them away, because
it could only cut a segment where a sentence ended.

Two model swaps were tried first and neither changed anything measurable:

| swapped | for | result |
|---|---|---|
| WeSpeaker ResNet34-LM | ResNet221, ResNet293, CAM++ | no difference |
| pyannote segmentation-3.0 | Reverb v1 | no difference |

That is what a downstream bottleneck looks like. Four embedding models spanning a 38% range of
published error rate all landed in the same place, and two segmentation models produced turn
boundaries within a fraction of a second of each other. Nothing upstream mattered because the
information was being discarded after it arrived.

Word-level attribution fixed it. The check, if this is ever in doubt again, is to compare turn
boundaries against transcript segment boundaries and count how many changes fall inside a
segment rather than at its edge — a saved `.scrb` archive carries both.

Beware of measuring one embedding model against another using a transcript this app produced:
those labels came from a model, so the incumbent wins by construction. `--speaker-models` says
so every time it runs. A reversed gap is still meaningful — a model whose same-voice distances
exceed its different-voice distances is separating nothing — but a few points between close
rivals means nothing at all.

## Known risks

- **Two ONNX Runtimes in one process.** sherpa-onnx bundles its own native ONNX Runtime, while
  Whisper uses the QNN build. Both loading into one process is the first thing to check if
  something crashes at startup on real hardware. Untested — no Windows machine was available
  when this was written.
- **Clustering does not exactly reproduce pyannote's own results.** There is an
  [open issue](https://github.com/k2-fsa/sherpa-onnx/issues/1708) tracking the mismatch. Expect
  good output, not identical-to-pyannote output.
- **Memory.** The native API takes the whole recording as one float array. An hour of 16 kHz
  mono is about 230 MB before the model allocates anything.
