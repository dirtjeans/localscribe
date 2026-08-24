# Speaker diarization

Speaker labels come from pyannote's segmentation model, run through
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
