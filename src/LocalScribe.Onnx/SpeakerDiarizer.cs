using LocalScribe.Core.Audio;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Transcription;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalScribe.Onnx;

/// <summary>
/// Works out who spoke when.
/// <para>
/// Two models, because the question has two halves that nothing answers at once. Segmentation
/// says how many voices are active in a ten-second window and where each one starts and stops,
/// but numbers them locally — its speaker 0 in one window has nothing to do with speaker 0 in
/// the next. Embeddings turn a stretch of one voice into a vector that is close to other
/// stretches of the same voice, which is what carries identity between windows. Clustering those
/// vectors is what turns a pile of local guesses into a recording with two people in it.
/// </para>
/// <para>
/// Whisper does neither of these things and cannot be made to. This runs alongside it, over the
/// same audio, and the result is joined to the transcript by time.
/// </para>
/// <para>
/// Both models are float32 with dynamic shapes, so they run on the CPU. The QNN provider needs
/// fixed shapes and quantised weights; making these NPU-capable is a separate exercise, and at
/// six and twenty-six megabytes they are not what makes a transcription slow.
/// </para>
/// </summary>
public sealed class SpeakerDiarizer : IDisposable
{
    private readonly InferenceSession _segmentation;
    private readonly InferenceSession _embedding;
    private readonly KaldiFbank _fbank = new();

    private readonly int _windowSamples;
    private readonly int _localSpeakers;
    private readonly int _maxOverlap;
    private readonly double _receptiveFieldShift;
    private readonly double _receptiveFieldStart;
    private readonly IReadOnlyList<IReadOnlyList<int>> _mapping;

    /// <summary>
    /// How far the analysis window moves each step. A tenth of the window, as pyannote uses:
    /// enough overlap that a speaker change is seen by several windows rather than landing on a
    /// boundary and being missed.
    /// </summary>
    private const double WindowShiftFraction = 0.1;

    /// <summary>
    /// Speech shorter than this is not given its own embedding. A third of a second of one voice
    /// is not enough to identify it, and a vector built from too little audio clusters by
    /// accident rather than by speaker.
    /// </summary>
    private const double MinimumTurnSeconds = 0.35;

    private SpeakerDiarizer(InferenceSession segmentation, InferenceSession embedding)
    {
        _segmentation = segmentation;
        _embedding = embedding;

        var metadata = segmentation.ModelMetadata.CustomMetadataMap;

        _windowSamples = Read(metadata, "window_size", 160_000);
        _localSpeakers = Read(metadata, "num_speakers", 3);
        _maxOverlap = Read(metadata, "powerset_max_classes", 2);

        // The model's output frames are not evenly spaced over the window: each one summarises a
        // receptive field, so the first is centred partway in. Getting this wrong shifts every
        // boundary by a fixed amount, which looks like a model that is slightly late rather than
        // like an arithmetic mistake.
        var fieldSize = Read(metadata, "receptive_field_size", 991);
        var fieldShift = Read(metadata, "receptive_field_shift", 270);
        var sampleRate = Read(metadata, "sample_rate", PcmAudio.WhisperSampleRate);

        _receptiveFieldShift = fieldShift / (double)sampleRate;
        _receptiveFieldStart = fieldSize / 2.0 / sampleRate;

        _mapping = PowersetDecoder.Mapping(_localSpeakers, _maxOverlap);
    }

    /// <summary>Opens both models from a directory holding segmentation.onnx and embedding.onnx.</summary>
    public static SpeakerDiarizer Load(string modelDirectory)
    {
        var segmentation = new InferenceSession(Require(modelDirectory, "segmentation.onnx"));

        try
        {
            var embedding = new InferenceSession(Require(modelDirectory, "embedding.onnx"));
            return new SpeakerDiarizer(segmentation, embedding);
        }
        catch
        {
            segmentation.Dispose();
            throw;
        }
    }

    private static string Require(string directory, string name)
    {
        var path = Path.Combine(directory, name);

        return File.Exists(path)
            ? path
            : throw new FileNotFoundException(
                $"No {name} in {directory}. Run 'localscribe-doctor --fetch-models --diarization' "
                + "to download the speaker models.",
                path);
    }

    /// <summary>Finds the speaker turns in a recording.</summary>
    /// <param name="audio">16 kHz mono, the same audio the transcriber saw.</param>
    /// <param name="threshold">Cosine distance at which two voices count as different people.</param>
    /// <param name="maxSpeakers">Upper bound on speakers, or null to let the threshold decide.</param>
    public IReadOnlyList<SpeakerTurn> Diarize(
        PcmAudio audio,
        double threshold = SpeakerClustering.DefaultThreshold,
        int? maxSpeakers = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);

        var local = FindLocalTurns(audio, cancellationToken);
        if (local.Count == 0)
        {
            return [];
        }

        var embeddings = new List<float[]>(local.Count);
        var kept = new List<(double Start, double End)>(local.Count);

        foreach (var (start, end) in local)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var embedding = Embed(audio, start, end);
            if (embedding is null)
            {
                continue;
            }

            embeddings.Add(embedding);
            kept.Add((start, end));
        }

        if (embeddings.Count == 0)
        {
            return [];
        }

        var labels = SpeakerClustering.Cluster(embeddings, threshold, maxSpeakers);

        var turns = kept
            .Select((span, i) => new SpeakerTurn(labels[i], span.Start, span.End))
            .OrderBy(t => t.StartSeconds)
            .ToList();

        return Merge(turns);
    }

    /// <summary>
    /// Runs the segmentation model across the recording and collects every stretch where one
    /// local speaker was active, as a span of time. Local speaker numbering is discarded here:
    /// it is meaningless between windows, and the embeddings decide identity.
    /// </summary>
    private List<(double Start, double End)> FindLocalTurns(PcmAudio audio, CancellationToken cancellationToken)
    {
        var shift = (int)(_windowSamples * WindowShiftFraction);
        var spans = new List<(double Start, double End)>();

        for (var offset = 0; offset < audio.Samples.Length; offset += shift)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var window = new float[_windowSamples];
            var available = Math.Min(_windowSamples, audio.Samples.Length - offset);

            if (available < _windowSamples / 4)
            {
                break;
            }

            Array.Copy(audio.Samples, offset, window, 0, available);

            using var outputs = _segmentation.Run(
            [
                NamedOnnxValue.CreateFromTensor(
                    _segmentation.InputMetadata.Keys.First(),
                    new DenseTensor<float>(window, [1, 1, _windowSamples])),
            ]);

            var tensor = outputs.First().AsTensor<float>();
            var frames = tensor.Dimensions[1];
            var scores = tensor.ToArray();

            var active = PowersetDecoder.Decode(scores, frames, _mapping, _localSpeakers);
            var windowStart = offset / (double)audio.SampleRate;

            for (var speaker = 0; speaker < _localSpeakers; speaker++)
            {
                spans.AddRange(SpansFor(active, frames, speaker, windowStart, available, audio.SampleRate));
            }
        }

        return Merge(spans);
    }

    /// <summary>Runs of consecutive active frames for one local speaker, as times.</summary>
    private IEnumerable<(double Start, double End)> SpansFor(
        bool[] active,
        int frames,
        int speaker,
        double windowStart,
        int availableSamples,
        int sampleRate)
    {
        var limit = windowStart + (availableSamples / (double)sampleRate);
        int? runStart = null;

        for (var frame = 0; frame <= frames; frame++)
        {
            var isActive = frame < frames && active[(frame * _localSpeakers) + speaker];

            if (isActive && runStart is null)
            {
                runStart = frame;
            }
            else if (!isActive && runStart is { } begin)
            {
                var start = windowStart + _receptiveFieldStart + (begin * _receptiveFieldShift);
                var end = windowStart + _receptiveFieldStart + (frame * _receptiveFieldShift);

                runStart = null;

                // The last window is zero-padded; anything the model hears in the padding is an
                // artefact of the padding.
                if (end > limit)
                {
                    end = limit;
                }

                if (end - start >= MinimumTurnSeconds)
                {
                    yield return (start, end);
                }
            }
        }
    }

    /// <summary>
    /// Joins spans that touch or overlap. Windows overlap by ninety percent, so the same speech
    /// arrives as a dozen near-identical spans and only the union of them is a turn.
    /// </summary>
    private static List<(double Start, double End)> Merge(List<(double Start, double End)> spans)
    {
        if (spans.Count == 0)
        {
            return spans;
        }

        var ordered = spans.OrderBy(s => s.Start).ToList();
        var merged = new List<(double Start, double End)> { ordered[0] };

        foreach (var span in ordered.Skip(1))
        {
            var last = merged[^1];

            if (span.Start <= last.End)
            {
                merged[^1] = (last.Start, Math.Max(last.End, span.End));
            }
            else
            {
                merged.Add(span);
            }
        }

        return merged;
    }

    /// <summary>Joins adjacent turns the clustering gave the same speaker.</summary>
    private static IReadOnlyList<SpeakerTurn> Merge(List<SpeakerTurn> turns)
    {
        var merged = new List<SpeakerTurn>();

        foreach (var turn in turns)
        {
            if (merged.Count > 0
                && merged[^1].Speaker == turn.Speaker
                && turn.StartSeconds - merged[^1].EndSeconds < 0.5)
            {
                merged[^1] = merged[^1] with { EndSeconds = Math.Max(merged[^1].EndSeconds, turn.EndSeconds) };
                continue;
            }

            merged.Add(turn);
        }

        return merged;
    }

    /// <summary>Embeds an explicit span. Exposed so the distances can be measured directly.</summary>
    public float[]? EmbedSpan(PcmAudio audio, double startSeconds, double endSeconds) =>
        Embed(audio, startSeconds, endSeconds);

    /// <summary>A speaker embedding for one stretch of audio, or null when it is too short.</summary>
    private float[]? Embed(PcmAudio audio, double startSeconds, double endSeconds)
    {
        var start = Math.Clamp((int)(startSeconds * audio.SampleRate), 0, audio.Samples.Length);
        var end = Math.Clamp((int)(endSeconds * audio.SampleRate), start, audio.Samples.Length);

        if (end - start < MinimumTurnSeconds * audio.SampleRate)
        {
            return null;
        }

        var features = _fbank.Compute(audio.Samples.AsSpan(start, end - start));
        if (features.Length == 0)
        {
            return null;
        }

        var frames = features.Length / _fbank.MelBins;

        using var outputs = _embedding.Run(
        [
            NamedOnnxValue.CreateFromTensor(
                _embedding.InputMetadata.Keys.First(),
                new DenseTensor<float>(features, [1, frames, _fbank.MelBins])),
        ]);

        return outputs.First().AsTensor<float>().ToArray();
    }

    /// <summary>
    /// Attaches speakers to a transcript by overlap: each segment gets whichever speaker was
    /// talking for most of it. Words and voices are found independently and their boundaries
    /// never line up exactly, so this has to be a majority rather than a match.
    /// </summary>
    public static IReadOnlyList<TranscriptSegment> Attribute(
        IReadOnlyList<TranscriptSegment> segments,
        IReadOnlyList<SpeakerTurn> turns)
    {
        ArgumentNullException.ThrowIfNull(segments);
        ArgumentNullException.ThrowIfNull(turns);

        if (turns.Count == 0)
        {
            return segments;
        }

        return segments.Select(segment =>
        {
            var best = turns
                .Select(turn => (turn, overlap: turn.OverlapWith(segment.StartSeconds, segment.EndSeconds)))
                .Where(x => x.overlap > 0)
                .OrderByDescending(x => x.overlap)
                .Select(x => x.turn)
                .FirstOrDefault();

            return best is null ? segment : segment with { Speaker = best.Label };
        }).ToList();
    }

    private static int Read(IReadOnlyDictionary<string, string> metadata, string key, int fallback) =>
        metadata.TryGetValue(key, out var value) && int.TryParse(value, out var parsed) ? parsed : fallback;

    public void Dispose()
    {
        _segmentation.Dispose();
        _embedding.Dispose();
    }
}
