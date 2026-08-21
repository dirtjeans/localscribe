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
    /// <param name="exactSpeakers">
    /// The known number of speakers, when the user has said. Far more reliable than inferring it
    /// from a distance threshold, which on a real recording tends to fail in both directions at
    /// once.
    /// </param>
    public IReadOnlyList<SpeakerTurn> Diarize(
        PcmAudio audio,
        double threshold = SpeakerClustering.DefaultThreshold,
        int? maxSpeakers = null,
        int? exactSpeakers = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);

        var local = CollectSpeechSpans(audio, progress, cancellationToken);
        if (local.Count == 0)
        {
            return [];
        }

        var embeddings = new List<float[]>(local.Count);
        var kept = new List<(double Start, double End)>(local.Count);

        for (var i = 0; i < local.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            progress?.Report(ScanShare + ((1 - ScanShare) * i / local.Count));

            var (start, end) = local[i];
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

        var labels = ClusterWithShortSpansAttached(embeddings, kept, threshold, maxSpeakers, exactSpeakers);

        var turns = kept
            .Select((span, i) => new SpeakerTurn(labels[i], span.Start, span.End))
            .OrderBy(t => t.StartSeconds)
            .ToList();

        return Tidy(turns);
    }

    /// <summary>
    /// How much of the reported progress the window scan accounts for. Measured roughly: the
    /// scan runs the segmentation model over every window, the embeddings run a smaller model
    /// over a few dozen short spans.
    /// </summary>
    private const double ScanShare = 0.8;

    /// <summary>
    /// Speech shorter than this identifies a voice poorly. Long enough to say "Good." and not
    /// much else, and an embedding built from that clusters by accident as often as by speaker.
    /// </summary>
    private const double ReliableTurnSeconds = 0.9;

    /// <summary>
    /// How much a second of separation counts against a match, when placing a short span. Small
    /// enough that a clearly different voice still wins, large enough to decide between two that
    /// sound alike.
    /// </summary>
    private const double GapWeightPerSecond = 0.02;

    /// <summary>
    /// Clusters the spans long enough to be trusted, then attaches the short ones to whichever
    /// of those they most resemble.
    /// <para>
    /// Short turns are not noise — "Good.", "Right.", "Mm." are most of what a conversation is
    /// made of — but they are too brief to identify on their own. Left in the clustering they
    /// invent speakers: half a second of one word became a third person in a two-person
    /// recording. Attached afterwards, they go to the nearest voice actually established
    /// elsewhere, which is the answer a listener would give.
    /// </para>
    /// </summary>
    private static int[] ClusterWithShortSpansAttached(
        List<float[]> embeddings,
        List<(double Start, double End)> spans,
        double threshold,
        int? maxSpeakers,
        int? exactSpeakers)
    {
        var reliable = Enumerable.Range(0, spans.Count)
            .Where(i => spans[i].End - spans[i].Start >= ReliableTurnSeconds)
            .ToList();

        // Nothing long enough to anchor on: cluster everything and take what comes.
        if (reliable.Count < 2)
        {
            return SpeakerClustering.Cluster(embeddings, threshold, maxSpeakers, exactSpeakers);
        }

        var reliableLabels = SpeakerClustering.Cluster(
            reliable.Select(i => embeddings[i]).ToList(), threshold, maxSpeakers, exactSpeakers);

        var labels = new int[spans.Count];
        for (var i = 0; i < reliable.Count; i++)
        {
            labels[reliable[i]] = reliableLabels[i];
        }

        for (var i = 0; i < spans.Count; i++)
        {
            if (reliable.Contains(i))
            {
                continue;
            }

            var nearest = 0;
            var best = double.MaxValue;

            for (var r = 0; r < reliable.Count; r++)
            {
                var distance = SpeakerClustering.CosineDistance(
                    Unit(embeddings[i]), Unit(embeddings[reliable[r]]));

                // Weighted by how far away in time it is, gently. Half a second of speech does
                // not identify anyone confidently, and when the voice is ambiguous the next best
                // evidence is who was talking either side of it — an interjection belongs to the
                // conversation around it far more often than to someone across the recording.
                var gap = Math.Max(0, Math.Max(
                    spans[reliable[r]].Start - spans[i].End,
                    spans[i].Start - spans[reliable[r]].End));

                var score = distance + (GapWeightPerSecond * gap);

                if (score < best)
                {
                    best = score;
                    nearest = reliableLabels[r];
                }
            }

            labels[i] = nearest;
        }

        return labels;
    }

    private static float[] Unit(float[] vector)
    {
        var sum = 0.0;
        foreach (var value in vector)
        {
            sum += value * value;
        }

        var magnitude = Math.Sqrt(sum);

        return magnitude < 1e-12 ? vector : vector.Select(v => (float)(v / magnitude)).ToArray();
    }

    /// <summary>
    /// Runs the segmentation model across the recording and collects every stretch where one
    /// local speaker was active.
    /// <para>
    /// Kept separate, one span per speaker per window, and deliberately not merged. Merging them
    /// into a single timeline first is the obvious thing to do and it is wrong: two people
    /// either side of a boundary become one span, and every speaker change inside a window
    /// disappears before anything has a chance to notice it. What survives is a recording that
    /// looks like one long turn each.
    /// </para>
    /// <para>
    /// Local speaker numbers are still meaningless between windows — the embeddings decide
    /// identity — but within a window they separate voices, and that is worth keeping.
    /// </para>
    /// </summary>
    private List<(double Start, double End)> CollectSpeechSpans(
        PcmAudio audio,
        IProgress<double>? progress,
        CancellationToken cancellationToken)
    {
        var shift = (int)(_windowSamples * WindowShiftFraction);
        var spans = new List<(double Start, double End)>();

        for (var offset = 0; offset < audio.Samples.Length; offset += shift)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Scanning is most of the work, so it gets most of the bar. The embeddings that
            // follow are a few dozen short runs and finish quickly by comparison.
            progress?.Report(ScanShare * offset / audio.Samples.Length);

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

        // Windows overlap by ninety percent, so the same speech arrives many times over. Near
        // duplicates are dropped rather than merged: merging would join neighbours across a
        // speaker change, which is the thing this is avoiding.
        return Deduplicate(spans);
    }

    /// <summary>
    /// Drops spans that say the same thing as one already kept. Two spans covering nearly the
    /// same seconds come from adjacent windows seeing the same voice; the longer is the better
    /// look at it.
    /// </summary>
    private static List<(double Start, double End)> Deduplicate(List<(double Start, double End)> spans)
    {
        var ordered = spans
            .OrderByDescending(s => s.End - s.Start)
            .ToList();

        var kept = new List<(double Start, double End)>();

        foreach (var span in ordered)
        {
            var duration = span.End - span.Start;

            var covered = kept.Any(k =>
            {
                var overlap = Math.Min(k.End, span.End) - Math.Max(k.Start, span.Start);
                return overlap > 0 && overlap >= duration * 0.75;
            });

            if (!covered)
            {
                kept.Add(span);
            }
        }

        return kept.OrderBy(s => s.Start).ToList();
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
    /// Joins consecutive turns the clustering gave the same speaker, and only those.
    /// <para>
    /// The gap allowed is small on purpose. Someone pausing for breath mid-thought should read
    /// as one turn, but a second of silence is also exactly where the other person starts
    /// talking, and joining across that is how a speaker change gets lost after surviving
    /// everything else.
    /// </para>
    /// </summary>
    private static IReadOnlyList<SpeakerTurn> Tidy(List<SpeakerTurn> turns)
    {
        var merged = new List<SpeakerTurn>();

        foreach (var turn in turns)
        {
            if (merged.Count > 0
                && merged[^1].Speaker == turn.Speaker
                && turn.StartSeconds - merged[^1].EndSeconds < 0.75)
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
    /// Attaches speakers to a transcript, splitting segments that span more than one turn.
    /// The work is in <see cref="SpeakerAttribution"/>, which is pure and tested.
    /// </summary>
    public static IReadOnlyList<TranscriptSegment> Attribute(
        IReadOnlyList<TranscriptSegment> segments,
        IReadOnlyList<SpeakerTurn> turns) =>
        SpeakerAttribution.Apply(segments, turns);

    private static int Read(IReadOnlyDictionary<string, string> metadata, string key, int fallback) =>
        metadata.TryGetValue(key, out var value) && int.TryParse(value, out var parsed) ? parsed : fallback;

    public void Dispose()
    {
        _segmentation.Dispose();
        _embedding.Dispose();
    }
}
