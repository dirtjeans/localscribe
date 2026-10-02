using System.Diagnostics;
using LocalScribe.Core.Audio;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Hardware;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalScribe.Onnx;

/// <summary>
/// Works out who spoke when with one end-to-end model, NVIDIA's Nemotron 3 Diarization, in
/// place of the segmentation, embedding and clustering that <see cref="SpeakerDiarizer"/>
/// chains together.
/// <para>
/// A trial, reached only through the doctor. The pyannote path is tuned and frozen and the
/// app still runs it; this exists so the two can be measured on the same recordings before
/// anything is decided.
/// </para>
/// <para>
/// The difference in kind matters more than any benchmark. The pipeline cuts speech into
/// stretches, turns each into a voice vector and clusters the vectors, so its speaker count
/// is a threshold's opinion — and on a poor recording the vectors scatter and one person
/// becomes nineteen. This model emits eight activity channels directly and was trained to
/// keep each person on one channel, so there is no threshold to scatter across; the price is
/// a ceiling of eight people and no knob to turn when it is wrong.
/// </para>
/// <para>
/// Two graphs. The first folds eight frames of log-mel into one embedding, 80 ms apiece. The
/// second is the transformer: it reads what <see cref="ArrivalOrderSpeakerCache"/> remembers,
/// then 27 seconds of new audio and three seconds of lookahead, and answers at the original
/// 10 ms. It runs on the CPU for the same reason the pyannote models do — dynamic shapes —
/// and under the same thread budget.
/// </para>
/// </summary>
public sealed class SortformerDiarizer : IDisposable
{
    /// <summary>Embedding frames per step: 27.2 seconds. The model's offline configuration.</summary>
    private const int ChunkFrames = 340;

    /// <summary>Embedding frames of lookahead past each chunk: 3.2 seconds.</summary>
    private const int LookaheadFrames = 40;

    /// <summary>Feature frames folded into each embedding.</summary>
    private const int Fold = 8;

    private const int EmbeddingSize = 512;
    private const int Speakers = ArrivalOrderSpeakerCache.Speakers;

    private readonly InferenceSession _embed;
    private readonly InferenceSession _step;
    private readonly PreEmphasisLogMel _features;
    private readonly float[] _silence;

    private SortformerDiarizer(
        InferenceSession embed,
        InferenceSession step,
        PreEmphasisLogMel features,
        float[] silence)
    {
        _embed = embed;
        _step = step;
        _features = features;
        _silence = silence;
    }

    /// <summary>Where the time went on the last run, for the doctor's report.</summary>
    public (TimeSpan Features, TimeSpan Embedding, TimeSpan Transformer) LastTimings { get; private set; }

    /// <summary>Loads the model from a directory holding the ONNX export.</summary>
    /// <param name="plan">The CPU thread budget, or null to let ONNX Runtime take every core.</param>
    /// <param name="fullPrecision">
    /// Use the float32 transformer rather than the int8 one. Four times the size on disk; which
    /// is faster on a given processor is a measurement, not a given.
    /// </param>
    public static SortformerDiarizer Load(
        string modelDirectory,
        ExecutionPlan? plan = null,
        bool fullPrecision = false)
    {
        using var options = new SessionOptions();

        if (plan is not null)
        {
            options.IntraOpNumThreads = plan.CpuBudget.IntraOpThreads;
            options.InterOpNumThreads = plan.CpuBudget.InterOpThreads;
        }

        var filters = Floats(Require(modelDirectory, "mel_filters.bin"));
        var silence = Floats(Require(modelDirectory, "silence_embeds.bin"));

        if (silence.Length != EmbeddingSize)
        {
            throw new InvalidDataException(
                $"silence_embeds.bin holds {silence.Length} values; the model's embeddings have {EmbeddingSize}.");
        }

        var embed = new InferenceSession(Require(modelDirectory, "embed.onnx"), options);

        try
        {
            var step = new InferenceSession(
                Require(modelDirectory, fullPrecision ? "step.onnx" : "step_int8.onnx"),
                options);

            return new SortformerDiarizer(embed, step, new PreEmphasisLogMel(filters), silence);
        }
        catch
        {
            embed.Dispose();
            throw;
        }
    }

    /// <summary>Hears a whole recording and reports each speaker's activity, frame by frame.</summary>
    /// <param name="audio">16 kHz mono.</param>
    public SpeakerActivity Diarize(
        PcmAudio audio,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);

        var frames = PreEmphasisLogMel.FrameCount(audio.Samples.Length);
        var embeddings = (frames + Fold - 1) / Fold;

        var cache = new ArrivalOrderSpeakerCache(_silence);
        var activity = new float[embeddings * Fold * Speakers];
        var written = 0;

        var featureTime = TimeSpan.Zero;
        var embedTime = TimeSpan.Zero;
        var stepTime = TimeSpan.Zero;

        for (var start = 0; start < embeddings; start += ChunkFrames)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var end = Math.Min(start + ChunkFrames, embeddings);
            var chunk = end - start;

            var fresh = Embed(
                audio.Samples,
                start,
                Math.Min(end + LookaheadFrames, embeddings),
                frames,
                ref featureTime,
                ref embedTime);

            var context = cache.Context();
            List<float[]> input = [.. context, .. fresh];

            var packed = new float[input.Count * EmbeddingSize];
            for (var i = 0; i < input.Count; i++)
            {
                input[i].CopyTo(packed, i * EmbeddingSize);
            }

            var watch = Stopwatch.StartNew();

            using var outputs = _step.Run(
            [
                NamedOnnxValue.CreateFromTensor(
                    "embeds",
                    new DenseTensor<float>(packed, [1, input.Count, EmbeddingSize])),
            ]);

            var logits = outputs.First().AsTensor<float>().ToArray();
            stepTime += watch.Elapsed;

            if (logits.Length != input.Count * Fold * Speakers)
            {
                throw new InvalidDataException(
                    $"The transformer answered with {logits.Length} values for {input.Count} frames; "
                    + $"expected {input.Count * Fold * Speakers}.");
            }

            // The model answers at 10 ms, but the cache thinks in 80 ms embedding frames, so its
            // opinion of each embedding is the average over the eight answers that frame covers.
            var pooled = new float[input.Count][];

            for (var frame = 0; frame < input.Count; frame++)
            {
                var row = pooled[frame] = new float[Speakers];

                for (var fine = 0; fine < Fold; fine++)
                {
                    var at = ((frame * Fold) + fine) * Speakers;

                    for (var speaker = 0; speaker < Speakers; speaker++)
                    {
                        row[speaker] += Sigmoid(logits[at + speaker]);
                    }
                }

                for (var speaker = 0; speaker < Speakers; speaker++)
                {
                    row[speaker] /= Fold;
                }
            }

            cache.Update(input, pooled, chunk);

            // Only the chunk's own frames are kept. The context ahead of them is memory being
            // re-read, and the lookahead behind them will be the next chunk's to answer with
            // more to go on.
            var from = context.Count * Fold * Speakers;
            var to = (context.Count + chunk) * Fold * Speakers;

            for (var i = from; i < to; i++)
            {
                activity[written++] = Sigmoid(logits[i]);
            }

            progress?.Report(end / (double)embeddings);
        }

        LastTimings = (featureTime, embedTime, stepTime);

        return new SpeakerActivity(activity, frames, Speakers);
    }

    /// <summary>
    /// Embeddings for one range. Each depends only on its own eight feature frames, so a range
    /// at a time gives exactly what the whole file at once would.
    /// </summary>
    private List<float[]> Embed(
        float[] samples,
        int first,
        int end,
        int frames,
        ref TimeSpan featureTime,
        ref TimeSpan embedTime)
    {
        var firstFrame = first * Fold;
        var endFrame = Math.Min(end * Fold, frames);

        var watch = Stopwatch.StartNew();
        var features = _features.Frames(samples, firstFrame, endFrame);
        featureTime += watch.Elapsed;

        watch.Restart();

        using var outputs = _embed.Run(
        [
            NamedOnnxValue.CreateFromTensor(
                "features",
                new DenseTensor<float>(features, [1, endFrame - firstFrame, _features.MelBins])),
        ]);

        var flat = outputs.First().AsTensor<float>().ToArray();
        embedTime += watch.Elapsed;

        var count = flat.Length / EmbeddingSize;
        var embeddings = new List<float[]>(count);

        for (var i = 0; i < count; i++)
        {
            embeddings.Add(flat.AsSpan(i * EmbeddingSize, EmbeddingSize).ToArray());
        }

        return embeddings;
    }

    /// <summary>
    /// A voice print per speaker, from the voice model the pyannote pipeline uses, for when the
    /// user's speaker count has to be met by merging.
    /// <para>
    /// From each speaker's longest stretches that nobody else talks over — five at most, a
    /// second and a half at least — averaged on the unit sphere. One stretch is a poor print;
    /// a stretch with someone else in it is a print of both.
    /// </para>
    /// </summary>
    public static IReadOnlyDictionary<int, float[]> VoicePrints(
        SpeakerDiarizer voices,
        PcmAudio audio,
        IReadOnlyList<SpeakerTurn> turns,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(voices);
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(turns);

        var prints = new Dictionary<int, float[]>();

        foreach (var speaker in turns.GroupBy(t => t.Speaker))
        {
            cancellationToken.ThrowIfCancellationRequested();

            var clean = speaker
                .Where(turn => turn.DurationSeconds >= 1.5)
                .Where(turn => !turns.Any(other =>
                    other.Speaker != turn.Speaker && other.OverlapWith(turn.StartSeconds, turn.EndSeconds) > 0))
                .OrderByDescending(turn => turn.DurationSeconds)
                .Take(5);

            float[]? sum = null;

            foreach (var turn in clean)
            {
                if (voices.EmbedSpan(audio, turn.StartSeconds, turn.EndSeconds) is not { } embedding)
                {
                    continue;
                }

                var length = Math.Sqrt(embedding.Sum(x => (double)x * x));
                if (length < 1e-9)
                {
                    continue;
                }

                sum ??= new float[embedding.Length];
                for (var i = 0; i < embedding.Length; i++)
                {
                    sum[i] += (float)(embedding[i] / length);
                }
            }

            if (sum is not null)
            {
                prints[speaker.Key] = sum;
            }
        }

        return prints;
    }

    private static float Sigmoid(float logit) => (float)(1 / (1 + Math.Exp(-logit)));

    private static float[] Floats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, values.Length * sizeof(float));
        return values;
    }

    private static string Require(string directory, string name)
    {
        var path = Path.Combine(directory, name);

        return File.Exists(path)
            ? path
            : throw new FileNotFoundException(
                $"No {name} in {directory}. The Nemotron trial reads an ONNX export of "
                + "nvidia/Nemotron-3-Diarization: embed.onnx, step_int8.onnx (or step.onnx), "
                + "mel_filters.bin and silence_embeds.bin.",
                path);
    }

    public void Dispose()
    {
        _embed.Dispose();
        _step.Dispose();
    }
}
