namespace LocalScribe.Core.Diarization;

/// <summary>
/// What a Sortformer diarizer remembers of the recording so far, so that its speaker 3 in
/// minute forty is the same person as its speaker 3 in minute two.
/// <para>
/// The model reads one chunk at a time and has no state of its own. Identity is carried by
/// what is put in front of each chunk: a fixed budget of earlier frames, a few dozen per
/// speaker, laid out speaker by speaker in the order the speakers first arrived. The model
/// was trained to read that layout and continue the numbering. Behind it sits a short queue
/// of the most recent frames, for plain acoustic continuity across the chunk boundary.
/// </para>
/// <para>
/// When the queue overflows, its oldest frames are offered to the cache, and when the cache
/// overflows it is compressed: every frame is scored for how clearly it shows one speaker
/// alone, and each speaker keeps their best. A frame where two people talk is a poor
/// example of either, so the score rewards confidence in one speaker and silence from the
/// rest.
/// </para>
/// <para>
/// Every constant here is part of the trained model, not a preference. The network learned
/// to read a cache of exactly this shape built by exactly this rule; a "better" rule would
/// show it something it has never seen. This is a port of the reference implementation's
/// offline configuration and should change only when the reference does.
/// </para>
/// </summary>
public sealed class ArrivalOrderSpeakerCache
{
    /// <summary>Output channels of the model, and so the most people it can tell apart.</summary>
    public const int Speakers = 8;

    private const int CacheFrames = 264;
    private const int QueueFrames = 40;
    private const int UpdatePeriod = 300;
    private const int SilenceFramesPerSpeaker = 1;

    /// <summary>Probabilities are clamped here before the logarithm, so one stray zero cannot dominate.</summary>
    private const double ScoreFloor = 0.25;

    /// <summary>A small preference for frames newer than the cache, so it follows a voice as it changes.</summary>
    private const double LatestBoost = 0.05;

    private const int PerSpeaker = (CacheFrames / Speakers) - SilenceFramesPerSpeaker;
    private const int MinimumPositive = PerSpeaker / 2;
    private const int StrongFrames = PerSpeaker * 3 / 4;
    private const int WeakFrames = PerSpeaker * 3 / 2;

    private static readonly double LogHalf = Math.Log(0.5);
    private static readonly float[] NoOne = new float[Speakers];

    private readonly float[] _silence;
    private List<float[]> _embeddings = [];
    private List<float[]> _probabilities = [];
    private List<float[]> _queue = [];
    private bool _compressed;

    /// <param name="silence">
    /// The model's learned embedding of silence, which separates one speaker's block of the
    /// cache from the next. Shipped beside the weights.
    /// </param>
    public ArrivalOrderSpeakerCache(float[] silence)
    {
        ArgumentNullException.ThrowIfNull(silence);
        _silence = silence;
    }

    /// <summary>Frames the model should see ahead of the next chunk: the cache, then the queue.</summary>
    public IReadOnlyList<float[]> Context() => [.. _embeddings, .. _queue];

    /// <summary>
    /// Folds one finished step back into the memory.
    /// </summary>
    /// <param name="input">Every frame the model was just shown: context, chunk, lookahead.</param>
    /// <param name="probabilities">Its verdict on each of those frames, one value per speaker.</param>
    /// <param name="chunkFrames">How many of them were the new chunk, not counting lookahead.</param>
    public void Update(IReadOnlyList<float[]> input, IReadOnlyList<float[]> probabilities, int chunkFrames)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(probabilities);

        var cached = _embeddings.Count;
        var chunkStart = cached + _queue.Count;

        var queue = new List<float[]>(_queue);
        for (var i = chunkStart; i < Math.Min(chunkStart + chunkFrames, input.Count); i++)
        {
            queue.Add(input[i]);
        }

        var length = queue.Count;
        var leaving = length > QueueFrames
            ? Math.Min(Math.Max(UpdatePeriod, length - QueueFrames), length)
            : 0;

        if (leaving > 0)
        {
            // Until the first compression the cache holds a plain prefix of the recording, and
            // the model's fresh opinion of those frames is better than the one stored when it
            // had heard less. After compression the frames are a selection, their stored
            // opinions are what selected them, and those are kept.
            var known = _compressed ? _probabilities : [.. probabilities.Take(cached)];

            var embeddings = new List<float[]>(_embeddings);
            var verdicts = new List<float[]>(known);

            for (var i = 0; i < leaving; i++)
            {
                embeddings.Add(queue[i]);
                verdicts.Add(probabilities[cached + i]);
            }

            queue.RemoveRange(0, leaving);

            if (embeddings.Count > CacheFrames)
            {
                (embeddings, verdicts) = Compress(embeddings, verdicts);
                _compressed = true;
            }

            _embeddings = embeddings;
            _probabilities = verdicts;
        }

        _queue = queue;
    }

    /// <summary>
    /// Keeps the frames that best show each speaker alone, in speaker order, each speaker's
    /// block closed by a frame of silence.
    /// </summary>
    private (List<float[]> Embeddings, List<float[]> Probabilities) Compress(
        List<float[]> embeddings,
        List<float[]> probabilities)
    {
        var frames = probabilities.Count;
        var scores = Score(probabilities);

        for (var frame = CacheFrames; frame < frames; frame++)
        {
            for (var speaker = 0; speaker < Speakers; speaker++)
            {
                scores[frame][speaker] += LatestBoost;
            }
        }

        // Two tiers of guaranteed places. Without them one talkative speaker's merely good
        // frames outscore a quiet speaker's best, and the quiet speaker is forgotten — which
        // is exactly the person a long recording needs remembered.
        Boost(scores, StrongFrames, -2 * LogHalf);
        Boost(scores, WeakFrames, -LogHalf);

        var scored = frames + SilenceFramesPerSpeaker;
        var ranked = new (int Index, double Score)[Speakers * scored];

        for (var speaker = 0; speaker < Speakers; speaker++)
        {
            for (var frame = 0; frame < scored; frame++)
            {
                var index = (speaker * scored) + frame;
                ranked[index] = (index, frame < frames ? scores[frame][speaker] : double.PositiveInfinity);
            }
        }

        Array.Sort(ranked, (a, b) =>
        {
            var byScore = b.Score.CompareTo(a.Score);
            return byScore != 0 ? byScore : a.Index.CompareTo(b.Index);
        });

        // A place won by a frame that scored nothing is a place nobody earned; it is filled
        // with silence and sorted to the end.
        var unearned = scored * Speakers;
        var kept = ranked
            .Take(CacheFrames)
            .Select(entry => double.IsNegativeInfinity(entry.Score) ? unearned : entry.Index)
            .Order()
            .ToArray();

        var keptEmbeddings = new List<float[]>(kept.Length);
        var keptProbabilities = new List<float[]>(kept.Length);

        foreach (var index in kept)
        {
            var frame = index == unearned ? frames : Math.Min(index % scored, frames);

            keptEmbeddings.Add(frame == frames ? _silence : embeddings[frame]);
            keptProbabilities.Add(frame == frames ? NoOne : probabilities[frame]);
        }

        return (keptEmbeddings, keptProbabilities);
    }

    /// <summary>
    /// How good an example of each speaker each frame is: high when that speaker is certain
    /// and everyone else is certainly quiet, and no score at all unless the speaker is more
    /// likely talking than not.
    /// </summary>
    private static double[][] Score(List<float[]> probabilities)
    {
        var frames = probabilities.Count;
        var scores = new double[frames][];
        var positive = new int[Speakers];
        Span<double> quiet = stackalloc double[Speakers];

        for (var frame = 0; frame < frames; frame++)
        {
            var p = probabilities[frame];
            double allQuiet = 0;

            for (var speaker = 0; speaker < Speakers; speaker++)
            {
                quiet[speaker] = Math.Log(Math.Max(1.0 - p[speaker], ScoreFloor));
                allQuiet += quiet[speaker];
            }

            var row = scores[frame] = new double[Speakers];

            for (var speaker = 0; speaker < Speakers; speaker++)
            {
                var score = p[speaker] > 0.5
                    ? Math.Log(Math.Max(p[speaker], ScoreFloor)) - quiet[speaker] + allQuiet - LogHalf
                    : double.NegativeInfinity;

                row[speaker] = score;

                if (score > 0)
                {
                    positive[speaker]++;
                }
            }
        }

        // A speaker with enough clean frames does not need their muddy ones: frames where
        // they talk but so does someone else are withdrawn. A speaker without enough keeps
        // whatever they have.
        for (var frame = 0; frame < frames; frame++)
        {
            for (var speaker = 0; speaker < Speakers; speaker++)
            {
                if (!(scores[frame][speaker] > 0)
                    && probabilities[frame][speaker] > 0.5
                    && positive[speaker] >= MinimumPositive)
                {
                    scores[frame][speaker] = double.NegativeInfinity;
                }
            }
        }

        return scores;
    }

    /// <summary>Raises each speaker's best <paramref name="count"/> frames by <paramref name="amount"/>.</summary>
    private static void Boost(double[][] scores, int count, double amount)
    {
        var frames = scores.Length;
        var order = new int[frames];

        for (var speaker = 0; speaker < Speakers; speaker++)
        {
            for (var i = 0; i < frames; i++)
            {
                order[i] = i;
            }

            var column = speaker;
            Array.Sort(order, (a, b) =>
            {
                var byScore = scores[b][column].CompareTo(scores[a][column]);
                return byScore != 0 ? byScore : a.CompareTo(b);
            });

            for (var i = 0; i < Math.Min(count, frames); i++)
            {
                scores[order[i]][speaker] += amount;
            }
        }
    }
}
