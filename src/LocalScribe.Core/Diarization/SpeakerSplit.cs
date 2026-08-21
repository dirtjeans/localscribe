namespace LocalScribe.Core.Diarization;

/// <summary>
/// Splits one speaker label in two, given one paragraph the user has identified as somebody
/// else.
/// <para>
/// The counterpart to renaming everywhere. Merging two labels into one is a single action
/// because the user supplies the whole answer — these are the same person — but splitting is
/// the harder direction: knowing that one paragraph was misattributed says nothing about which
/// of the other thirty were. Doing it by hand means renaming each one, and re-running the whole
/// diarization with a higher speaker count throws away every correction already made.
/// </para>
/// <para>
/// So the user's paragraph is treated as a labelled example, and the rest are sorted against it
/// by voice. That is one click for what was thirty, and it uses the one thing the user knows
/// that the clustering does not.
/// </para>
/// </summary>
public static class SpeakerSplit
{
    /// <param name="JoinsExample">
    /// Indexes into the candidate list, of the ones that belong with the example.
    /// </param>
    /// <param name="Separation">
    /// How much farther apart the two groups are than each is scattered within itself. One
    /// means no structure at all; the two-speaker sample measures about two.
    /// </param>
    /// <param name="Split">
    /// False when the candidates do not divide into two voices, in which case
    /// <paramref name="JoinsExample"/> is empty and nothing should be renamed but the example.
    /// </param>
    public sealed record Result(IReadOnlyList<int> JoinsExample, double Separation, bool Split);

    /// <summary>
    /// Sorts <paramref name="candidates"/> into the example's voice and the other one.
    /// </summary>
    /// <param name="example">Embedding of the paragraph the user identified.</param>
    /// <param name="candidates">Embeddings of the other paragraphs sharing the label.</param>
    /// <param name="threshold">
    /// Cosine distance past which two embeddings are different people. Used only when there is
    /// a single candidate and so nothing to measure scatter against.
    /// </param>
    public static Result ByExample(
        float[] example,
        IReadOnlyList<float[]> candidates,
        double threshold = SpeakerClustering.DefaultThreshold)
    {
        ArgumentNullException.ThrowIfNull(example);
        ArgumentNullException.ThrowIfNull(candidates);

        if (candidates.Count == 0)
        {
            return new Result([], 0, false);
        }

        // Scaled to unit length: CosineDistance is 1 - dot, which is a cosine only when both
        // sides already have length one. Handing it raw embeddings returns a number that is not
        // a distance at all — on real turns it came back as -3.9.
        var points = new List<float[]> { Unit(example) };
        points.AddRange(candidates.Select(Unit));

        // The same clustering the diarizer uses, asked for two groups. Average linkage over
        // every pair holds up on a handful of noisy points where splitting about two centroids
        // does not: on the two-speaker sample one of five turns embeds closer to the other
        // speaker than to its own, and that single point drags a centroid across the gap.
        var labels = SpeakerClustering.Cluster(points, threshold, exactSpeakers: 2);

        var mine = labels[0];
        var joined = new List<int>();

        for (var i = 1; i < labels.Length; i++)
        {
            if (labels[i] == mine)
            {
                joined.Add(i - 1);
            }
        }

        var (separation, apart) = SeparationOf(points, labels, threshold);

        // Refuse to split what does not divide. Asking for two groups always returns two, however
        // alike the voices are, and a user can be wrong about a paragraph; forcing a split on a
        // single voice would scatter a correctly-labelled speaker across two names.
        //
        // Measured as a ratio rather than against a fixed distance, because the scale is not
        // knowable in advance. The threshold that separates individual embeddings is calibrated
        // for exactly that and means nothing applied to group centroids, which sit systematically
        // closer together; comparing the two was why a clean two-speaker split scored 0.338
        // against a 0.42 bar and was refused.
        // Two conditions, because each catches what the other misses. The ratio asks whether
        // there is any structure here; it can be fooled by a set of near-identical embeddings,
        // where both figures approach zero and their quotient stops meaning anything. The
        // absolute distance asks whether the structure is large enough to be two people rather
        // than one person's variation between sentences.
        var divides = separation >= MinimumSeparation
            && apart >= threshold * MinimumShareOfThreshold;

        return divides
            ? new Result(joined, separation, true)
            : new Result([], separation, false);
    }

    /// <summary>
    /// How many times farther apart the groups must be than they are internally scattered.
    /// Below this there is one voice being cut in half.
    /// </summary>
    private const double MinimumSeparation = 1.5;

    /// <summary>
    /// How much of the different-speaker threshold the groups must be apart in absolute terms.
    /// Below one because the figure is a mean over every crossing pair, which includes the
    /// closest ones, so it sits under the distance at which a single pair would be called two
    /// people. The two-speaker sample measures 0.44 against a 0.42 threshold.
    /// </summary>
    private const double MinimumShareOfThreshold = 0.7;

    /// <summary>
    /// Mean distance between the groups over mean distance within them.
    /// <para>
    /// With a single candidate there are no within-group pairs to measure, so the one distance
    /// available is compared against the calibrated threshold instead and expressed on the same
    /// scale as the ratio, so that callers have one number to reason about.
    /// </para>
    /// </summary>
    private static (double Ratio, double Apart) SeparationOf(
        List<float[]> points,
        int[] labels,
        double threshold)
    {
        var within = new List<double>();
        var between = new List<double>();

        for (var i = 0; i < points.Count; i++)
        {
            for (var j = i + 1; j < points.Count; j++)
            {
                var distance = SpeakerClustering.CosineDistance(points[i], points[j]);
                ((labels[i] == labels[j]) ? within : between).Add(distance);
            }
        }

        if (between.Count == 0)
        {
            return (0, 0);
        }

        var apart = between.Average();

        if (within.Count == 0)
        {
            return (apart / threshold, apart);
        }

        var scatter = within.Average();

        // Identical embeddings would divide by zero; the absolute test decides those.
        return (scatter < 1e-9 ? double.PositiveInfinity : apart / scatter, apart);
    }

    /// <summary>Scales to unit length, so that a dot product is a cosine.</summary>
    private static float[] Unit(float[] vector)
    {
        var sum = 0.0;
        foreach (var value in vector)
        {
            sum += value * value;
        }

        var magnitude = Math.Sqrt(sum);
        if (magnitude < 1e-12)
        {
            return (float[])vector.Clone();
        }

        var scaled = new float[vector.Length];
        for (var i = 0; i < vector.Length; i++)
        {
            scaled[i] = (float)(vector[i] / magnitude);
        }

        return scaled;
    }
}
