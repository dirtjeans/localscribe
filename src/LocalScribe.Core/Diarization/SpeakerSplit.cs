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
    /// Mean cosine distance between the two groups. Two voices measure about 0.29; one voice cut
    /// arbitrarily in half measures about 0.03.
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
    /// Cosine distance past which two embeddings are different people, used for the clustering
    /// itself. Whether the result is worth acting on is a separate question with a separate
    /// number — see <see cref="MinimumSeparation"/>.
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

        var separation = BetweenGroups(points, labels);

        // Refuse to split what does not divide. Asking for two groups always returns two, however
        // alike the voices are, and a user can be wrong about a paragraph; forcing a split on a
        // single voice would scatter a correctly-labelled speaker across two names.
        return separation >= MinimumSeparation
            ? new Result(joined, separation, true)
            : new Result([], separation, false);
    }

    /// <summary>
    /// How far apart the two groups must be, as a mean cosine distance, to be two people.
    /// <para>
    /// Measured over paragraph-length embeddings on a real recording, which is what this works
    /// on. Two people sit 0.78 apart at their closest 0.70; one person's own paragraphs sit at
    /// 0.25, at worst 0.36. This is the middle of that gap.
    /// </para>
    /// <para>
    /// It was 0.15, and before that 0.7 of the clustering threshold. Both were fitted to
    /// embeddings computed without mean normalisation, where the recording channel dominated the
    /// voice and every distance was squeezed towards zero. Normalising quadrupled the scale, so
    /// every number fitted to the old one had to be measured again.
    /// </para>
    /// </summary>
    private const double MinimumSeparation = 0.5;

    /// <summary>Mean distance between the groups, or zero when everything landed in one.</summary>
    private static double BetweenGroups(List<float[]> points, int[] labels)
    {
        var between = new List<double>();

        for (var i = 0; i < points.Count; i++)
        {
            for (var j = i + 1; j < points.Count; j++)
            {
                if (labels[i] != labels[j])
                {
                    between.Add(SpeakerClustering.CosineDistance(points[i], points[j]));
                }
            }
        }

        return between.Count == 0 ? 0 : between.Average();
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
