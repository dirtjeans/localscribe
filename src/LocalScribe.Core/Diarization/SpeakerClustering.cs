namespace LocalScribe.Core.Diarization;

/// <summary>
/// Groups speaker embeddings into speakers.
/// <para>
/// The segmentation model works a window at a time and numbers the voices it hears within each
/// one. Those numbers mean nothing across windows: speaker 0 in the third window and speaker 0
/// in the fourth are unrelated. Embeddings are what carry identity between windows, and
/// clustering them is what turns a pile of local guesses into a recording with two people in it.
/// </para>
/// <para>
/// Agglomerative, on cosine distance, stopping at a threshold rather than a count — because the
/// number of speakers is exactly what nobody knows in advance. Average linkage, so one unusual
/// embedding cannot drag a cluster across the threshold on its own.
/// </para>
/// </summary>
public static class SpeakerClustering
{
    /// <summary>
    /// Cosine distance past which two embeddings are treated as different people.
    /// <para>
    /// Empirical, and the single number most worth tuning: too low splits one speaker into
    /// several, too high merges two into one. Splitting is the kinder failure — a transcript
    /// with "Speaker 3" appearing spuriously is readable, one that attributes a sentence to the
    /// wrong person is worse than one with no labels at all, and a label is now something the
    /// user can correct in one click.
    /// </para>
    /// <para>
    /// Measured rather than guessed. On a two-speaker recording, 0.30 split them into four,
    /// 0.50 merged them into one, and everything between 0.40 and 0.45 attributed every turn
    /// correctly. This sits in the middle of that band, a little towards splitting.
    /// </para>
    /// <para>
    /// It was calibrated against synthesised voices, which are cleaner and more consistent than
    /// people, so real recordings may want it moved. The doctor takes --threshold for exactly
    /// that.
    /// </para>
    /// </summary>
    public const double DefaultThreshold = 0.42;

    /// <summary>
    /// Assigns each embedding a speaker index.
    /// </summary>
    /// <param name="embeddings">One vector per item, all the same length.</param>
    /// <param name="threshold">Cosine distance at which merging stops.</param>
    /// <param name="maxSpeakers">
    /// Upper bound, or null for none. Merging continues past the threshold if it would otherwise
    /// leave more clusters than this.
    /// </param>
    /// <returns>A speaker index per embedding, numbered by first appearance.</returns>
    public static int[] Cluster(
        IReadOnlyList<float[]> embeddings,
        double threshold = DefaultThreshold,
        int? maxSpeakers = null)
    {
        ArgumentNullException.ThrowIfNull(embeddings);

        if (embeddings.Count == 0)
        {
            return [];
        }

        if (embeddings.Count == 1)
        {
            return [0];
        }

        var normalised = embeddings.Select(Normalise).ToList();
        var clusters = Enumerable.Range(0, normalised.Count).Select(i => new List<int> { i }).ToList();

        while (clusters.Count > 1)
        {
            var (a, b, distance) = ClosestPair(clusters, normalised);

            var forced = maxSpeakers is { } limit && clusters.Count > limit;

            if (distance > threshold && !forced)
            {
                break;
            }

            clusters[a].AddRange(clusters[b]);
            clusters.RemoveAt(b);
        }

        return Label(clusters, normalised.Count);
    }

    /// <summary>The two clusters with the smallest average distance between their members.</summary>
    private static (int A, int B, double Distance) ClosestPair(
        List<List<int>> clusters,
        List<float[]> embeddings)
    {
        var bestA = 0;
        var bestB = 1;
        var best = double.MaxValue;

        for (var i = 0; i < clusters.Count; i++)
        {
            for (var j = i + 1; j < clusters.Count; j++)
            {
                var total = 0.0;

                foreach (var left in clusters[i])
                {
                    foreach (var right in clusters[j])
                    {
                        total += CosineDistance(embeddings[left], embeddings[right]);
                    }
                }

                var average = total / (clusters[i].Count * clusters[j].Count);

                if (average < best)
                {
                    best = average;
                    bestA = i;
                    bestB = j;
                }
            }
        }

        return (bestA, bestB, best);
    }

    /// <summary>
    /// Numbers clusters by when they are first heard, so "Speaker 1" is whoever spoke first
    /// rather than whichever cluster happened to be built first.
    /// </summary>
    private static int[] Label(List<List<int>> clusters, int count)
    {
        var labels = new int[count];
        var ordered = clusters.OrderBy(c => c.Min()).ToList();

        for (var speaker = 0; speaker < ordered.Count; speaker++)
        {
            foreach (var index in ordered[speaker])
            {
                labels[index] = speaker;
            }
        }

        return labels;
    }

    /// <summary>Cosine distance in [0, 2]. Zero means identical direction.</summary>
    public static double CosineDistance(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        var dot = 0.0;

        for (var i = 0; i < left.Length && i < right.Length; i++)
        {
            dot += left[i] * right[i];
        }

        return 1.0 - dot;
    }

    /// <summary>Scales to unit length, so a dot product is a cosine.</summary>
    private static float[] Normalise(float[] vector)
    {
        var sum = 0.0;
        foreach (var value in vector)
        {
            sum += value * value;
        }

        var magnitude = Math.Sqrt(sum);
        if (magnitude < 1e-12)
        {
            return vector;
        }

        var result = new float[vector.Length];
        for (var i = 0; i < vector.Length; i++)
        {
            result[i] = (float)(vector[i] / magnitude);
        }

        return result;
    }
}
