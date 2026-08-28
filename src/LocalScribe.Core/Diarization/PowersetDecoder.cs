namespace LocalScribe.Core.Diarization;

/// <summary>
/// Turns the segmentation model's class scores into per-speaker activity.
/// <para>
/// pyannote's segmentation model does not emit one score per speaker. It emits one score per
/// <em>combination</em> of speakers — silence, each speaker alone, and each pair talking at once
/// — which is what lets a single argmax express overlapping speech instead of having to pick a
/// winner. Three speakers with at most two overlapping gives seven classes.
/// </para>
/// <para>
/// So the class index has to be expanded back into a set before anything downstream can use it.
/// The mapping is not arbitrary and is not stored in the model: it is the combinations in
/// order, shortest first, which is the order pyannote builds them in.
/// </para>
/// </summary>
public static class PowersetDecoder
{
    /// <summary>
    /// Builds the class-to-speakers mapping, e.g. for 3 speakers overlapping at most 2:
    /// <c>{}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}</c>.
    /// </summary>
    /// <param name="speakers">Local speakers per window, 3 in the published model.</param>
    /// <param name="maxOverlap">How many may talk at once, 2 in the published model.</param>
    public static IReadOnlyList<IReadOnlyList<int>> Mapping(int speakers, int maxOverlap)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(speakers, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxOverlap, 1);

        var classes = new List<IReadOnlyList<int>> { Array.Empty<int>() };

        for (var size = 1; size <= maxOverlap; size++)
        {
            classes.AddRange(Combinations(speakers, size));
        }

        return classes;
    }

    private static IEnumerable<IReadOnlyList<int>> Combinations(int speakers, int size)
    {
        var indices = new int[size];

        IEnumerable<IReadOnlyList<int>> Build(int position, int start)
        {
            if (position == size)
            {
                yield return indices.ToArray();
                yield break;
            }

            for (var i = start; i < speakers; i++)
            {
                indices[position] = i;

                foreach (var result in Build(position + 1, i + 1))
                {
                    yield return result;
                }
            }
        }

        return Build(0, 0);
    }

    /// <summary>
    /// Expands a window of class scores into per-speaker activity.
    /// </summary>
    /// <param name="scores">Frames × classes, frame-major, as the model returns them.</param>
    /// <param name="frames">Number of frames.</param>
    /// <param name="mapping">From <see cref="Mapping"/>.</param>
    /// <param name="speakers">Local speaker count.</param>
    /// <returns>Frames × speakers of booleans, frame-major.</returns>
    public static bool[] Decode(
        ReadOnlySpan<float> scores,
        int frames,
        IReadOnlyList<IReadOnlyList<int>> mapping,
        int speakers)
    {
        ArgumentNullException.ThrowIfNull(mapping);

        var classes = mapping.Count;
        var active = new bool[frames * speakers];

        for (var frame = 0; frame < frames; frame++)
        {
            var offset = frame * classes;
            var best = 0;
            var bestScore = float.NegativeInfinity;

            for (var c = 0; c < classes; c++)
            {
                if (scores[offset + c] > bestScore)
                {
                    bestScore = scores[offset + c];
                    best = c;
                }
            }

            foreach (var speaker in mapping[best])
            {
                active[(frame * speakers) + speaker] = true;
            }
        }

        return active;
    }

    /// <summary>
    /// Runs of frames on which at least two local speakers are active at once — the model's own
    /// crosstalk testimony. The overlap classes exist precisely so the powerset can say "both",
    /// and this is the one place in the pipeline that reads them as such: everything downstream
    /// resolves each moment to a single winner, which is right for attribution and silent about
    /// the contest.
    /// </summary>
    /// <param name="active">From <see cref="Decode"/>, frame-major.</param>
    /// <param name="frames">Number of frames.</param>
    /// <param name="speakers">Local speaker count.</param>
    /// <returns>Half-open frame runs: first frame in the run, first frame past it.</returns>
    public static IEnumerable<(int First, int Until)> OverlappedFrames(
        bool[] active, int frames, int speakers)
    {
        ArgumentNullException.ThrowIfNull(active);

        int? runStart = null;

        for (var frame = 0; frame <= frames; frame++)
        {
            var contested = false;

            if (frame < frames)
            {
                var count = 0;

                for (var speaker = 0; speaker < speakers; speaker++)
                {
                    if (active[(frame * speakers) + speaker])
                    {
                        count++;
                    }
                }

                contested = count >= 2;
            }

            if (contested && runStart is null)
            {
                runStart = frame;
            }
            else if (!contested && runStart is { } begin)
            {
                runStart = null;
                yield return (begin, frame);
            }
        }
    }
}
