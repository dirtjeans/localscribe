namespace LocalScribe.Core.Alignment;

/// <summary>
/// Finds where each letter of a known transcript was said, given a CTC model's frame-by-frame
/// opinion of what it heard.
/// <para>
/// The transcript is already known, so this is not recognition — it is the far easier problem of
/// deciding which frames go with which letters. A CTC model emits, for every twenty milliseconds
/// of audio, a score for each letter and for "nothing in particular". The alignment is the path
/// through those frames that spells the transcript and scores highest, which is a Viterbi search
/// over a trellis: at each frame the path either stays on the current letter, or moves to the
/// next one.
/// </para>
/// <para>
/// This is what Whisper cannot give us. Its exported graphs emit no cross-attention, so the
/// usual way of extracting word times from Whisper itself is closed, and word times were being
/// estimated from loudness. Those estimates are good to about half a second; this is good to
/// about the length of one frame.
/// </para>
/// </summary>
public static class CtcForcedAlignment
{
    /// <param name="Token">Index into the model's alphabet.</param>
    /// <param name="FirstFrame">First frame this letter occupies.</param>
    /// <param name="LastFrame">Last frame it occupies, inclusive.</param>
    /// <param name="Score">Mean log probability along the way, as a confidence.</param>
    public sealed record Placement(int Token, int FirstFrame, int LastFrame, double Score);

    /// <summary>
    /// Places every token of <paramref name="targets"/> onto a frame.
    /// </summary>
    /// <param name="logProbabilities">
    /// Frame-major log probabilities: frame 0's whole alphabet, then frame 1's, and so on.
    /// </param>
    /// <param name="frames">How many frames there are.</param>
    /// <param name="alphabet">How many tokens the model knows.</param>
    /// <param name="targets">The transcript, as token indexes.</param>
    /// <param name="blank">The token meaning "nothing in particular".</param>
    /// <returns>One placement per target, or null when the audio is too short to hold them.</returns>
    public static IReadOnlyList<Placement>? Align(
        ReadOnlySpan<float> logProbabilities,
        int frames,
        int alphabet,
        IReadOnlyList<int> targets,
        int blank = 0)
    {
        ArgumentNullException.ThrowIfNull(targets);

        if (frames <= 0 || alphabet <= 0 || targets.Count == 0)
        {
            return null;
        }

        // The alphabet with a blank woven through it: blank, letter, blank, letter … blank.
        // This is how CTC is normally written down and it is not decoration — a blank between
        // two of the same letter is the only thing that stops "ll" collapsing into "l", so the
        // path has to be allowed to sit on one.
        var states = (targets.Count * 2) + 1;
        var extended = new int[states];

        for (var i = 0; i < targets.Count; i++)
        {
            extended[(i * 2) + 1] = targets[i];
        }

        for (var i = 0; i < states; i += 2)
        {
            extended[i] = blank;
        }

        if (frames < LeastFramesFor(targets, blank))
        {
            return null;
        }

        var trellis = new double[frames, states];
        var cameFrom = new int[frames, states];

        for (var t = 0; t < frames; t++)
        {
            for (var s2 = 0; s2 < states; s2++)
            {
                trellis[t, s2] = double.NegativeInfinity;
            }
        }

        // A path may open on the first blank or straight onto the first letter.
        trellis[0, 0] = logProbabilities[blank];
        if (states > 1)
        {
            trellis[0, 1] = logProbabilities[extended[1]];
        }

        for (var t = 1; t < frames; t++)
        {
            var row = t * alphabet;

            for (var s2 = 0; s2 < states; s2++)
            {
                var best = trellis[t - 1, s2];
                var from = s2;

                if (s2 > 0 && trellis[t - 1, s2 - 1] > best)
                {
                    best = trellis[t - 1, s2 - 1];
                    from = s2 - 1;
                }

                // Stepping over a blank, allowed only between two different letters. Between two
                // of the same, the blank is what keeps them apart and cannot be skipped.
                if (s2 > 1
                    && extended[s2] != blank
                    && extended[s2] != extended[s2 - 2]
                    && trellis[t - 1, s2 - 2] > best)
                {
                    best = trellis[t - 1, s2 - 2];
                    from = s2 - 2;
                }

                if (double.IsNegativeInfinity(best))
                {
                    continue;
                }

                trellis[t, s2] = best + logProbabilities[row + extended[s2]];
                cameFrom[t, s2] = from;
            }
        }

        // A path may close on the last letter or on the blank after it.
        var last = states - 1;
        if (states > 1 && trellis[frames - 1, states - 2] > trellis[frames - 1, states - 1])
        {
            last = states - 2;
        }

        if (double.IsNegativeInfinity(trellis[frames - 1, last]))
        {
            return null;
        }

        var path = new int[frames];
        var at = last;

        for (var t = frames - 1; t >= 0; t--)
        {
            path[t] = at;
            at = t > 0 ? cameFrom[t, at] : at;
        }

        // Each letter owns the frames the path spent on it.
        var placements = new List<Placement>(targets.Count);

        for (var i = 0; i < targets.Count; i++)
        {
            var state = (i * 2) + 1;
            var first = -1;
            var final = -1;
            var total = 0.0;
            var counted = 0;

            for (var t = 0; t < frames; t++)
            {
                if (path[t] != state)
                {
                    continue;
                }

                if (first < 0)
                {
                    first = t;
                }

                final = t;
                total += logProbabilities[(t * alphabet) + targets[i]];
                counted++;
            }

            if (first < 0)
            {
                // The path never rested on this letter, which means the audio could not hold it.
                return null;
            }

            placements.Add(new Placement(targets[i], first, final, total / counted));
        }

        return placements;
    }

    /// <summary>
    /// The fewest frames a transcript can possibly occupy: one per letter, plus one for the
    /// blank that has to sit between any two of the same letter running together.
    /// </summary>
    private static int LeastFramesFor(IReadOnlyList<int> targets, int blank)
    {
        var least = targets.Count;

        for (var i = 1; i < targets.Count; i++)
        {
            if (targets[i] == targets[i - 1])
            {
                least++;
            }
        }

        return least;
    }
}
