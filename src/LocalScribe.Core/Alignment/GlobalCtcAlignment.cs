namespace LocalScribe.Core.Alignment;

/// <summary>
/// Places an entire transcript onto an entire recording in one pass.
/// <para>
/// Aligning one segment at a time inside a window was tried in every configuration a window
/// has — anchored to the stamps, chained from the previous words, bias-corrected, widened on
/// evidence — and each fixed one failure by creating another, because a window is a local
/// decision about a global constraint. The transcript's text and the recording's time are
/// jointly monotonic: the first occurrence of a repeated phrase comes before the second, an
/// aside before the sentence it interrupts. A window cannot see its neighbours, so it is free
/// to lock onto the twin of a phrase; a single pass over everything cannot, because the path
/// must spend the whole text in order on the whole recording.
/// </para>
/// <para>
/// The pass is a blank-extended Viterbi, the same trellis the per-segment aligner walks, but
/// banded: full frames-by-states is a quarter of a billion cells on a seven-minute recording,
/// and almost all of them are absurd — the thousandth letter is not spoken in the first ten
/// seconds. The stamps, noisy as they are, are easily good enough to say roughly when each
/// letter falls, so the search is confined to a corridor around that guess, wide enough to
/// swallow every drift ever measured with an order of magnitude to spare. Inside the corridor
/// the stamps have no further say; the audio decides.
/// </para>
/// </summary>
public static class GlobalCtcAlignment
{
    /// <summary>
    /// Walks the whole trellis and reports where every token landed.
    /// </summary>
    /// <param name="scores">The recording's per-frame log probabilities.</param>
    /// <param name="targets">The whole transcript, as token indexes, in order.</param>
    /// <param name="blank">The token meaning "nothing in particular".</param>
    /// <param name="centerStates">
    /// The corridor's centre for each frame: the trellis state expected to be active then,
    /// non-decreasing. Derived from the stamps by the caller.
    /// </param>
    /// <param name="halfBand">How far either side of the centre the path may wander, in states.</param>
    /// <returns>One placement per target, or null when no path fits the corridor.</returns>
    public static IReadOnlyList<CtcForcedAlignment.Placement>? Align(
        AlignmentScores scores,
        IReadOnlyList<int> targets,
        int blank,
        IReadOnlyList<int> centerStates,
        int halfBand,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(scores);
        ArgumentNullException.ThrowIfNull(targets);
        ArgumentNullException.ThrowIfNull(centerStates);

        var frames = scores.Frames;
        var states = (targets.Count * 2) + 1;

        if (frames <= 0 || targets.Count == 0 || centerStates.Count != frames || halfBand < 2)
        {
            return null;
        }

        // More states than frames can carry is unplaceable text, full stop.
        if (states > (frames * 2) + 1)
        {
            return null;
        }

        var width = Math.Min(states, (halfBand * 2) + 1);

        // The corridor. Its floor and ceiling both only ever rise, so the path can never be
        // forced backwards through the text, whatever the stamps did.
        var lows = new int[frames];
        var floor = 0;

        for (var t = 0; t < frames; t++)
        {
            var low = Math.Clamp(centerStates[t] - halfBand, 0, Math.Max(0, states - width));
            floor = Math.Max(floor, low);
            lows[t] = floor;
        }

        // The path must be able to finish: the last frames' corridor has to reach the final
        // states, and the first frames' corridor has to include the start.
        if (lows[0] > 1 || lows[^1] + width < states - 1)
        {
            return null;
        }

        var previous = new double[width];
        var current = new double[width];
        var choices = new byte[(long)frames * width];

        Array.Fill(previous, double.NegativeInfinity);

        var row = scores.Between(0, 1);

        // Frame zero can be the opening blank or the first letter, nothing else.
        if (lows[0] == 0)
        {
            previous[0] = row[blank];

            if (width > 1)
            {
                previous[1] = row[Token(targets, 1, blank)];
            }
        }
        else
        {
            previous[1 - lows[0]] = row[Token(targets, 1, blank)];
        }

        for (var t = 1; t < frames; t++)
        {
            if ((t & 1023) == 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
            }

            row = scores.Between(t, 1);

            var low = lows[t];
            var slide = low - lows[t - 1];

            for (var w = 0; w < width; w++)
            {
                var state = low + w;

                if (state >= states)
                {
                    current[w] = double.NegativeInfinity;
                    continue;
                }

                // The same three roads as the per-segment trellis: stay, step from the
                // previous state, or hop the blank between two different letters.
                var stay = At(previous, w + slide, width);
                var step = At(previous, w + slide - 1, width);
                var hop = state >= 2 && Token(targets, state, blank) != blank
                    && Token(targets, state, blank) != Token(targets, state - 2, blank)
                        ? At(previous, w + slide - 2, width)
                        : double.NegativeInfinity;

                var best = stay;
                byte choice = 0;

                if (step > best)
                {
                    best = step;
                    choice = 1;
                }

                if (hop > best)
                {
                    best = hop;
                    choice = 2;
                }

                current[w] = best + row[Token(targets, state, blank)];
                choices[((long)t * width) + w] = choice;
            }

            (previous, current) = (current, previous);
        }

        // Finish on the last letter or the closing blank, whichever scored better.
        var endLow = lows[^1];
        var last = states - 1 - endLow;
        var beforeLast = states - 2 - endLow;

        var endAt = double.NegativeInfinity;
        var endState = -1;

        if (last >= 0 && last < width && previous[last] > endAt)
        {
            endAt = previous[last];
            endState = states - 1;
        }

        if (beforeLast >= 0 && beforeLast < width && previous[beforeLast] > endAt)
        {
            endAt = previous[beforeLast];
            endState = states - 2;
        }

        if (endState < 0 || double.IsNegativeInfinity(endAt))
        {
            return null;
        }

        // Walk back, turning the state path into per-token frame spans.
        var first = new int[targets.Count];
        var lastFrame = new int[targets.Count];
        Array.Fill(first, int.MaxValue);
        Array.Fill(lastFrame, int.MinValue);

        var at = endState;

        for (var t = frames - 1; t >= 0; t--)
        {
            if ((at & 1) == 1)
            {
                var token = at >> 1;
                first[token] = Math.Min(first[token], t);
                lastFrame[token] = Math.Max(lastFrame[token], t);
            }

            if (t == 0)
            {
                break;
            }

            at -= choices[((long)t * width) + (at - lows[t])];
        }

        var placements = new CtcForcedAlignment.Placement[targets.Count];

        for (var i = 0; i < targets.Count; i++)
        {
            // A token the path never visited was skipped by a hop, which the trellis only
            // allows over blanks — it cannot happen to a letter. Guard anyway: a zero-length
            // placement at its neighbour beats an exception.
            var from = first[i] == int.MaxValue ? (i > 0 ? lastFrame[i - 1] : 0) : first[i];
            var to = lastFrame[i] == int.MinValue ? from : lastFrame[i];

            placements[i] = new CtcForcedAlignment.Placement(targets[i], Math.Max(0, from), Math.Max(0, to), 0);
        }

        return placements;
    }

    private static int Token(IReadOnlyList<int> targets, int state, int blank) =>
        (state & 1) == 1 ? targets[state >> 1] : blank;

    private static double At(double[] rowValues, int index, int width) =>
        index >= 0 && index < width ? rowValues[index] : double.NegativeInfinity;
}
