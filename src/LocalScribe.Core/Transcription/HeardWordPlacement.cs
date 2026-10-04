namespace LocalScribe.Core.Transcription;

/// <summary>
/// Times a segment's words from the times the transcriber itself gave them as it heard them.
/// <para>
/// Whisper's cross-attention (DTW) word times are coarser than the aligner's — graded against
/// it, a steady lag of about 0.15 s and two-thirds of words within 0.1 s once the lag is
/// removed — but they exist the moment a window is transcribed, minutes before the scan
/// reaches it. That is the trade: the transcript is clickable as it streams, and the scan
/// replaces these times with measured ones wherever it places words.
/// </para>
/// <para>
/// Matched by text within the segment's time span rather than by which window produced
/// them, because the text on screen is rarely what came out of the decoder verbatim: windows
/// are stitched at their overlaps, looped tails trimmed, and cleanup rewrites punctuation.
/// Only the times are borrowed, the same discipline the aligner's words follow.
/// </para>
/// </summary>
public static class HeardWordPlacement
{
    /// <summary>How far outside a segment's stamps a heard word may fall and still count.</summary>
    public const double ReachSeconds = 3.0;

    /// <summary>
    /// The share of a segment's words that must be found among the heard ones. Below it the
    /// pairing is guesswork — a segment rewritten past recognition, or stamps that missed
    /// their words — and the segment stays untimed rather than wrongly timed.
    /// </summary>
    public const double LeastMatchedShare = 0.5;

    /// <summary>
    /// The segment's words, one per whitespace-separated word of its text and carrying its
    /// offset, timed from the heard words; or null when too few could be paired.
    /// </summary>
    /// <param name="segment">The segment as it will be displayed.</param>
    /// <param name="heard">Every word the transcriber heard, recording-absolute, in order.</param>
    public static IReadOnlyList<WordTimings.Word>? Place(
        TranscriptSegment segment,
        IReadOnlyList<WordTimings.Word> heard)
    {
        ArgumentNullException.ThrowIfNull(segment);
        ArgumentNullException.ThrowIfNull(heard);

        var own = Split(segment.Text);

        if (own.Count == 0 || heard.Count == 0)
        {
            return null;
        }

        var from = segment.StartSeconds - ReachSeconds;
        var to = segment.EndSeconds + ReachSeconds;

        // Sorted by time, because the windows overlap by two seconds and the seam's words
        // arrive twice, slightly out of order; a subsequence match needs one timeline.
        var nearby = heard
            .Where(word => word.StartSeconds >= from && word.StartSeconds <= to)
            .Select(word => (Folded: Fold(word.Text), word.StartSeconds))
            .Where(word => word.Folded.Length > 0)
            .OrderBy(word => word.StartSeconds)
            .ToList();

        var folded = own.Select(word => Fold(word.Text)).ToList();
        var foldable = folded.Count(word => word.Length > 0);

        if (nearby.Count == 0 || foldable == 0)
        {
            return null;
        }

        var starts = new double?[own.Count];
        var matched = 0;

        foreach (var (mine, theirs) in Match(folded, nearby.Select(word => word.Folded).ToList()))
        {
            starts[mine] = nearby[theirs].StartSeconds;
            matched++;
        }

        if (matched < foldable * LeastMatchedShare)
        {
            return null;
        }

        Interpolate(starts, segment.StartSeconds, segment.EndSeconds);

        var words = new List<WordTimings.Word>(own.Count);

        for (var i = 0; i < own.Count; i++)
        {
            var start = starts[i]!.Value;

            // Each word runs to where the next begins — DTW gives one moment per word, not a
            // span — capped, so a pause is not spent lit on the word before it.
            var next = i + 1 < own.Count ? starts[i + 1]!.Value : Math.Max(segment.EndSeconds, start);
            var end = Math.Min(next, start + WordTimings.LongestWordSeconds);

            words.Add(new WordTimings.Word(own[i].Text, start, Math.Max(end, start)) { Offset = own[i].Offset });
        }

        return words;
    }

    /// <summary>
    /// Fills the words no heard word answered for, evenly between the found ones on either
    /// side, and keeps the whole run from ever stepping backwards in time.
    /// </summary>
    private static void Interpolate(double?[] starts, double segmentStart, double segmentEnd)
    {
        var i = 0;

        while (i < starts.Length)
        {
            if (starts[i] is not null)
            {
                i++;
                continue;
            }

            var gapStart = i;

            while (i < starts.Length && starts[i] is null)
            {
                i++;
            }

            var before = gapStart > 0 ? starts[gapStart - 1]!.Value : Math.Min(segmentStart, starts[i] ?? segmentStart);
            var after = i < starts.Length ? starts[i]!.Value : Math.Max(segmentEnd, before);
            var steps = i - gapStart + 1;

            for (var k = gapStart; k < i; k++)
            {
                starts[k] = before + ((after - before) * (k - gapStart + 1) / steps);
            }
        }

        for (var k = 1; k < starts.Length; k++)
        {
            starts[k] = Math.Max(starts[k]!.Value, starts[k - 1]!.Value);
        }
    }

    /// <summary>Longest common subsequence of two folded word lists, as index pairs in order.</summary>
    private static List<(int Mine, int Theirs)> Match(List<string> mine, List<string> theirs)
    {
        var table = new int[mine.Count + 1, theirs.Count + 1];

        for (var i = mine.Count - 1; i >= 0; i--)
        {
            for (var j = theirs.Count - 1; j >= 0; j--)
            {
                table[i, j] = mine[i].Length > 0 && mine[i] == theirs[j]
                    ? table[i + 1, j + 1] + 1
                    : Math.Max(table[i + 1, j], table[i, j + 1]);
            }
        }

        var pairs = new List<(int, int)>();
        int x = 0, y = 0;

        while (x < mine.Count && y < theirs.Count)
        {
            if (mine[x].Length > 0 && mine[x] == theirs[y])
            {
                pairs.Add((x++, y++));
            }
            else if (table[x + 1, y] >= table[x, y + 1])
            {
                x++;
            }
            else
            {
                y++;
            }
        }

        return pairs;
    }

    private static List<(string Text, int Offset)> Split(string text)
    {
        var words = new List<(string, int)>();
        var at = 0;

        while (at < text.Length)
        {
            while (at < text.Length && char.IsWhiteSpace(text[at]))
            {
                at++;
            }

            var from = at;

            while (at < text.Length && !char.IsWhiteSpace(text[at]))
            {
                at++;
            }

            if (at > from)
            {
                words.Add((text[from..at], from));
            }
        }

        return words;
    }

    private static string Fold(string text) =>
        new([.. text.Where(char.IsLetterOrDigit).Select(char.ToLowerInvariant)]);
}
