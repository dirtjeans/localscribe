namespace LocalScribe.Core.Transcription;

/// <summary>
/// Puts cleaned-up text back into the segments it came from, keeping every timing.
/// <para>
/// The missing half of the cleanup stage. The refiner takes a window of transcript, sends it to
/// a language model, and gets back one flat string — punctuated, capitalised, and with no idea
/// which word was spoken when. The transcript view needs the opposite: segments with start and
/// end times, because that is what click-to-play seeks with, what the waveform highlights, and
/// what speaker labels attach to.
/// </para>
/// <para>
/// Without a way across that gap the cleaned text simply had nowhere to go, so it was computed
/// and dropped, and every transcript displayed, saved, and copied was the raw one. This is the
/// way across.
/// </para>
/// </summary>
public static class CleanedTextAlignment
{
    /// <summary>
    /// Rewrites <paramref name="segments"/> with the words of <paramref name="cleaned"/>,
    /// leaving each segment's start and end untouched.
    /// </summary>
    /// <param name="segments">The window as transcribed.</param>
    /// <param name="cleaned">The same window as the model returned it.</param>
    public static IReadOnlyList<TranscriptSegment> Apply(
        IReadOnlyList<TranscriptSegment> segments,
        string cleaned) =>
        TryApply(segments, cleaned) ?? segments;

    /// <summary>
    /// Like <see cref="Apply"/>, but null when the cleaned text cannot be walked back onto the
    /// segments without writing some of it twice — so the caller can count the window as
    /// failed and keep it raw, instead of shipping a transcript with a window in it twice.
    /// </summary>
    public static IReadOnlyList<TranscriptSegment>? TryApply(
        IReadOnlyList<TranscriptSegment> segments,
        string cleaned)
    {
        ArgumentNullException.ThrowIfNull(segments);
        ArgumentNullException.ThrowIfNull(cleaned);

        if (segments.Count == 0)
        {
            return segments;
        }

        var words = Split(cleaned);
        if (words.Length == 0)
        {
            return segments;
        }

        // One segment is the common case and needs none of the machinery below.
        if (segments.Count == 1)
        {
            return [segments[0] with { Text = string.Join(" ", words) }];
        }

        var spoken = new List<(string Word, int Segment)>();
        for (var s = 0; s < segments.Count; s++)
        {
            foreach (var word in Split(segments[s].Text))
            {
                spoken.Add((Normalise(word), s));
            }
        }

        var assigned = new List<string>[segments.Count];
        for (var s = 0; s < segments.Count; s++)
        {
            assigned[s] = [];
        }

        // Walked in step. Cleanup rewrites punctuation and casing but leaves the words in order,
        // so matching them back is a matter of keeping a place in the original and allowing for
        // the two things cleanup legitimately does: drop a filler word, and change one.
        var at = 0;
        var current = 0;

        for (var i = 0; i < words.Length; i++)
        {
            var word = words[i];
            var normalised = Normalise(word);
            var found = -1;

            for (var ahead = 0; ahead < Lookahead && at + ahead < spoken.Count; ahead++)
            {
                if (spoken[at + ahead].Word == normalised)
                {
                    found = at + ahead;
                    break;
                }
            }

            // Lost the place: the model dropped or reworded more than the lookahead covers.
            // Without recovery the walk never moves again and every remaining word lands in
            // the segment it is stuck in. A run of consecutive matches further on is strong
            // enough evidence to jump to — one matching word alone could be a "the" from the
            // middle of the next sentence.
            if (found < 0)
            {
                found = Resynchronise(spoken, at, words, i);
            }

            if (found >= 0)
            {
                current = spoken[found].Segment;
                at = found + 1;
            }

            // A word that matches nothing was introduced by the cleanup — a contraction split in
            // two, a term corrected from the glossary. It belongs where the words either side of
            // it went, which is the segment we are standing in.
            assigned[current].Add(word);
        }

        var rewritten = new TranscriptSegment[segments.Count];
        var keptRaw = 0;

        for (var s = 0; s < segments.Count; s++)
        {
            // A segment that came out empty keeps what it had. Losing its words would lose a
            // seekable line from the transcript, which is worse than leaving one unpunctuated.
            if (assigned[s].Count == 0)
            {
                rewritten[s] = segments[s];
                keptRaw += Split(segments[s].Text).Length;
            }
            else
            {
                rewritten[s] = segments[s] with { Text = string.Join(" ", assigned[s]) };
            }
        }

        // Kept-raw words sit beside cleaned words covering the same speech. A stray "um uh"
        // emptied by cleanup is the legitimate case and costs a couple of words; whole
        // sentences kept raw mean the walk lost its place and the window is in the transcript
        // twice — which no amount of punctuation is worth.
        return keptRaw > Math.Max(KeptRawAllowance, spoken.Count * KeptRawShare) ? null : rewritten;
    }

    /// <summary>How far past its place the walk will search for a run to resume from.</summary>
    private const int ResyncReach = 64;

    /// <summary>Consecutive matching words needed to trust a jump.</summary>
    private const int ResyncRun = 3;

    /// <summary>Raw words a mapping may keep beside cleaned text: a filler segment's worth.</summary>
    private const int KeptRawAllowance = 4;

    /// <summary>...or this share of the window, on a long one.</summary>
    private const double KeptRawShare = 0.15;

    private static int Resynchronise(
        List<(string Word, int Segment)> spoken, int at, string[] words, int from)
    {
        var run = Math.Min(ResyncRun, words.Length - from);

        for (var p = at; p < Math.Min(spoken.Count - run + 1, at + ResyncReach); p++)
        {
            var matches = true;

            for (var k = 0; k < run && matches; k++)
            {
                matches = spoken[p + k].Word == Normalise(words[from + k]);
            }

            if (matches)
            {
                return p;
            }
        }

        return -1;
    }

    /// <summary>
    /// How far ahead to look for a word before treating it as newly introduced. Enough to step
    /// over a filler or two; short enough that a genuinely new word does not match something
    /// from the middle of the next sentence.
    /// </summary>
    private const int Lookahead = 4;

    private static string[] Split(string text) =>
        text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

    /// <summary>Reduced to what both readings of a word will agree on.</summary>
    private static string Normalise(string word) =>
        new(word.Where(char.IsLetterOrDigit).Select(char.ToLowerInvariant).ToArray());
}
