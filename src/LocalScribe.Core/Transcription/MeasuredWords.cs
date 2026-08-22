namespace LocalScribe.Core.Transcription;

/// <summary>
/// Pairs a segment's words with times measured by the aligner.
/// <para>
/// The aligner works one segment at a time and returns a flat run of words for the whole
/// recording, so getting them back out means asking which of them were said during a given
/// segment. That works only while the pool holds exactly one alignment. It once held two — the
/// pipeline aligned the raw transcript in parallel with cleanup and then aligned the finished
/// transcript again afterwards, appending both — and two alignments of one recording cover the
/// same seconds, so a segment found about twice the words it had text for. Where the counts
/// happened to agree anyway the words were paired against an interleaving of the two, which put
/// the marker sometimes ahead of the sound and sometimes behind it, with no pattern to it.
/// </para>
/// <para>
/// Kept here rather than beside the caller so that the rule can be shown to hold.
/// </para>
/// </summary>
public static class MeasuredWords
{
    /// <summary>Slack around a segment boundary, in seconds.</summary>
    public const double Tolerance = 0.005;

    /// <summary>
    /// The words of <paramref name="segment"/> carrying measured times, or null when the pool
    /// cannot supply them.
    /// </summary>
    /// <param name="segment">The segment as the reader sees it, which is the text that wins.</param>
    /// <param name="pool">Every word the aligner placed, for the whole recording.</param>
    /// <param name="own">The segment's own words, already split out.</param>
    /// <returns>
    /// The segment's own words with the pool's times, or null when the two disagree about how
    /// many words there are. Returning null is how the caller knows to keep the estimate: a
    /// segment timed against the wrong words is worse than one timed roughly.
    /// </returns>
    public static IReadOnlyList<WordTimings.Word>? Pair(
        TranscriptSegment segment,
        IReadOnlyList<WordTimings.Word> pool,
        IReadOnlyList<WordTimings.Word> own)
    {
        ArgumentNullException.ThrowIfNull(segment);
        ArgumentNullException.ThrowIfNull(pool);
        ArgumentNullException.ThrowIfNull(own);

        if (own.Count == 0)
        {
            return null;
        }

        // A word belongs to the segment its start falls inside. Both bounds carry the same
        // tolerance so the windows stay a clean partition: a word starting on a boundary belongs
        // to the segment beginning there, never to both and never to neither.
        var measured = pool
            .Where(w => w.StartSeconds >= segment.StartSeconds - Tolerance
                && w.StartSeconds < segment.EndSeconds - Tolerance)
            .OrderBy(w => w.StartSeconds)
            .ToList();

        if (measured.Count != own.Count)
        {
            return null;
        }

        // The text is the segment's, the times are the pool's. Cleanup rewrites punctuation after
        // alignment has run and it is the cleaned text that should be read; only the times are
        // borrowed.
        return [.. own.Select((word, i) => new WordTimings.Word(
            word.Text, measured[i].StartSeconds, measured[i].EndSeconds) { Offset = word.Offset })];
    }
}
