namespace LocalScribe.Core.Transcription;

/// <summary>
/// Pairs a segment's words with the times measured for it.
/// <para>
/// The text is the segment's and only the times are borrowed. Cleanup rewrites punctuation after
/// alignment has run and it is the cleaned text that should be read.
/// </para>
/// <para>
/// Words used to be fetched out of one pool covering the whole recording, by asking which of them
/// were said during a segment. That worked only while every word sat inside the segment it came
/// from, and it stopped being true: the aligner now searches outside a segment's stated bounds,
/// because those bounds are the least trustworthy thing about it. Words are kept with the segment
/// they were measured for instead, which is both simpler and immune to a question the pool could
/// answer wrongly.
/// </para>
/// </summary>
public static class MeasuredWords
{
    /// <summary>
    /// The segment's own words carrying measured times, or null when the two do not correspond.
    /// </summary>
    /// <param name="measured">What the aligner placed for this segment, in order.</param>
    /// <param name="own">The segment's own words, already split out.</param>
    /// <returns>
    /// The words with their measured times, or null when the two disagree about how many words
    /// there are. Returning null is how the caller knows to keep the estimate: a segment timed
    /// against the wrong words is worse than one timed roughly.
    /// </returns>
    public static IReadOnlyList<WordTimings.Word>? Pair(
        IReadOnlyList<WordTimings.Word> measured,
        IReadOnlyList<WordTimings.Word> own)
    {
        ArgumentNullException.ThrowIfNull(measured);
        ArgumentNullException.ThrowIfNull(own);

        if (own.Count == 0 || measured.Count != own.Count)
        {
            return null;
        }

        return [.. own.Select((word, i) => new WordTimings.Word(
            word.Text, measured[i].StartSeconds, measured[i].EndSeconds) { Offset = word.Offset })];
    }
}
