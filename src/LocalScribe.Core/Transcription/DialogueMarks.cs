namespace LocalScribe.Core.Transcription;

/// <summary>
/// Splits segments where Whisper marked a change of speaker.
/// <para>
/// Whisper writes dialogue the way a script does: a dash at the start of a line for a new
/// speaker, and a dash mid-line when someone cuts in. It does this unprompted, from the same
/// audio the diarizer is working on, and it was being thrown away — the marks survived into the
/// transcript as stray punctuation and nothing acted on them.
/// </para>
/// <para>
/// They are worth having because they come from a different model looking at a different thing.
/// The diarizer knows voices and timing; the transcriber knows words, and hears a speaker change
/// in the shape of the sentence. On a debate recording it marked eight changes in twenty-six
/// segments, including one where two people share a single segment — "Because, again-- - I asked
/// you" — which no amount of voice comparison would have divided, because the segment was never
/// offered for division.
/// </para>
/// <para>
/// A split here is not an attribution. It only ensures that when two people share a segment,
/// there are two segments for the speakers to be attached to, each landing in its own turn.
/// </para>
/// </summary>
public static class DialogueMarks
{
    /// <summary>
    /// Splits every segment at the dialogue marks in it, apportioning the time between the
    /// halves, and removes the marks.
    /// </summary>
    public static IReadOnlyList<TranscriptSegment> Split(IReadOnlyList<TranscriptSegment> segments)
    {
        ArgumentNullException.ThrowIfNull(segments);

        var result = new List<TranscriptSegment>(segments.Count);

        foreach (var segment in segments)
        {
            var parts = Parts(segment.Text);

            if (parts.Count <= 1)
            {
                result.Add(parts.Count == 1 && parts[0] != segment.Text
                    ? segment with { Text = parts[0] }
                    : segment);

                continue;
            }

            // Time is shared out by how much was said, which is the same rule used when a
            // segment straddles two turns. Nothing better is available: the marks say where the
            // speaker changed, not when.
            var total = parts.Sum(part => part.Length);
            var at = segment.StartSeconds;

            for (var i = 0; i < parts.Count; i++)
            {
                var share = (segment.EndSeconds - segment.StartSeconds) * parts[i].Length / total;
                var end = i == parts.Count - 1 ? segment.EndSeconds : at + share;

                result.Add(segment with { Text = parts[i], StartSeconds = at, EndSeconds = end });
                at = end;
            }
        }

        return result;
    }

    /// <summary>
    /// The pieces of one segment's text, marks removed. One piece means there was nothing to
    /// split.
    /// </summary>
    internal static List<string> Parts(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var parts = new List<string>();
        var start = 0;

        for (var i = 0; i < text.Length; i++)
        {
            if (!IsMarkAt(text, i))
            {
                continue;
            }

            var piece = text[start..i].Trim();
            if (piece.Length > 0)
            {
                parts.Add(piece);
            }

            // Past the dash and any run of them: an interruption is written "again-- - I asked",
            // where the pair belongs to the sentence before and the single one is the change.
            start = i;
            while (start < text.Length && (text[start] == '-' || text[start] == ' '))
            {
                start++;
            }

            i = start - 1;
        }

        var last = text[start..].Trim();
        if (last.Length > 0)
        {
            parts.Add(last);
        }

        return parts.Count == 0 ? [text.Trim()] : parts;
    }

    /// <summary>
    /// True where a dash is script punctuation rather than part of a word.
    /// <para>
    /// The test is deliberately narrow. A hyphen inside a word, a minus sign, and a pair of
    /// dashes marking an interruption are all ordinary text; only a dash that opens the line, or
    /// stands alone between spaces, means somebody else started talking.
    /// </para>
    /// </summary>
    private static bool IsMarkAt(string text, int i)
    {
        if (text[i] != '-')
        {
            return false;
        }

        // Opening the segment.
        if (i == 0)
        {
            return i + 1 < text.Length && text[i + 1] == ' ';
        }

        // Standing alone, with a space either side. The dash run is stepped over afterwards, so
        // "again-- - I" matches only at the third dash, where the space before it is.
        return text[i - 1] == ' '
            && i + 1 < text.Length
            && text[i + 1] == ' ';
    }
}
