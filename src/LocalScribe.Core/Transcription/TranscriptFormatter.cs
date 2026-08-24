using System.Text;

namespace LocalScribe.Core.Transcription;

/// <summary>A run of segments that belong together, with the span they cover.</summary>
/// <param name="Segments">The segments, in order.</param>
/// <param name="Speaker">Who was talking, when that is known.</param>
public sealed record TranscriptParagraph(IReadOnlyList<TranscriptSegment> Segments, string? Speaker = null)
{
    /// <summary>True when most of this paragraph was spoken over somebody else.</summary>
    public bool Overlapped => Segments.Count > 0 && Segments.All(segment => segment.Overlapped);

    /// <summary>The paragraph as flowing text.</summary>
    public string Text => string.Join(" ", Segments.Select(s => s.Text.Trim()).Where(t => t.Length > 0));

    /// <summary>Offset of the first word.</summary>
    public double StartSeconds => Segments.Count == 0 ? 0 : Segments[0].StartSeconds;

    /// <summary>Offset of the last word.</summary>
    public double EndSeconds => Segments.Count == 0 ? 0 : Segments[^1].EndSeconds;
}

/// <summary>
/// Groups a transcript into paragraphs, and writes it out.
/// <para>
/// Whisper returns a sequence of short spans and nothing else. Joined end to end they make one
/// unbroken wall of text, which is accurate and nearly unreadable — long enough to be worth
/// transcribing is long enough to need structure.
/// </para>
/// <para>
/// The break is taken on silence rather than on length or sentence count. A speaker pausing is
/// the only signal in the data that actually corresponds to a change of thought, and it is the
/// same signal a person would use listening to the recording. Sentence endings refine where the
/// break lands; they do not decide that one happens.
/// </para>
/// </summary>
public static class TranscriptFormatter
{
    /// <summary>
    /// Silence long enough to read as a new thought. Ordinary between-sentence pauses run to a
    /// few hundred milliseconds; anything past this is someone stopping rather than breathing.
    /// </summary>
    public const double DefaultParagraphPauseSeconds = 1.2;

    /// <summary>
    /// Length past which a paragraph is broken at the next sentence end regardless of pauses.
    /// A speaker in full flow can run for minutes without a gap, and a page-long paragraph is
    /// the problem this is here to solve.
    /// </summary>
    public const int DefaultMaxParagraphCharacters = 700;

    /// <summary>Groups segments into paragraphs.</summary>
    /// <param name="segments">Time-ordered segments.</param>
    /// <param name="pauseSeconds">Silence that starts a new paragraph.</param>
    /// <param name="maxCharacters">Length past which a sentence end will break the paragraph.</param>
    public static IReadOnlyList<TranscriptParagraph> Paragraphs(
        IReadOnlyList<TranscriptSegment> segments,
        double pauseSeconds = DefaultParagraphPauseSeconds,
        int maxCharacters = DefaultMaxParagraphCharacters)
    {
        ArgumentNullException.ThrowIfNull(segments);

        var paragraphs = new List<TranscriptParagraph>();
        var current = new List<TranscriptSegment>();
        var length = 0;

        foreach (var segment in segments)
        {
            if (string.IsNullOrWhiteSpace(segment.Text))
            {
                continue;
            }

            if (current.Count > 0 && StartsNewParagraph(current[^1], segment, pauseSeconds, length, maxCharacters))
            {
                paragraphs.Add(new TranscriptParagraph(current, current[0].Speaker));
                current = [];
                length = 0;
            }

            current.Add(segment);
            length += segment.Text.Trim().Length + 1;
        }

        if (current.Count > 0)
        {
            paragraphs.Add(new TranscriptParagraph(current, current[0].Speaker));
        }

        return paragraphs;
    }

    private static bool StartsNewParagraph(
        TranscriptSegment previous,
        TranscriptSegment next,
        double pauseSeconds,
        int lengthSoFar,
        int maxCharacters)
    {
        // A different speaker always starts a paragraph. Running two people together is worse
        // than any amount of bad spacing.
        if (previous.Speaker != next.Speaker)
        {
            return true;
        }

        // A pause, but only between sentences. Breaking mid-sentence puts one sentence under two
        // headings and reads as the transcript having lost something — which is how it was
        // reported: "Because Metabase plugs into every" ended one paragraph and "database a
        // company connects to it" began the next, both the same speaker.
        //
        // The gap itself is often not real. Segments are placed by measuring where their words
        // are, so two that run straight on can still end up a second apart on the clock, and a
        // rule that trusts the clock over the grammar will split them.
        if (next.StartSeconds - previous.EndSeconds >= pauseSeconds && EndsSentence(previous.Text))
        {
            return true;
        }

        // Only break a long paragraph where a sentence actually ended, so the split reads as a
        // paragraph rather than as a line that ran out of room.
        return lengthSoFar >= maxCharacters && EndsSentence(previous.Text);
    }

    private static bool EndsSentence(string text)
    {
        var trimmed = text.TrimEnd();

        return trimmed.Length > 0 && trimmed[^1] is '.' or '?' or '!';
    }

    /// <summary>Plain text, one paragraph per blank-line-separated block.</summary>
    public static string ToPlainText(IReadOnlyList<TranscriptParagraph> paragraphs)
    {
        ArgumentNullException.ThrowIfNull(paragraphs);

        var builder = new StringBuilder();

        foreach (var paragraph in paragraphs)
        {
            if (builder.Length > 0)
            {
                builder.AppendLine().AppendLine();
            }

            if (paragraph.Speaker is { Length: > 0 } speaker)
            {
                builder.Append(speaker).Append(": ");
            }

            builder.Append(paragraph.Text);
        }

        return builder.ToString();
    }

    /// <summary>
    /// Markdown with a timestamp heading per paragraph, for pasting somewhere that keeps
    /// formatting.
    /// </summary>
    public static string ToMarkdown(IReadOnlyList<TranscriptParagraph> paragraphs, string? title = null)
    {
        ArgumentNullException.ThrowIfNull(paragraphs);

        var builder = new StringBuilder();

        if (title is { Length: > 0 })
        {
            builder.Append("# ").AppendLine(title).AppendLine();
        }

        foreach (var paragraph in paragraphs)
        {
            var speaker = paragraph.Speaker is { Length: > 0 } name ? $"**{name}** " : string.Empty;

            builder
                .Append("**[")
                .Append(Clock(paragraph.StartSeconds))
                .Append("]** ")
                .Append(speaker)
                .AppendLine(paragraph.Text)
                .AppendLine();
        }

        return builder.ToString();
    }

    /// <summary>
    /// SubRip subtitles, one cue per segment rather than per paragraph: a cue has to stay on
    /// screen only as long as the words it shows.
    /// </summary>
    public static string ToSubRip(IReadOnlyList<TranscriptSegment> segments)
    {
        ArgumentNullException.ThrowIfNull(segments);

        var builder = new StringBuilder();
        var index = 1;

        foreach (var segment in segments.Where(s => !string.IsNullOrWhiteSpace(s.Text)))
        {
            var speaker = segment.Speaker is { Length: > 0 } name ? $"{name}: " : string.Empty;

            builder
                .Append(index++).AppendLine()
                .Append(SubRipTime(segment.StartSeconds))
                .Append(" --> ")
                .AppendLine(SubRipTime(segment.EndSeconds))
                .Append(speaker)
                .AppendLine(segment.Text.Trim())
                .AppendLine();
        }

        return builder.ToString();
    }

    /// <summary>Wall-clock position, e.g. 1:04 or 1:02:03.</summary>
    public static string Clock(double seconds)
    {
        var span = TimeSpan.FromSeconds(Math.Max(0, seconds));

        return span.TotalHours >= 1
            ? $"{(int)span.TotalHours}:{span.Minutes:00}:{span.Seconds:00}"
            : $"{span.Minutes}:{span.Seconds:00}";
    }

    private static string SubRipTime(double seconds)
    {
        var span = TimeSpan.FromSeconds(Math.Max(0, seconds));

        return $"{(int)span.TotalHours:00}:{span.Minutes:00}:{span.Seconds:00},{span.Milliseconds:000}";
    }
}
