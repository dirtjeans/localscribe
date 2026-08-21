using LocalScribe.Core.Transcription;
using Microsoft.UI.Xaml;

namespace LocalScribe.App;

/// <summary>
/// One paragraph as the list shows it.
/// <para>
/// A view type rather than binding the domain record directly, because the list needs a couple
/// of things the transcript has no business knowing about: a clock string and a XAML
/// Visibility. Keeping those here is what lets <see cref="TranscriptParagraph"/> stay in Core,
/// where it is testable without a UI.
/// </para>
/// </summary>
public sealed class ParagraphView
{
    private readonly TranscriptParagraph _paragraph;

    public ParagraphView(TranscriptParagraph paragraph)
    {
        ArgumentNullException.ThrowIfNull(paragraph);

        _paragraph = paragraph;
        Text = paragraph.Text;
        Speaker = paragraph.Speaker ?? string.Empty;
        StartSeconds = paragraph.StartSeconds;
        EndSeconds = paragraph.EndSeconds;
        Clock = TranscriptFormatter.Clock(paragraph.StartSeconds);
    }

    public string Text { get; }

    public string Speaker { get; }

    /// <summary>Where clicking this paragraph starts playback.</summary>
    public double StartSeconds { get; }

    public double EndSeconds { get; }

    /// <summary>The position, for the gutter.</summary>
    public string Clock { get; }

    public Visibility HasSpeaker => Speaker.Length > 0 ? Visibility.Visible : Visibility.Collapsed;

    /// <summary>True when a playback position falls inside this paragraph.</summary>
    public bool Contains(double seconds) => seconds >= StartSeconds && seconds < EndSeconds;

    /// <summary>
    /// When in the recording a search term is actually said, as spans of time.
    /// <para>
    /// Resolved to the segment holding each match rather than to the paragraph, because a
    /// paragraph can run for half a minute and "somewhere in there" is not a useful thing to
    /// point at on a waveform. Whisper timestamps segments, not words, so a segment is as fine
    /// as the data honestly goes.
    /// </para>
    /// </summary>
    public IReadOnlyList<(double Start, double End)> MatchSpans(string query)
    {
        if (string.IsNullOrEmpty(query))
        {
            return [];
        }

        var spans = new List<(double, double)>();

        // Rebuilt the same way Text was joined, so an offset into Text means the same thing here.
        var offset = 0;

        foreach (var segment in _paragraph.Segments)
        {
            var text = segment.Text.Trim();
            if (text.Length == 0)
            {
                continue;
            }

            if (text.Contains(query, StringComparison.OrdinalIgnoreCase))
            {
                spans.Add((segment.StartSeconds, segment.EndSeconds));
            }

            offset += text.Length + 1;
        }

        // A match split across the join between two segments belongs to neither, and would
        // otherwise be missed entirely. Falling back to the whole paragraph is imprecise but
        // never wrong.
        if (spans.Count == 0 && Text.Contains(query, StringComparison.OrdinalIgnoreCase))
        {
            spans.Add((StartSeconds, EndSeconds));
        }

        return spans;
    }
}
