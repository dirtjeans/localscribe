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
    public ParagraphView(TranscriptParagraph paragraph)
    {
        ArgumentNullException.ThrowIfNull(paragraph);

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
}
