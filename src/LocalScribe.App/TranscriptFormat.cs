namespace LocalScribe.App;

/// <summary>The shapes a transcript can be saved or copied in.</summary>
public enum TranscriptFormat
{
    /// <summary>Paragraphs separated by blank lines. What the window shows.</summary>
    PlainText,

    /// <summary>Paragraphs with a timestamp each, for pasting somewhere that keeps formatting.</summary>
    Markdown,

    /// <summary>SubRip subtitles, for putting the words back on the video they came from.</summary>
    SubRip,
}
