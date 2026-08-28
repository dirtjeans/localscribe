using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class TranscriptFormatterTests
{
    private static TranscriptSegment Segment(string text, double start, double end, string? speaker = null) =>
        new(text, start, end, Speaker: speaker);

    [Fact]
    public void SpeechWithoutPausesStaysOneParagraph()
    {
        var paragraphs = TranscriptFormatter.Paragraphs(
        [
            Segment("One thing.", 0, 2),
            Segment("Then another.", 2.1, 4),
            Segment("Then a third.", 4.2, 6),
        ]);

        Assert.Single(paragraphs);
        Assert.Equal("One thing. Then another. Then a third.", paragraphs[0].Text);
    }

    /// <summary>
    /// The crosstalk badge shows only on a paragraph that is wholly crosstalk, so a contested
    /// segment must not be merged into clean speech: either the warning would vanish or it
    /// would stretch over speech that was perfectly clear.
    /// </summary>
    [Fact]
    public void CrosstalkKeepsToItsOwnParagraph()
    {
        var paragraphs = TranscriptFormatter.Paragraphs(
        [
            Segment("A clear sentence.", 0, 2, "A"),
            Segment("Said over somebody.", 2.1, 4, "A") with { Overlapped = true },
            Segment("Clear again.", 4.2, 6, "A"),
        ]);

        Assert.Equal(3, paragraphs.Count);
        Assert.False(paragraphs[0].Overlapped);
        Assert.True(paragraphs[1].Overlapped);
        Assert.False(paragraphs[2].Overlapped);
    }

    /// <summary>
    /// Silence is the only signal in the data that corresponds to a change of thought, which is
    /// why the break is taken on it rather than on sentence count.
    /// </summary>
    [Fact]
    public void ALongPauseStartsANewParagraph()
    {
        var paragraphs = TranscriptFormatter.Paragraphs(
        [
            Segment("So that is the background.", 0, 3),
            Segment("Now, the proposal.", 6, 9),
        ]);

        Assert.Equal(2, paragraphs.Count);
        Assert.Equal("So that is the background.", paragraphs[0].Text);
        Assert.Equal("Now, the proposal.", paragraphs[1].Text);
    }

    [Fact]
    public void ParagraphsCarryTheSpanTheyCover()
    {
        var paragraphs = TranscriptFormatter.Paragraphs(
        [
            Segment("First.", 1.5, 3),
            Segment("Second.", 3.1, 7.25),
        ]);

        Assert.Equal(1.5, paragraphs[0].StartSeconds);
        Assert.Equal(7.25, paragraphs[0].EndSeconds);
    }

    /// <summary>
    /// A speaker in full flow can run for minutes without a gap, and a page-long paragraph is
    /// the thing this exists to prevent.
    /// </summary>
    [Fact]
    public void AVeryLongRunIsBrokenAtASentenceEnd()
    {
        var segments = Enumerable.Range(0, 40)
            .Select(i => Segment($"Sentence number {i} goes here and runs on.", i * 2, (i * 2) + 1.9))
            .ToList();

        var paragraphs = TranscriptFormatter.Paragraphs(segments, pauseSeconds: 5.0, maxCharacters: 200);

        Assert.True(paragraphs.Count > 1);
        Assert.All(paragraphs, p => Assert.EndsWith(".", p.Text, StringComparison.Ordinal));
    }

    /// <summary>Mid-sentence is never a paragraph break, however long the run has become.</summary>
    [Fact]
    public void ALongRunWithNoSentenceEndIsNotBrokenMidSentence()
    {
        var segments = Enumerable.Range(0, 30)
            .Select(i => Segment($"and then {i}", i * 2, (i * 2) + 1.9))
            .ToList();

        var paragraphs = TranscriptFormatter.Paragraphs(segments, pauseSeconds: 5.0, maxCharacters: 50);

        Assert.Single(paragraphs);
    }

    [Fact]
    public void AChangeOfSpeakerAlwaysStartsAParagraph()
    {
        var paragraphs = TranscriptFormatter.Paragraphs(
        [
            Segment("Shall we start?", 0, 2, "Speaker 1"),
            Segment("Yes, go ahead.", 2.1, 4, "Speaker 2"),
        ]);

        Assert.Equal(2, paragraphs.Count);
        Assert.Equal("Speaker 1", paragraphs[0].Speaker);
        Assert.Equal("Speaker 2", paragraphs[1].Speaker);
    }

    [Fact]
    public void EmptySegmentsAreIgnored()
    {
        var paragraphs = TranscriptFormatter.Paragraphs(
        [
            Segment("Something.", 0, 2),
            Segment("   ", 2, 3),
            Segment("More.", 3, 4),
        ]);

        Assert.Single(paragraphs);
        Assert.Equal("Something. More.", paragraphs[0].Text);
    }

    [Fact]
    public void NothingInNothingOut() =>
        Assert.Empty(TranscriptFormatter.Paragraphs([]));

    [Fact]
    public void PlainTextSeparatesParagraphsWithABlankLine()
    {
        var text = TranscriptFormatter.ToPlainText(TranscriptFormatter.Paragraphs(
        [
            Segment("First thought.", 0, 2),
            Segment("Second thought.", 6, 8),
        ]));

        Assert.Equal($"First thought.{Environment.NewLine}{Environment.NewLine}Second thought.", text);
    }

    [Fact]
    public void PlainTextNamesTheSpeakerWhenThereIsOne()
    {
        var text = TranscriptFormatter.ToPlainText(TranscriptFormatter.Paragraphs(
            [Segment("Hello there.", 0, 2, "Speaker 1")]));

        Assert.Equal("Speaker 1: Hello there.", text);
    }

    [Fact]
    public void MarkdownCarriesATimestampPerParagraph()
    {
        var markdown = TranscriptFormatter.ToMarkdown(
            TranscriptFormatter.Paragraphs([Segment("Later on.", 125, 128)]),
            title: "Meeting");

        Assert.Contains("# Meeting", markdown, StringComparison.Ordinal);
        Assert.Contains("**[2:05]**", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void SubRipUsesOneCuePerSegment()
    {
        var srt = TranscriptFormatter.ToSubRip(
        [
            Segment("First line.", 0, 2.5),
            Segment("Second line.", 2.5, 4),
        ]);

        Assert.Contains("00:00:00,000 --> 00:00:02,500", srt, StringComparison.Ordinal);
        Assert.Contains("00:00:02,500 --> 00:00:04,000", srt, StringComparison.Ordinal);
        Assert.Contains("1\n", srt.ReplaceLineEndings("\n"), StringComparison.Ordinal);
        Assert.Contains("2\n", srt.ReplaceLineEndings("\n"), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0, "0:00")]
    [InlineData(9, "0:09")]
    [InlineData(65, "1:05")]
    [InlineData(3661, "1:01:01")]
    public void TheClockReadsAsAPosition(double seconds, string expected) =>
        Assert.Equal(expected, TranscriptFormatter.Clock(seconds));
}
