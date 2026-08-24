using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class ParagraphBreakTests
{
    private static TranscriptSegment Said(string speaker, string text, double from, double to) =>
        new(text, from, to) { Speaker = speaker };

    /// <summary>
    /// One sentence under two headings reads as the transcript having lost something. The gap is
    /// often not real anyway: segments are placed by measuring where their words are, so two that
    /// run straight on can still land a second apart on the clock.
    /// </summary>
    [Fact]
    public void APauseMidSentenceDoesNotStartAParagraph()
    {
        var paragraphs = TranscriptFormatter.Paragraphs(
        [
            Said("Mindy", "Because Metabase plugs into every", 10, 12),
            Said("Mindy", "database a company connects to it, one crack server can expose them.", 14, 18),
        ]);

        Assert.Single(paragraphs);
    }

    /// <summary>Between sentences it still does, which is what the rule is for.</summary>
    [Fact]
    public void APauseBetweenSentencesStillStartsOne()
    {
        var paragraphs = TranscriptFormatter.Paragraphs(
        [
            Said("Mindy", "That is the whole story.", 10, 12),
            Said("Mindy", "And in other news, a flaw in Metabase.", 14, 18),
        ]);

        Assert.Equal(2, paragraphs.Count);
    }

    /// <summary>A different speaker always does, pause or not, sentence or not.</summary>
    [Fact]
    public void ADifferentSpeakerAlwaysStartsOne()
    {
        var paragraphs = TranscriptFormatter.Paragraphs(
        [
            Said("Mindy", "and that is why we", 10, 12),
            Said("Ken", "think it matters.", 12.1, 14),
        ]);

        Assert.Equal(2, paragraphs.Count);
    }

    /// <summary>No pause, no break, whatever the grammar.</summary>
    [Fact]
    public void RunningStraightOnStaysOneParagraph()
    {
        var paragraphs = TranscriptFormatter.Paragraphs(
        [
            Said("Ken", "Zero trust has become a mandate. That's why", 10, 12),
            Said("Ken", "the Cloud Security Alliance created a certification.", 12.1, 16),
        ]);

        Assert.Single(paragraphs);
    }
}
