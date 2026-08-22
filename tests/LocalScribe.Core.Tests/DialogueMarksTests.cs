using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class DialogueMarksTests
{
    private static TranscriptSegment Segment(string text, double start = 0, double end = 10) =>
        new(text, start, end);

    /// <summary>The case from the debate recording that nothing could divide before.</summary>
    [Fact]
    public void TwoSpeakersInOneSegmentBecomeTwoSegments()
    {
        var result = DialogueMarks.Split([Segment("- Because, again-- - I asked you", 0, 4)]);

        Assert.Equal(2, result.Count);
        Assert.Equal("Because, again--", result[0].Text);
        Assert.Equal("I asked you", result[1].Text);
    }

    /// <summary>
    /// The halves have to land in different turns or the split achieves nothing, so the time is
    /// shared out rather than both halves claiming the whole segment.
    /// </summary>
    [Fact]
    public void TheTimeIsSharedOutBetweenThem()
    {
        var result = DialogueMarks.Split([Segment("- Yes I agree with that - No you don't", 10, 20)]);

        Assert.Equal(2, result.Count);
        Assert.Equal(10, result[0].StartSeconds, 2);
        Assert.Equal(result[0].EndSeconds, result[1].StartSeconds, 2);
        Assert.Equal(20, result[1].EndSeconds, 2);
        Assert.True(result[1].StartSeconds > 10, "the second speaker must start later than the first");
    }

    [Fact]
    public void ALeadingMarkIsRemovedWithoutSplitting()
    {
        var result = DialogueMarks.Split([Segment("- In what way?")]);

        Assert.Single(result);
        Assert.Equal("In what way?", result[0].Text);
    }

    /// <summary>
    /// The narrowness is the point. A hyphenated word is not two people, and mangling ordinary
    /// text to catch a speaker change would be a poor trade.
    /// </summary>
    [Theory]
    [InlineData("It was a well-known problem")]
    [InlineData("The score was three-nil at half-time")]
    [InlineData("A state-of-the-art co-operative agreement")]
    [InlineData("She said it was over--")]
    public void OrdinaryHyphensAreLeftAlone(string text)
    {
        var result = DialogueMarks.Split([Segment(text)]);

        Assert.Single(result);
        Assert.Equal(text, result[0].Text);
    }

    [Fact]
    public void ThreeSpeakersInOneSegmentBecomeThree()
    {
        var result = DialogueMarks.Split([Segment("- Yes - No - Perhaps", 0, 9)]);

        Assert.Equal(3, result.Count);
        Assert.Equal(["Yes", "No", "Perhaps"], result.Select(s => s.Text));
    }

    [Fact]
    public void EverythingElseAboutASegmentSurvives()
    {
        IReadOnlyList<TranscriptSegment> segments =
        [
            new("- Yes indeed - No it isn't", 4, 8, AverageLogProbability: -0.3, NoSpeechProbability: 0.02, Speaker: "Kim"),
        ];

        var result = DialogueMarks.Split(segments);

        Assert.All(result, s => Assert.Equal("Kim", s.Speaker));
        Assert.All(result, s => Assert.Equal(-0.3, s.AverageLogProbability, 3));
        Assert.All(result, s => Assert.Equal(0.02, s.NoSpeechProbability, 3));
    }

    [Fact]
    public void SegmentsWithNoMarksAreUntouched()
    {
        IReadOnlyList<TranscriptSegment> segments =
        [
            Segment("Atheists reject God, but they don't understand what they're rejecting.", 0, 5),
            Segment("It's directly relevant.", 5, 7),
        ];

        var result = DialogueMarks.Split(segments);

        Assert.Equal(segments, result);
    }

    [Fact]
    public void NothingAtAllIsNotAFailure() => Assert.Empty(DialogueMarks.Split([]));

    [Fact]
    public void ASegmentOfOnlyAMarkIsNotLost()
    {
        var result = DialogueMarks.Split([Segment("-")]);

        Assert.Single(result);
    }
}
