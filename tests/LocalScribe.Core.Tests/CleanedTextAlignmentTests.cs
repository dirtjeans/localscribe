using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class CleanedTextAlignmentTests
{
    private static TranscriptSegment Segment(string text, double start, double end) =>
        new(text, start, end);

    [Fact]
    public void PunctuationLandsInTheRightSegments()
    {
        IReadOnlyList<TranscriptSegment> segments =
        [
            Segment("okay i am going to test this", 0, 3),
            Segment("lets see how well it works", 3, 6),
        ];

        var result = CleanedTextAlignment.Apply(
            segments,
            "Okay, I am going to test this. Let's see how well it works.");

        Assert.Equal("Okay, I am going to test this.", result[0].Text);
        Assert.Equal("Let's see how well it works.", result[1].Text);
    }

    /// <summary>
    /// The whole point of the exercise. A cleaned transcript that has lost its timings cannot be
    /// clicked, cannot be highlighted against the waveform, and cannot carry a speaker label.
    /// </summary>
    [Fact]
    public void TimingsSurvive()
    {
        IReadOnlyList<TranscriptSegment> segments =
        [
            Segment("first bit", 1.5, 4.25),
            Segment("second bit", 4.25, 9.0),
        ];

        var result = CleanedTextAlignment.Apply(segments, "First bit. Second bit.");

        Assert.Equal(1.5, result[0].StartSeconds);
        Assert.Equal(4.25, result[0].EndSeconds);
        Assert.Equal(4.25, result[1].StartSeconds);
        Assert.Equal(9.0, result[1].EndSeconds);
    }

    /// <summary>Speaker labels and confidences ride along untouched.</summary>
    [Fact]
    public void EverythingButTheTextIsPreserved()
    {
        IReadOnlyList<TranscriptSegment> segments =
        [
            new("hello there", 0, 2, AverageLogProbability: -0.25, NoSpeechProbability: 0.01, Speaker: "Kim"),
        ];

        var result = CleanedTextAlignment.Apply(segments, "Hello there.");

        Assert.Equal("Kim", result[0].Speaker);
        Assert.Equal(-0.25, result[0].AverageLogProbability);
        Assert.Equal(0.01, result[0].NoSpeechProbability);
    }

    /// <summary>Removing filler shifts every later word one place; the split must still land.</summary>
    [Fact]
    public void DroppedFillerDoesNotShiftTheBoundary()
    {
        IReadOnlyList<TranscriptSegment> segments =
        [
            Segment("so um i think we should", 0, 3),
            Segment("uh ship it on friday", 3, 6),
        ];

        var result = CleanedTextAlignment.Apply(
            segments,
            "So I think we should ship it on Friday.");

        Assert.Equal("So I think we should", result[0].Text);
        Assert.Equal("ship it on Friday.", result[1].Text);
    }

    /// <summary>A glossary correction is a word that matches nothing on either side of it.</summary>
    [Fact]
    public void ACorrectedWordStaysWithItsNeighbours()
    {
        IReadOnlyList<TranscriptSegment> segments =
        [
            Segment("we deployed to cuber netties", 0, 3),
            Segment("on tuesday morning", 3, 6),
        ];

        var result = CleanedTextAlignment.Apply(
            segments,
            "We deployed to Kubernetes on Tuesday morning.");

        Assert.Contains("Kubernetes", result[0].Text);
        Assert.Equal("on Tuesday morning.", result[1].Text);
    }

    /// <summary>
    /// A segment the model emptied keeps what it had. An empty line cannot be clicked and would
    /// vanish from the transcript, which is a worse outcome than one unpunctuated line.
    /// </summary>
    [Fact]
    public void ASegmentLeftWithNothingKeepsItsWords()
    {
        IReadOnlyList<TranscriptSegment> segments =
        [
            Segment("um", 0, 1),
            Segment("right lets begin", 1, 4),
        ];

        var result = CleanedTextAlignment.Apply(segments, "Right, let's begin.");

        Assert.Equal("um", result[0].Text);
        Assert.Equal("Right, let's begin.", result[1].Text);
    }

    [Fact]
    public void OneSegmentTakesTheWholeReply()
    {
        IReadOnlyList<TranscriptSegment> segments = [Segment("hello there how are you", 0, 3)];

        var result = CleanedTextAlignment.Apply(segments, "Hello there, how are you?");

        Assert.Single(result);
        Assert.Equal("Hello there, how are you?", result[0].Text);
    }

    [Fact]
    public void AnEmptyReplyChangesNothing()
    {
        IReadOnlyList<TranscriptSegment> segments = [Segment("something was said", 0, 3)];

        var result = CleanedTextAlignment.Apply(segments, "   ");

        Assert.Equal("something was said", result[0].Text);
    }

    [Fact]
    public void NoSegmentsIsNotAFailure() =>
        Assert.Empty(CleanedTextAlignment.Apply([], "anything at all"));

    /// <summary>
    /// Words must not be duplicated across the boundary. An earlier sketch assigned every
    /// unmatched word to the segment in hand, which quietly repeated the tail of one segment at
    /// the head of the next.
    /// </summary>
    [Fact]
    public void NoWordIsWrittenTwice()
    {
        IReadOnlyList<TranscriptSegment> segments =
        [
            Segment("one two three", 0, 3),
            Segment("four five six", 3, 6),
        ];

        var result = CleanedTextAlignment.Apply(segments, "One, two, three. Four, five, six.");

        var all = string.Join(" ", result.Select(segment => segment.Text));
        Assert.Equal("One, two, three. Four, five, six.", all);
    }
}
