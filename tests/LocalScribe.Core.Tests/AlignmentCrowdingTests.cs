using LocalScribe.Core.Alignment;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class AlignmentCrowdingTests
{
    private static TranscriptSegment Said(string text, double from, double to) => new(text, from, to);

    private static TimedSegment Placed(string text, double from, double to) =>
        new(new TranscriptSegment(text, from, to), []);

    /// <summary>
    /// The two columns are the point. Segments arriving from the transcriber do not overlap, so
    /// a difference between before and after is what aligning did rather than what it inherited —
    /// which is exactly what measuring from a saved archive cannot tell you.
    /// </summary>
    [Fact]
    public void CrowdingCreatedIsSeparatedFromCrowdingInherited()
    {
        var report = AlignmentCrowding.Describe(
            [Said("one", 0, 5), Said("two", 5, 10), Said("three", 10, 15)],
            [Placed("one", 0, 6), Placed("two", 4, 11), Placed("three", 9, 15)]);

        Assert.Equal(0, report.OverlappedBefore);
        Assert.Equal(2, report.OverlappedAfter);
    }

    [Fact]
    public void InheritedOverlapIsReportedAsSuch()
    {
        var report = AlignmentCrowding.Describe(
            [Said("one", 0, 6), Said("two", 4, 10)],
            [Placed("one", 0, 6), Placed("two", 4, 10)]);

        Assert.Equal(1, report.OverlappedBefore);
        Assert.Equal(1, report.OverlappedAfter);
    }

    /// <summary>A segment inside another can never be the one being spoken.</summary>
    [Fact]
    public void SwallowedSegmentsAreCounted()
    {
        var report = AlignmentCrowding.Describe(
            [Said("long", 0, 15), Said("short", 15, 17)],
            [Placed("long", 0, 15), Placed("short", 3, 4)]);

        Assert.Equal(1, report.Swallowed);
        Assert.Equal("short", report.WorstText);
    }

    [Fact]
    public void HowFarSegmentsMovedIsReportedWorstFirst()
    {
        var report = AlignmentCrowding.Describe(
            [Said("a", 10, 12), Said("b", 20, 22)],
            [Placed("a", 9.5, 12), Placed("b", 17, 22)]);

        Assert.Equal(2, report.Moved.Count);
        Assert.Equal("b", report.Moved[0].Text);
        Assert.Equal(-3, report.Moved[0].By, 3);
    }

    /// <summary>
    /// A moved-by figure across two different segments is noise, so only segments recognisable on
    /// both sides are compared.
    /// </summary>
    [Fact]
    public void SegmentsThatChangedTextAreNotComparedForMovement()
    {
        var report = AlignmentCrowding.Describe(
            [Said("before splitting", 10, 20)],
            [Placed("before", 4, 12)]);

        Assert.Empty(report.Moved);
    }

    [Fact]
    public void RoundingIsNotStacking()
    {
        var report = AlignmentCrowding.Describe(
            [Said("one", 0, 5), Said("two", 5, 10)],
            [Placed("one", 0, 5.01), Placed("two", 5, 10)]);

        Assert.Equal(0, report.OverlappedAfter);
    }

    [Fact]
    public void TheReportReadsAsText()
    {
        var text = AlignmentCrowding.Format(
            AlignmentCrowding.Describe(
                [Said("one", 0, 5), Said("two", 5, 10)],
                [Placed("one", 0, 6), Placed("two", 4, 10)]),
            "a recording");

        Assert.Contains("a recording", text, StringComparison.Ordinal);
        Assert.Contains("Overlapping before  0", text, StringComparison.Ordinal);
        Assert.Contains("Overlapping after   1", text, StringComparison.Ordinal);
    }

    [Fact]
    public void NothingAtAllIsNotAFailure()
    {
        var report = AlignmentCrowding.Describe([], []);

        Assert.Equal(0, report.Segments);
        Assert.Empty(report.Moved);
    }
}
