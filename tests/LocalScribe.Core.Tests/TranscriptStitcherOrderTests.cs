using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class TranscriptStitcherOrderTests
{
    private static TranscriptSegment Segment(string text, double start, double end) =>
        new(text, start, end);

    /// <summary>
    /// Taken from a real recording. Trimming the repeated words at a window seam leaves the
    /// second segment's start where it was, so both segments claim the same two seconds — and
    /// anything walking the transcript in time then walks backwards.
    /// </summary>
    [Fact]
    public void TwoSegmentsCannotHoldTheSameMoment()
    {
        var stitched = new TranscriptStitcher().Stitch(
        [
            [Segment("That's how it's defined in the Old Testament.", 27.09, 29.98)],
            [Segment("Elijah and in Jonah.", 27.98, 32.02)],
        ]);

        Assert.Equal(2, stitched.Count);
        Assert.True(stitched[1].StartSeconds >= stitched[0].EndSeconds,
            $"the second segment starts at {stitched[1].StartSeconds:F2}, before the first ends at {stitched[0].EndSeconds:F2}");
    }

    [Fact]
    public void ASegmentPushedForwardKeepsItsLength()
    {
        var stitched = new TranscriptStitcher().Stitch(
        [
            [Segment("first thing said", 0, 10)],
            [Segment("second thing said", 8, 12)],
        ]);

        Assert.Equal(10, stitched[1].StartSeconds, 2);
        Assert.True(stitched[1].EndSeconds - stitched[1].StartSeconds >= 4 - 0.01,
            "the segment should keep the four seconds it was given");
    }

    [Fact]
    public void SegmentsThatAlreadyRunInOrderAreLeftAlone()
    {
        var stitched = new TranscriptStitcher().Stitch(
        [
            [Segment("one thing", 0, 5)],
            [Segment("another thing", 6, 10)],
        ]);

        Assert.Equal(0, stitched[0].StartSeconds, 2);
        Assert.Equal(6, stitched[1].StartSeconds, 2);
        Assert.Equal(10, stitched[1].EndSeconds, 2);
    }

    /// <summary>Every segment after a pushed one is pushed clear of it in turn.</summary>
    [Fact]
    public void AWholeRunComesOutInOrder()
    {
        var stitched = new TranscriptStitcher().Stitch(
        [
            [Segment("alpha here", 0, 10)],
            [Segment("bravo here", 5, 9)],
            [Segment("charlie here", 6, 11)],
            [Segment("delta here", 20, 25)],
        ]);

        for (var i = 1; i < stitched.Count; i++)
        {
            Assert.True(stitched[i].StartSeconds >= stitched[i - 1].EndSeconds,
                $"segment {i} starts at {stitched[i].StartSeconds:F2}, before {stitched[i - 1].EndSeconds:F2}");
        }
    }
}
