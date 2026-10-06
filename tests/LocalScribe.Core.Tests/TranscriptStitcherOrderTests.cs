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

    /// <summary>
    /// Taken from the debate, the windows as the transcriber returned them. The fourth window
    /// began at 78.84 s, inside the third, and opened by repeating the third's last sentence.
    /// Trimmed of it, its text still claimed to start at 78.84 — before that sentence — and the
    /// sentence was printed thirty seconds late, after the whole of the fourth window.
    /// </summary>
    [Fact]
    public void TrimmingASeamDoesNotSendTheSentenceAfterIt()
    {
        var stitched = new TranscriptStitcher().Stitch(
        [
            [
                Segment("Not in the least.", 76.00, 76.88),
                Segment("I don't understand how you're using it in the least.", 76.88, 79.28),
                Segment("That's why I'm trying to define it.", 79.28, 80.84),
            ],
            [
                Segment("That's why I'm trying to define it. My definition of God as conscience is a lot more precise.", 78.84, 108.84),
            ],
            [Segment("ideas of God. I didn't make that point.", 106.84, 116.01)],
        ]);

        Assert.Equal(
            [
                "Not in the least.",
                "I don't understand how you're using it in the least.",
                "That's why I'm trying to define it.",
                "My definition of God as conscience is a lot more precise.",
                "ideas of God. I didn't make that point.",
            ],
            stitched.Select(s => s.Text));

        // The trimmed window starts where the sentence it repeated ended, not before it.
        Assert.Equal(80.84, stitched[3].StartSeconds, 2);
    }

    /// <summary>
    /// Within one window the decoder's order is the order said. A segment stamped later than
    /// the one written after it is moved back to it rather than sent past it.
    /// </summary>
    [Fact]
    public void AStampOutOfOrderInsideAWindowIsMovedNotObeyed()
    {
        var repaired = TranscriptStitcher.InDecoderOrder(
        [
            Segment("That's why I'm trying to define it.", 112.3, 113.8),
            Segment("My definition of God as conscience is a lot more precise.", 82.3, 112.3),
        ], 82.3, 112.3);

        var stitched = new TranscriptStitcher().Stitch([repaired]);

        Assert.Equal(
            ["That's why I'm trying to define it.", "My definition of God as conscience is a lot more precise."],
            stitched.Select(s => s.Text));
        Assert.Equal((82.3, 83.8), (stitched[0].StartSeconds, stitched[0].EndSeconds));
    }

    [Fact]
    public void AStampedLateSegmentKeepsItsLength()
    {
        var repaired = TranscriptStitcher.InDecoderOrder(
            [Segment("first", 20, 22), Segment("second", 5, 10), Segment("third", 10, 15)], 0, 30);

        Assert.Equal((5.0, 7.0), (repaired[0].StartSeconds, repaired[0].EndSeconds));
    }

    [Fact]
    public void AStampBeyondTheWindowKeepsItsLengthInsideIt()
    {
        var moved = Assert.Single(TranscriptStitcher.InDecoderOrder([Segment("late", 112.3, 113.8)], 52.3, 82.3));

        Assert.Equal(80.8, moved.StartSeconds, 3);
        Assert.Equal(82.3, moved.EndSeconds, 3);
    }

    [Fact]
    public void ASegmentRunningPastTheWindowOnlyLosesItsOverhang()
    {
        // Its start was heard inside the window; only the end is past the audio.
        var cut = Assert.Single(TranscriptStitcher.InDecoderOrder([Segment("cut", 80.0, 84.0)], 52.3, 82.3));

        Assert.Equal((80.0, 82.3), (cut.StartSeconds, cut.EndSeconds));
    }

    [Fact]
    public void NeverBeforeTheWindowBegan()
    {
        // Longer than the whole window: it cannot be fitted, only kept inside.
        var moved = Assert.Single(TranscriptStitcher.InDecoderOrder([Segment("long", 90, 130)], 52.3, 82.3));

        Assert.Equal((52.3, 82.3), (moved.StartSeconds, moved.EndSeconds));
    }

    [Fact]
    public void StampsInsideTheWindowAreUntouched()
    {
        IReadOnlyList<TranscriptSegment> segments = [Segment("a", 53, 60), Segment("b", 60, 82.3)];

        Assert.Equal(segments, TranscriptStitcher.InDecoderOrder(segments, 52.3, 82.3));
    }

    [Fact]
    public void SegmentsStampedAtTheSameMomentKeepTheDecodersOrder()
    {
        var stitched = new TranscriptStitcher().Stitch(
        [
            [Segment("First said.", 10, 10), Segment("Then this.", 10, 12)],
        ]);

        Assert.Equal(["First said.", "Then this."], stitched.Select(s => s.Text));
    }
}
