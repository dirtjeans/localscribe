using LocalScribe.Core.Alignment;
using Xunit;

namespace LocalScribe.Core.Tests;

public class AlignmentScoresTests
{
    private const int Alphabet = 4;

    /// <summary>A window whose every value says which frame it came from.</summary>
    private static float[] Window(int firstFrame, int frames) =>
        [.. Enumerable.Range(0, frames * Alphabet).Select(i => (float)(firstFrame + (i / Alphabet)))];

    [Fact]
    public void FramesMapToTheTimesTheyCover()
    {
        var scores = new AlignmentScores(1000, Alphabet, 0.02);

        Assert.Equal(0, scores.FrameAt(0));
        Assert.Equal(50, scores.FrameAt(1));
        Assert.Equal(1.0, scores.SecondsAt(50), 6);
    }

    /// <summary>Asking about a moment past the end must not walk off the grid.</summary>
    [Fact]
    public void AnInstantBeyondTheRecordingIsClamped()
    {
        var scores = new AlignmentScores(100, Alphabet, 0.02);

        Assert.Equal(100, scores.FrameAt(9999));
        Assert.Equal(0, scores.FrameAt(-5));
    }

    [Fact]
    public void AWindowLandsWhereItIsPut()
    {
        var scores = new AlignmentScores(100, Alphabet, 0.02);

        scores.Fill(30, Window(30, 10));

        var read = scores.Between(30, 10);
        Assert.Equal(30f, read[0]);
        Assert.Equal(39f, read[^1]);
    }

    /// <summary>
    /// The number of frames a network returns for a given number of samples is its own business,
    /// so the last window can reach past the end. Dropping the overhang is right; throwing is not.
    /// </summary>
    [Fact]
    public void AWindowOverhangingTheEndIsTrimmed()
    {
        var scores = new AlignmentScores(100, Alphabet, 0.02);

        scores.Fill(95, Window(95, 20));

        Assert.Equal(95f, scores.Between(95, 5)[0]);
    }

    [Fact]
    public void AWindowStartingPastTheEndIsIgnored()
    {
        var scores = new AlignmentScores(100, Alphabet, 0.02);

        scores.Fill(200, Window(200, 5));

        Assert.Equal(0f, scores.Between(99, 1)[0]);
    }

    /// <summary>Successive windows must abut, not overlap or leave a gap.</summary>
    [Fact]
    public void WindowsLaidSideBySideCoverEveryFrame()
    {
        var scores = new AlignmentScores(90, Alphabet, 0.02);

        for (var at = 0; at < 90; at += 30)
        {
            scores.Fill(at, Window(at, 30));
        }

        var all = scores.Between(0, 90);

        for (var frame = 0; frame < 90; frame++)
        {
            Assert.Equal((float)frame, all[frame * Alphabet]);
        }
    }

    [Fact]
    public void AWindowMustBeAWholeNumberOfFrames() =>
        Assert.Throws<ArgumentException>(() =>
            new AlignmentScores(10, Alphabet, 0.02).Fill(0, new float[Alphabet + 1]));

    [Fact]
    public void ReadingPastTheEndIsRefused() =>
        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            var scores = new AlignmentScores(10, Alphabet, 0.02);
            _ = scores.Between(8, 5).Length;
        });

    [Fact]
    public void AnEmptyGridIsRefused() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new AlignmentScores(0, Alphabet, 0.02));
}
