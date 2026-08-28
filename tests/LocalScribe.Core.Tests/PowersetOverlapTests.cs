using LocalScribe.Core.Diarization;
using Xunit;

namespace LocalScribe.Core.Tests;

public class PowersetOverlapTests
{
    /// <summary>Frames × 3 speakers, written as rows of who is active.</summary>
    private static bool[] Frames(params int[][] activePerFrame)
    {
        var active = new bool[activePerFrame.Length * 3];

        for (var frame = 0; frame < activePerFrame.Length; frame++)
        {
            foreach (var speaker in activePerFrame[frame])
            {
                active[(frame * 3) + speaker] = true;
            }
        }

        return active;
    }

    [Fact]
    public void TwoVoicesAtOnceMakeARun()
    {
        var active = Frames([0], [0, 1], [0, 1], [1], [1]);

        var runs = PowersetDecoder.OverlappedFrames(active, 5, 3).ToList();

        Assert.Equal([(1, 3)], runs);
    }

    [Fact]
    public void OneVoiceNeverDoes()
    {
        var active = Frames([0], [0], [1], [1], []);

        Assert.Empty(PowersetDecoder.OverlappedFrames(active, 5, 3));
    }

    [Fact]
    public void ARunReachingTheLastFrameStillCloses()
    {
        var active = Frames([0], [0, 2], [0, 2]);

        Assert.Equal([(1, 3)], PowersetDecoder.OverlappedFrames(active, 3, 3).ToList());
    }

    [Fact]
    public void SeparateContestsStaySeparate()
    {
        var active = Frames([0, 1], [0], [1, 2], [2]);

        Assert.Equal(
            [(0, 1), (2, 3)],
            PowersetDecoder.OverlappedFrames(active, 4, 3).ToList());
    }
}
