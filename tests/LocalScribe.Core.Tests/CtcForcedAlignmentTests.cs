using LocalScribe.Core.Alignment;
using Xunit;

namespace LocalScribe.Core.Tests;

public class CtcForcedAlignmentTests
{
    private const int Blank = 0;
    private const int Alphabet = 4;      // blank, a, b, c

    /// <summary>
    /// Frame-major log probabilities from a description like "a a - b", where '-' is the blank.
    /// The named token gets almost all the probability for that frame.
    /// </summary>
    private static float[] Heard(params string[] frames)
    {
        var data = new float[frames.Length * Alphabet];

        for (var t = 0; t < frames.Length; t++)
        {
            var winner = frames[t] switch { "-" => Blank, "a" => 1, "b" => 2, _ => 3 };

            for (var k = 0; k < Alphabet; k++)
            {
                data[(t * Alphabet) + k] = k == winner ? -0.05f : -4.0f;
            }
        }

        return data;
    }

    [Fact]
    public void EachLetterLandsOnTheFrameItWasHeard()
    {
        var heard = Heard("a", "-", "-", "b", "-", "-", "c");

        var placed = CtcForcedAlignment.Align(heard, 7, Alphabet, [1, 2, 3], Blank);

        Assert.NotNull(placed);
        Assert.Equal(3, placed!.Count);
        Assert.Equal(0, placed[0].FirstFrame);
        Assert.Equal(3, placed[1].FirstFrame);
        Assert.Equal(6, placed[2].FirstFrame);
    }

    /// <summary>
    /// A letter owns only the frames the path actually spent on it. The quiet between two
    /// letters belongs to the blank, not to whichever letter came first — that is what CTC
    /// means by a blank, and treating the gap as part of the preceding letter would stretch
    /// every word to meet the next one.
    /// </summary>
    [Fact]
    public void TheQuietBetweenLettersBelongsToNeither()
    {
        var placed = CtcForcedAlignment.Align(Heard("a", "-", "-", "b"), 4, Alphabet, [1, 2], Blank);

        Assert.NotNull(placed);
        Assert.Equal(0, placed![0].FirstFrame);
        Assert.Equal(0, placed[0].LastFrame);
        Assert.Equal(3, placed[1].FirstFrame);
        Assert.Equal(3, placed[1].LastFrame);
    }

    /// <summary>
    /// Two of the same letter running together need a blank between them, or CTC would read the
    /// pair as one. There must be room for it.
    /// </summary>
    [Fact]
    public void ARepeatedLetterNeedsRoomForTheGap()
    {
        Assert.Null(CtcForcedAlignment.Align(Heard("a", "a"), 2, Alphabet, [1, 1], Blank));

        var placed = CtcForcedAlignment.Align(Heard("a", "-", "a"), 3, Alphabet, [1, 1], Blank);

        Assert.NotNull(placed);
        Assert.True(placed![1].FirstFrame > placed[0].FirstFrame);
    }

    [Fact]
    public void MoreLettersThanFramesCannotBeAligned() =>
        Assert.Null(CtcForcedAlignment.Align(Heard("a", "b"), 2, Alphabet, [1, 2, 3], Blank));

    [Fact]
    public void TheLettersComeBackInOrder()
    {
        var placed = CtcForcedAlignment.Align(
            Heard("a", "b", "c", "a", "b", "c"), 6, Alphabet, [1, 2, 3, 1, 2, 3], Blank);

        Assert.NotNull(placed);
        for (var i = 1; i < placed!.Count; i++)
        {
            Assert.True(placed[i].FirstFrame > placed[i - 1].FirstFrame,
                $"letter {i} was placed at {placed[i].FirstFrame}, not after {placed[i - 1].FirstFrame}");
        }
    }

    /// <summary>
    /// Silence at the front belongs to nobody. A transcript that starts two frames in should not
    /// have its first letter dragged back to frame zero.
    /// </summary>
    [Fact]
    public void LeadingSilenceIsNotClaimed()
    {
        var placed = CtcForcedAlignment.Align(Heard("-", "-", "-", "a", "b"), 5, Alphabet, [1, 2], Blank);

        Assert.NotNull(placed);
        Assert.Equal(3, placed![0].FirstFrame);
    }

    [Fact]
    public void ConfidenceComesBackWithThePlacement()
    {
        var placed = CtcForcedAlignment.Align(Heard("a", "b"), 2, Alphabet, [1, 2], Blank);

        Assert.NotNull(placed);
        Assert.All(placed!, p => Assert.InRange(p.Score, -1.0, 0.0));
    }

    [Fact]
    public void NothingToAlignIsNotAFailure() =>
        Assert.Null(CtcForcedAlignment.Align(Heard("a"), 1, Alphabet, [], Blank));
}
