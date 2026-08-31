using LocalScribe.Core.Alignment;
using Xunit;

namespace LocalScribe.Core.Tests;

public class AlignmentScoresPrefixTests
{
    [Fact]
    public void APrefixReadsTheSameMemory()
    {
        var scores = new AlignmentScores(10, 2, 0.02);
        scores.Fill(0, new float[] { 1, 2, 3, 4 });

        var prefix = scores.Prefix(2);

        Assert.Equal(2, prefix.Frames);
        Assert.Equal(scores.Between(0, 2).ToArray(), prefix.Between(0, 2).ToArray());

        // Filled through the whole grid, visible through the view: one grid, two windows onto
        // it, which is what lets a prefix be aligned while the scan writes further rows.
        scores.Fill(0, new float[] { 9, 9, 9, 9 });
        Assert.Equal(9, prefix.Between(0, 1)[0]);
    }

    [Fact]
    public void APrefixNeverReachesPastItsCut()
    {
        var prefix = new AlignmentScores(10, 2, 0.02).Prefix(3);

        Assert.Throws<System.ArgumentOutOfRangeException>(() => prefix.Between(2, 2));
    }

    [Fact]
    public void AskingForEverythingIsTheGridItself()
    {
        var scores = new AlignmentScores(5, 2, 0.02);

        Assert.Same(scores, scores.Prefix(5));
        Assert.Same(scores, scores.Prefix(50));
    }
}
