using LocalScribe.Core.Diarization;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// Being told how many people are talking.
/// <para>
/// A threshold is a way of guessing the count, and on a real recording a threshold that is
/// slightly wrong fails in both directions at once: two voices that sound alike merge while one
/// that moves nearer the microphone splits in two. No single number fixes both, because they are
/// not the same error. Being told there are three ends the argument.
/// </para>
/// </summary>
public class ExactSpeakerCountTests
{
    private static float[] Near(float x, float y, float jitter) => [x + jitter, y - jitter, 0.1f];

    /// <summary>Three tight groups, so a threshold could in principle find them.</summary>
    private static List<float[]> ThreeGroups() =>
    [
        Near(1, 0, 0.01f), Near(1, 0, -0.01f),
        Near(0, 1, 0.01f), Near(0, 1, -0.01f),
        Near(-1, -1, 0.01f), Near(-1, -1, -0.01f),
    ];

    [Fact]
    public void ATooHighThresholdWouldMergeEverything() =>
        Assert.Single(SpeakerClustering.Cluster(ThreeGroups(), threshold: 5.0).Distinct());

    [Fact]
    public void ATooLowThresholdWouldSplitEverything() =>
        Assert.Equal(6, SpeakerClustering.Cluster(ThreeGroups(), threshold: 0.0).Distinct().Count());

    /// <summary>
    /// The count wins over the threshold in both directions, which is the whole point: the same
    /// number that would otherwise merge everything, and the one that would split everything,
    /// both give three.
    /// </summary>
    [Theory]
    [InlineData(5.0)]
    [InlineData(0.0)]
    [InlineData(0.42)]
    public void AKnownCountIsHonouredWhateverTheThreshold(double threshold)
    {
        var labels = SpeakerClustering.Cluster(ThreeGroups(), threshold, exactSpeakers: 3);

        Assert.Equal(3, labels.Distinct().Count());
    }

    [Fact]
    public void TheGroupsFoundAreTheRealOnes()
    {
        var labels = SpeakerClustering.Cluster(ThreeGroups(), exactSpeakers: 3);

        // Neighbouring pairs were built from the same point, so they belong together.
        Assert.Equal(labels[0], labels[1]);
        Assert.Equal(labels[2], labels[3]);
        Assert.Equal(labels[4], labels[5]);
        Assert.NotEqual(labels[0], labels[2]);
        Assert.NotEqual(labels[2], labels[4]);
    }

    [Fact]
    public void AskingForOneSpeakerMergesEverything() =>
        Assert.Single(SpeakerClustering.Cluster(ThreeGroups(), exactSpeakers: 1).Distinct());

    /// <summary>
    /// Asking for more speakers than there are embeddings cannot invent them, and must not hang
    /// trying.
    /// </summary>
    [Fact]
    public void AskingForMoreSpeakersThanSpansIsHarmless()
    {
        var labels = SpeakerClustering.Cluster(ThreeGroups(), exactSpeakers: 20);

        Assert.Equal(6, labels.Length);
        Assert.True(labels.Distinct().Count() <= 6);
    }

    [Fact]
    public void SpeakersAreStillNumberedByFirstAppearance()
    {
        var labels = SpeakerClustering.Cluster(ThreeGroups(), exactSpeakers: 3);

        Assert.Equal(0, labels[0]);
        Assert.Equal(1, labels[2]);
        Assert.Equal(2, labels[4]);
    }
}
