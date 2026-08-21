using LocalScribe.Core.Diarization;
using Xunit;

namespace LocalScribe.Core.Tests;

public class SpeakerSplitTests
{
    /// <summary>
    /// A voice, as a point in embedding space. Two distinct directions stand in for two people
    /// and <paramref name="drift"/> for the variation between things the same person said.
    /// </summary>
    private static float[] Voice(double angle, double drift = 0)
    {
        var theta = angle + drift;
        return [(float)Math.Cos(theta), (float)Math.Sin(theta), 0.05f];
    }

    private static readonly double Alice = 0.0;
    private static readonly double Bob = Math.PI / 2;

    [Fact]
    public void TheOtherSpeakersParagraphsAreFound()
    {
        var example = Voice(Alice);

        // Three of Alice's, two of Bob's, interleaved as they would be in a conversation.
        float[][] candidates =
        [
            Voice(Bob, 0.05),
            Voice(Alice, -0.04),
            Voice(Bob, -0.03),
            Voice(Alice, 0.02),
            Voice(Alice, 0.06),
        ];

        var result = SpeakerSplit.ByExample(example, candidates);

        Assert.True(result.Split);
        Assert.Equal([1, 3, 4], result.JoinsExample);
    }

    /// <summary>
    /// The case that matters most, because it is the one that damages a good transcript. If the
    /// user marks a paragraph that was correctly labelled, there is no second voice to find and
    /// the right answer is to change nothing else.
    /// </summary>
    [Fact]
    public void OneVoiceIsNotSplitInTwo()
    {
        var example = Voice(Alice);

        float[][] candidates =
        [
            Voice(Alice, 0.03),
            Voice(Alice, -0.05),
            Voice(Alice, 0.01),
            Voice(Alice, -0.02),
        ];

        var result = SpeakerSplit.ByExample(example, candidates);

        Assert.False(result.Split);
        Assert.Empty(result.JoinsExample);
    }

    [Fact]
    public void AParagraphOnItsOwnHasNothingToCompareAgainst()
    {
        var result = SpeakerSplit.ByExample(Voice(Alice), []);

        Assert.False(result.Split);
        Assert.Empty(result.JoinsExample);
    }

    /// <summary>
    /// The example is the one thing the user is certain about, so it anchors its own side even
    /// when every candidate belongs to the other person.
    /// </summary>
    [Fact]
    public void TheExampleCanBeTheOnlyOneOfItsVoice()
    {
        float[][] candidates = [Voice(Bob), Voice(Bob, 0.04), Voice(Bob, -0.03)];

        var result = SpeakerSplit.ByExample(Voice(Alice), candidates);

        Assert.True(result.Split);
        Assert.Empty(result.JoinsExample);
    }

    [Fact]
    public void SeparationReportsHowFarApartTheVoicesWere()
    {
        var clear = SpeakerSplit.ByExample(
            Voice(Alice),
            [Voice(Bob), Voice(Alice, 0.02)]);

        Assert.True(clear.Separation > SpeakerClustering.DefaultThreshold);
    }

    /// <summary>
    /// The case that sent the first attempt back to the drawing board, taken from the
    /// two-speaker sample: one of the other speaker's turns embeds closer to the example than
    /// one of the example speaker's own turns does. Splitting about two centroids gets this
    /// wrong, because the one bad point drags a centroid across the gap. Average linkage over
    /// every pair does not.
    /// </summary>
    [Fact]
    public void OneMisleadingPointDoesNotDecideTheSplit()
    {
        var example = Voice(Alice);

        float[][] candidates =
        [
            Voice(Alice, 0.26),      // truly Alice, but not a close match
            Voice(Bob, -0.60),       // truly Bob, and closer to the example than the above
            Voice(Alice, 0.07),
            Voice(Bob, 0.10),
            Voice(Bob, -0.05),
        ];

        var result = SpeakerSplit.ByExample(example, candidates);

        Assert.True(result.Split);
        Assert.Equal([0, 2], result.JoinsExample);
    }

    /// <summary>Order in, order out: callers index their own paragraph list with these.</summary>
    [Fact]
    public void IndexesReferToTheCandidateList()
    {
        float[][] candidates = [Voice(Bob), Voice(Bob, 0.02), Voice(Alice, 0.01)];

        var result = SpeakerSplit.ByExample(Voice(Alice), candidates);

        Assert.True(result.Split);
        Assert.Equal([2], result.JoinsExample);
    }
}
