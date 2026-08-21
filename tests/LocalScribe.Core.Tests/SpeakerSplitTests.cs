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

        Assert.True(clear.Separation > RealTwoSpeakerDistance);
    }

    // The two cases below are calibrated to distances measured on real audio rather than chosen,
    // because choosing them is how the refusal test came to reject every genuine split. Two
    // speakers' paragraphs sit 0.29 apart in cosine distance and barely move with paragraph
    // length: 0.33 at five seconds, 0.29 at fifteen, 0.29 at forty-five. One speaker's
    // paragraphs forced into two groups sit at 0.036 and 0.005.
    private const double RealTwoSpeakerDistance = 0.29;

    private const double RealOneSpeakerDistance = 0.036;

    /// <summary>An angle whose cosine distance from zero is <paramref name="distance"/>.</summary>
    private static double Apart(double distance) => Math.Acos(1 - distance);

    /// <summary>
    /// The failure the user hit. Two speakers really are only about 0.29 apart at paragraph
    /// length, and an earlier refusal test demanded 0.294 — so it declined to split anything on
    /// any recording whose paragraphs ran longer than a few seconds, which is most of them.
    /// </summary>
    [Fact]
    public void TwoVoicesAtTheDistanceRealSpeakersSitApartAreSplit()
    {
        var them = Apart(RealTwoSpeakerDistance);

        var result = SpeakerSplit.ByExample(
            Voice(Alice),
            [Voice(them), Voice(Alice, 0.02), Voice(them, 0.03), Voice(them, -0.02)]);

        Assert.True(result.Split);
        Assert.Equal([1], result.JoinsExample);
    }

    /// <summary>
    /// And the other side of the same number. A ratio of between-group to within-group scatter
    /// cannot tell this apart — one speaker cut in half scores 2.25, well past any sane ratio
    /// bar — so the absolute distance has to be what decides it.
    /// </summary>
    [Fact]
    public void OneVoiceAtTheDistanceOneSpeakerVariesByIsNotSplit()
    {
        var drift = Apart(RealOneSpeakerDistance);

        var result = SpeakerSplit.ByExample(
            Voice(Alice),
            [Voice(drift), Voice(drift, 0.01), Voice(Alice, 0.005), Voice(drift, -0.01)]);

        Assert.False(result.Split);
        Assert.Empty(result.JoinsExample);
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
