using LocalScribe.Core.Alignment;
using Xunit;

namespace LocalScribe.Core.Tests;

public class CredibleAnchorsTests
{
    // A dense seven-minute recording: ~5500 letter tokens over 440 seconds, as the podcast
    // that surfaced the inflation actually measures.
    private const int Tokens = 5500;
    private const double Limit = 440;

    [Fact]
    public void TheEndpointsAlwaysSurvive()
    {
        var kept = CredibleAnchors.Prune([(0, 0), (Limit, Tokens)], Tokens, Limit);

        Assert.Equal(2, kept.Count);
    }

    [Fact]
    public void HonestAnchorsSurvive()
    {
        // Text arriving at roughly the recording's own pace, with ordinary stamp noise.
        var kept = CredibleAnchors.Prune(
            [(0, 0), (100, 1200), (223, 2800), (355, 4400), (Limit, Tokens)], Tokens, Limit);

        Assert.Equal(5, kept.Count);
    }

    [Fact]
    public void AnAnchorLeavingMoreTextThanTheTimeCanCarryIsDropped()
    {
        // The inflation, clamped: a stamp claiming that with a second and a half left, a
        // quarter of the transcript is still unspoken. Believing it crams that text into the
        // last frames and pushes the whole tail out of the corridor.
        var kept = CredibleAnchors.Prune([(0, 0), (438.5, 4000), (Limit, Tokens)], Tokens, Limit);

        Assert.DoesNotContain(kept, a => a.Token == 4000);
    }

    [Fact]
    public void AnAnchorClaimingTextFasterThanSpeechIsDropped()
    {
        // The mirror lie: five hundred tokens said to have been spoken in the first second.
        var kept = CredibleAnchors.Prune([(0, 0), (1.0, 500), (Limit, Tokens)], Tokens, Limit);

        Assert.DoesNotContain(kept, a => a.Token == 500);
    }

    [Fact]
    public void AQuietRecordingsOneDenseSentenceIsNotImpossible()
    {
        // Ten minutes of near-silence holding one late sentence: the average pace is tiny, and
        // without the sprint floor the honest anchors of that sentence would read as lies.
        var kept = CredibleAnchors.Prune([(0, 0), (570, 0), (575, 60), (600, 60)], 60, 600);

        Assert.Equal(4, kept.Count);
    }
}
