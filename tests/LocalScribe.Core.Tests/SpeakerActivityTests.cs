using LocalScribe.Core.Diarization;
using Xunit;

namespace LocalScribe.Core.Tests;

public class SpeakerActivityTests
{
    private const int Speakers = 4;

    /// <summary>A hundred frames to the second; each entry is a speaker talking over a range.</summary>
    private static SpeakerActivity Heard(int frames, params (int Speaker, int From, int To)[] talking)
    {
        var probabilities = new float[frames * Speakers];

        foreach (var (speaker, from, to) in talking)
        {
            for (var frame = from; frame < to; frame++)
            {
                probabilities[(frame * Speakers) + speaker] = 0.9f;
            }
        }

        return new SpeakerActivity(probabilities, frames, Speakers);
    }

    [Fact]
    public void ABreathIsNotABoundary()
    {
        var turns = Heard(300, (0, 0, 100), (0, 110, 200)).Turns();

        var turn = Assert.Single(turns);
        Assert.Equal(0.0, turn.StartSeconds, 6);
        Assert.Equal(2.0, turn.EndSeconds, 6);
    }

    [Fact]
    public void ARealPauseIs()
    {
        var turns = Heard(300, (0, 0, 100), (0, 150, 250)).Turns();

        Assert.Equal(2, turns.Count);
        Assert.All(turns, turn => Assert.Equal(0, turn.Speaker));
    }

    [Fact]
    public void AFlickerIsNotAPerson()
    {
        // Channel 1 rises for five hundredths of a second: a cough, not a speaker. Channel 2 is
        // a real second voice and must be Speaker 2, not Speaker 3 with a hole before it.
        var turns = Heard(400, (0, 0, 100), (1, 120, 125), (2, 200, 300)).Turns();

        Assert.Equal([0, 1], turns.Select(turn => turn.Speaker));
    }

    [Fact]
    public void SpeakersAreNumberedAsTheyArrive()
    {
        var turns = Heard(400, (3, 0, 100), (0, 200, 300)).Turns();

        Assert.Equal([0, 1], turns.Select(turn => turn.Speaker));
        Assert.Equal(0.0, turns[0].StartSeconds, 6);
    }

    [Fact]
    public void TwoVoicesAtOnceAreBothKeptAndMarked()
    {
        var activity = Heard(300, (0, 0, 200), (1, 150, 300));

        var turns = activity.Turns();
        Assert.Equal(2, turns.Count);
        Assert.True(turns[0].EndSeconds > turns[1].StartSeconds);

        var (start, end) = Assert.Single(activity.Overlaps());
        Assert.Equal(1.5, start, 6);
        Assert.Equal(2.0, end, 6);
    }

    [Fact]
    public void TakingTurnsIsNotCrosstalk()
    {
        Assert.Empty(Heard(300, (0, 0, 150), (1, 150, 300)).Overlaps());
    }

    [Fact]
    public void ARunReachingTheEndStillCloses()
    {
        var turn = Assert.Single(Heard(200, (0, 100, 200)).Turns());

        Assert.Equal(2.0, turn.EndSeconds, 6);
    }

    [Fact]
    public void RawRunsCountWhatTheModelSaidBeforeTidying()
    {
        var activity = Heard(300, (0, 0, 100), (0, 110, 200), (1, 120, 125));

        Assert.Equal(3, activity.RawRuns());
        Assert.Single(activity.Turns());
    }
}
