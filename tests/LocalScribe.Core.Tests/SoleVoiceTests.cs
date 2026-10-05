using LocalScribe.Core.Diarization;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// Labelling a line before its words are timed is safe only when one person spoke all of it.
/// </summary>
public class SoleVoiceTests
{
    private static SpeakerTurn Turn(int speaker, double start, double end) => new(speaker, start, end);

    [Fact]
    public void OneVoiceThroughoutIsThatVoice()
    {
        IReadOnlyList<SpeakerTurn> turns = [Turn(2, 0, 12), Turn(2, 13, 30)];

        Assert.Equal("Speaker 3", SoleVoice.Of(turns, 0, 30));
    }

    [Fact]
    public void TwoVoicesAreNobodyYet()
    {
        // The line will be divided when its words are timed; labelling it whole now would put
        // one name on a conversation.
        IReadOnlyList<SpeakerTurn> turns = [Turn(0, 0, 20), Turn(1, 20, 30)];

        Assert.Null(SoleVoice.Of(turns, 0, 30));
    }

    [Fact]
    public void AShortVoiceElsewhereDoesNotCount()
    {
        IReadOnlyList<SpeakerTurn> turns = [Turn(0, 0, 30), Turn(1, 40, 60)];

        Assert.Equal("Speaker 1", SoleVoice.Of(turns, 0, 30));
    }

    [Fact]
    public void AFlickerFromSomeoneElseIsNotASecondPerson()
    {
        IReadOnlyList<SpeakerTurn> turns = [Turn(0, 0, 30), Turn(1, 14, 14.3)];

        Assert.Equal("Speaker 1", SoleVoice.Of(turns, 0, 30));
    }

    [Fact]
    public void AnInterjectionIsASecondPerson()
    {
        IReadOnlyList<SpeakerTurn> turns = [Turn(0, 0, 30), Turn(1, 14, 15.2)];

        Assert.Null(SoleVoice.Of(turns, 0, 30));
    }

    [Fact]
    public void SilenceOrTooLittleSpeechIsNobody()
    {
        Assert.Null(SoleVoice.Of([], 0, 30));
        Assert.Null(SoleVoice.Of([Turn(0, 10, 10.6)], 0, 30));
    }
}
