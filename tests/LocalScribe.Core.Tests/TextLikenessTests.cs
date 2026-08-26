using LocalScribe.Core.Alignment;
using Xunit;

namespace LocalScribe.Core.Tests;

public class TextLikenessTests
{
    [Fact]
    public void TextOverItsOwnAudioScoresHigh() =>
        Assert.True(TextLikeness.Share(
            "For more news and insights", "formorenewsandinsights") > 0.95);

    /// <summary>The decode misspells freely, and resemblance must survive that.</summary>
    [Fact]
    public void MisspellingsDoNotSinkTheScore() =>
        Assert.True(TextLikeness.Share(
            "That's Raghu Nandakumara from the Black Hat Show floor.",
            "thatsragunandakumarafromtheblakhatshowflor") > 0.85);

    /// <summary>
    /// The case the measure exists for: an invented line crammed onto somebody else's speech.
    /// It is placed, it spans time, and it does not read as itself.
    /// </summary>
    [Fact]
    public void TextOverSomebodyElsesAudioScoresLow() =>
        Assert.True(TextLikeness.Share(
            "And stay tuned to this feed for our award-winning flagship podcast",
            "soareallyexcitingconferencethatsragunandakumara") < 0.55);

    [Fact]
    public void EmptyTextScoresNothing() =>
        Assert.Equal(0, TextLikeness.Share("", "anything"));

    [Fact]
    public void SilenceScoresNothing() =>
        Assert.Equal(0, TextLikeness.Share("some words", ""));
}
