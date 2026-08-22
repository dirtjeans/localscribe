using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class RepeatedPhraseTests
{
    /// <summary>
    /// The case from the debate recording. One sentence, transcribed twice, in the middle of a
    /// segment where nothing was looking for it.
    /// </summary>
    [Fact]
    public void ASentenceSaidTwiceIsSaidOnce()
    {
        const string doubled = "It's directly relevant. Atheists reject God, but they don't "
            + "understand what they're rejecting. But they don't understand what they're "
            + "rejecting. You accept conscience as a guide,";

        Assert.Equal(
            "It's directly relevant. Atheists reject God, but they don't understand what they're "
            + "rejecting. You accept conscience as a guide,",
            RepeatedPhrase.Trim(doubled));
    }

    /// <summary>The copy that survives is the first, which is punctuated to follow what precedes it.</summary>
    [Fact]
    public void TheFirstCopyIsTheOneKept()
    {
        var trimmed = RepeatedPhrase.Trim("and so the whole thing falls apart. And so the whole thing falls apart.");

        Assert.Equal("and so the whole thing falls apart.", trimmed);
    }

    /// <summary>A decoder that stuck more than once still leaves one copy.</summary>
    [Fact]
    public void ThreeCopiesBecomeOne()
    {
        var trimmed = RepeatedPhrase.Trim(
            "we have to look at the evidence we have to look at the evidence we have to look at the evidence");

        Assert.Equal("we have to look at the evidence", trimmed);
    }

    /// <summary>
    /// Punctuation and casing differ between passes and must not prevent a match. A decoder that
    /// loops often leaves a dash between the two copies, and counting that as a word between them
    /// would hide the repeat.
    /// </summary>
    [Fact]
    public void PunctuationAndCasingDoNotHideARepeat()
    {
        var trimmed = RepeatedPhrase.Trim("I think that is completely wrong — I think that is completely wrong.");

        Assert.Equal("I think that is completely wrong", trimmed);
    }

    // ----------------------------------------------------------------- speech that must survive

    /// <summary>People repeat themselves for emphasis, and that is not a fault to be corrected.</summary>
    [Theory]
    [InlineData("No, no, no, that isn't what I said.")]
    [InlineData("I know, I know.")]
    [InlineData("It was very, very good.")]
    [InlineData("Wait wait wait, hold on.")]
    [InlineData("What? What? I can't hear you.")]
    public void ShortRepetitionIsHowSpeechSounds(string spoken) =>
        Assert.Equal(spoken, RepeatedPhrase.Trim(spoken));

    /// <summary>
    /// Four words repeated is still short enough to be someone speaking. The line has to be
    /// somewhere, and it is drawn above this.
    /// </summary>
    [Fact]
    public void JustUnderTheThresholdSurvives()
    {
        const string spoken = "you have to go you have to go";

        Assert.Equal(spoken, RepeatedPhrase.Trim(spoken));
    }

    /// <summary>A phrase used twice with other words between it is not a repeat.</summary>
    [Fact]
    public void ARepeatMustBeImmediate()
    {
        const string spoken = "the point I am making is that the point I am making is fair";

        Assert.Equal(spoken, RepeatedPhrase.Trim(spoken));
    }

    /// <summary>Text with nothing wrong with it comes back as the same string.</summary>
    [Fact]
    public void UntouchedTextIsReturnedUnchanged()
    {
        const string spoken = "He's equal in stature to Moses. So it's not arbitrary.";

        Assert.Same(spoken, RepeatedPhrase.Trim(spoken));
    }

    [Fact]
    public void PunctuationAloneCannotCarryARepeat() =>
        Assert.Equal("- - - - - - - - - -", RepeatedPhrase.Trim("- - - - - - - - - -"));

    [Fact]
    public void EmptyTextIsNotAFailure() =>
        Assert.Equal(string.Empty, RepeatedPhrase.Trim(string.Empty));
}
