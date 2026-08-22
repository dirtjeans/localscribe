using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class TranscriptQualityTests
{
    // Both readings below are real output from the same model over the same seconds of speech,
    // one pass apart.
    private const string Formatted =
        "Okay, I'm going to test the transcription ability one more time. "
        + "Let's see how well it works. Does it punctuate well? I hope so.";

    private const string Degenerate =
        "okay i'm going to test the transcription ability one more time "
        + "let's see how well it works does it punctuate well hope so";

    [Fact]
    public void TheDegenerateReadingIsRecognised() =>
        Assert.True(TranscriptQuality.LooksUnformatted(Degenerate));

    [Fact]
    public void TheFormattedReadingIsNot() =>
        Assert.False(TranscriptQuality.LooksUnformatted(Formatted));

    /// <summary>
    /// An apostrophe is not sentence punctuation. "let's" appears in both readings, so counting
    /// it would make the degenerate one look formatted.
    /// </summary>
    [Fact]
    public void AnApostropheAloneDoesNotCountAsFormatting() =>
        Assert.True(TranscriptQuality.LooksUnformatted("let's see how well it works"));

    [Fact]
    public void ACapitalAloneIsEnoughToCount() =>
        Assert.False(TranscriptQuality.LooksUnformatted("Okay let us see"));

    [Fact]
    public void EmptyTextIsNotTreatedAsDegenerate() =>
        Assert.False(TranscriptQuality.LooksUnformatted("   "));

    [Fact]
    public void TheFormattedReadingIsPreferred() =>
        Assert.True(TranscriptQuality.PreferCandidate(Degenerate, Formatted));

    [Fact]
    public void AFormattedReadingIsNeverReplacedByAnother() =>
        Assert.False(TranscriptQuality.PreferCandidate(Formatted, Formatted));

    [Fact]
    public void ADegenerateCandidateIsNoImprovement() =>
        Assert.False(TranscriptQuality.PreferCandidate(Degenerate, Degenerate));

    /// <summary>
    /// The guard must not trade words for commas. An earlier pass that had heard only half the
    /// sentence is not a better reading of it, however tidily it is presented.
    /// </summary>
    [Fact]
    public void AShorterFormattedFragmentDoesNotReplaceACompleteReading() =>
        Assert.False(TranscriptQuality.PreferCandidate(Degenerate, "Okay, I'm going to test."));

    [Fact]
    public void ASlightlyShorterReadingIsStillAcceptable()
    {
        // Whisper drops the odd word between passes; that is not the same as losing the tail.
        const string current = "one two three four five six seven eight nine ten";
        const string candidate = "One, two, three, four, five, six, seven, eight, nine.";

        Assert.True(TranscriptQuality.PreferCandidate(current, candidate));
    }

    /// <summary>
    /// The check that stops a repair becoming a rewrite. Conditioning the model on example text
    /// can leave the example in the output — a confidently punctuated sentence made partly of
    /// words nobody said, which reads worse than the flat one it replaced.
    /// </summary>
    [Fact]
    public void ARetryThatInventsWordsIsNotTheSameThing()
    {
        const string original = "okay i am going to test the transcription ability one more time";
        const string garbled = ". and I you for Okay, that is right. Okay's see. Okay";

        Assert.False(TranscriptQuality.SaysTheSameThing(original, garbled));
    }

    [Fact]
    public void TheSameWordsPunctuatedDifferentlyStillCount()
    {
        Assert.True(TranscriptQuality.SaysTheSameThing(
            "okay i am going to test this one more time",
            "Okay, I am going to test this one more time."));
    }

    [Fact]
    public void ALittleDriftIsTolerated()
    {
        // Two readings of the same speech rarely agree on every word, and demanding they do
        // would reject every genuine repair.
        Assert.True(TranscriptQuality.SaysTheSameThing(
            "one two three four five six seven eight nine ten",
            "One, two, three, four, five, six, seven, eight, nine."));
    }

    /// <summary>
    /// Whisper does not fall silent on audio it cannot read — it invents fluent sentences with
    /// no change in tone to warn anybody. Its own confidence is the only thing that tells them
    /// apart, and -1 is the figure OpenAI's implementation uses for a decode worth retrying.
    /// </summary>
    [Theory]
    [InlineData(-0.15, false)]   // ordinary clear speech
    [InlineData(-0.62, false)]   // unremarkable
    [InlineData(-1.10, true)]    // past the point the model would retry itself
    [InlineData(-2.40, true)]    // noise
    public void GuessworkIsRecognisedByTheModelsOwnConfidence(double confidence, bool guessing) =>
        Assert.Equal(guessing, TranscriptQuality.SoundsLikeGuesswork("a perfectly ordinary sentence of speech", confidence));

    [Fact]
    public void ConfidentWordsOverSilenceAreStillDisbelieved() =>
        Assert.True(TranscriptQuality.SoundsLikeGuesswork("words over silence", -0.2, noSpeechProbability: 0.9));

    /// <summary>
    /// Taken from the debate recording: thirty seconds where the model latched onto a phrase and
    /// could not let go. Its confidence through this is perfectly ordinary — it is certain of
    /// every word — so only the shape of the text gives it away.
    /// </summary>
    [Fact]
    public void AModelTalkingInCirclesIsRecognised()
    {
        const string looping =
            "I didn't make that point. You don't. I didn't make that point. You don't. I don't. "
            + "I don't. I don't. I don't. I don't. I don't. I don't. I don't. I don't. I don't. "
            + "I don't. I don't. I don't. I don't. I don't. I don't. I don't. I don";

        Assert.True(TranscriptQuality.LoopsOnItself(looping));
        Assert.True(TranscriptQuality.SoundsLikeGuesswork(looping, -0.1));
    }

    [Fact]
    public void OrdinarySpeechDoesNotLookLikeALoop()
    {
        const string ordinary =
            "My definition of God as conscience is a lot more precise and oriented than your "
            + "definition of the God that you hypothetically disbelieve in. But it's irrelevant "
            + "to the fault lines of this debate. How is it irrelevant? Because in common "
            + "parlance when we are talking about atheists we are not talking about that.";

        Assert.False(TranscriptQuality.LoopsOnItself(ordinary));
    }

    /// <summary>
    /// A genuine repetition for emphasis is not a loop, and a transcript that ellipsised it
    /// would be deleting something the speaker meant.
    /// </summary>
    [Fact]
    public void SayingSomethingTwiceForEmphasisIsNotALoop()
    {
        const string emphatic =
            "It's directly relevant. Atheists reject God, but they don't understand what they're "
            + "rejecting. But they don't understand what they're rejecting. You accept conscience "
            + "as a guide, and conscience is one of the defining characteristics of God.";

        Assert.False(TranscriptQuality.LoopsOnItself(emphatic));
    }

    /// <summary>
    /// The real shape of the failure, from the debate recording: twenty seconds of speech and
    /// then ten of the model stuck. The good part has to survive — discarding the segment would
    /// take it too.
    /// </summary>
    [Fact]
    public void OnlyTheLoopingTailIsCut()
    {
        const string mixed =
            "Your point that there are these polisimous ideas of God. I didn't make that point. "
            + "I don't. I don't. I don't. I don't. I don't. I don't. I don't.";

        var trimmed = TranscriptQuality.TrimLoopedTail(mixed);

        Assert.StartsWith("Your point that there are these polisimous ideas of God.", trimmed, StringComparison.Ordinal);
        Assert.EndsWith(TranscriptQuality.Unintelligible, trimmed, StringComparison.Ordinal);
        Assert.DoesNotContain("I don't. I don't.", trimmed, StringComparison.Ordinal);
    }

    /// <summary>Emphasis is not a loop. Four times running is; twice is someone making a point.</summary>
    /// <summary>
    /// The exact text from the recording, ending mid-word. Matching strictly from the last word
    /// compares "don" against "don't", finds no repeat, and leaves the whole loop in place.
    /// </summary>
    [Fact]
    public void ALoopThatRunsOutMidPhraseIsStillFound()
    {
        const string cutOff =
            "I didn't make that point. You don't. I don't. I don't. I don't. "
            + "I don't. I don't. I don't. I don't. I don";

        var trimmed = TranscriptQuality.TrimLoopedTail(cutOff);

        Assert.EndsWith(TranscriptQuality.Unintelligible, trimmed, StringComparison.Ordinal);
        Assert.DoesNotContain("I don't. I don't.", trimmed, StringComparison.Ordinal);
    }

    [Fact]
    public void RepeatingYourselfTwiceIsLeftAlone()
    {
        const string emphatic =
            "Atheists reject God, but they don't understand what they're rejecting. "
            + "But they don't understand what they're rejecting.";

        Assert.Equal(emphatic, TranscriptQuality.TrimLoopedTail(emphatic));
    }

    [Fact]
    public void OrdinarySpeechKeepsItsEnding()
    {
        const string ordinary =
            "I just, I think your point is irrelevant to the way that people tend to use these words.";

        Assert.Equal(ordinary, TranscriptQuality.TrimLoopedTail(ordinary));
    }

    [Fact]
    public void ASegmentThatIsNothingButALoopBecomesAnEllipsis()
    {
        var trimmed = TranscriptQuality.TrimLoopedTail("I don't. I don't. I don't. I don't. I don't.");

        Assert.Equal(TranscriptQuality.Unintelligible, trimmed);
    }

    [Fact]
    public void AShortLineIsNotJudged() =>
        Assert.False(TranscriptQuality.LoopsOnItself("No. No. No. No."));

    [Fact]
    public void NothingToCompareAgainstIsNotAFailure() =>
        Assert.True(TranscriptQuality.SaysTheSameThing(string.Empty, "anything at all"));

    // Both of the following are real replies from phi-3.5-mini-instruct-qnn-npu asked to
    // punctuate the line below and told, in as many words, to keep every word and add no notes.
    private const string RawWindow =
        "okay i am going to test the transcription ability one more time lets see how well "
        + "it works does it punctuate well i hope so";

    [Fact]
    public void CleanupThatDropsAClauseIsRefused()
    {
        // "let's see how well it works" is simply gone.
        const string lost =
            "Okay, I'm going to test the transcription ability one more time. "
            + "Does it punctuate well? I hope so.";

        Assert.False(TranscriptQuality.IsFaithfulCleanup(RawWindow, lost));
    }

    [Fact]
    public void CleanupThatExplainsItselfIsRefused()
    {
        const string chatty =
            "Okay, I am going to test the transcription ability one more time. Let's see how "
            + "well it works. Does it punctuate well? I hope so. "
            + "(Note: I've added a question mark at the end of the sentence to indicate it's a "
            + "question, and capitalized the first letter of the sentence to follow standard "
            + "punctuation and capitalization rules.)";

        Assert.False(TranscriptQuality.IsFaithfulCleanup(RawWindow, chatty));
    }

    [Fact]
    public void AGenuineCleanupIsAccepted()
    {
        const string good =
            "Okay, I am going to test the transcription ability one more time. Let's see how "
            + "well it works. Does it punctuate well? I hope so.";

        Assert.True(TranscriptQuality.IsFaithfulCleanup(RawWindow, good));
    }

    /// <summary>Removing filler is the job, not a failure of it.</summary>
    [Fact]
    public void DroppingFillerIsStillFaithful()
    {
        Assert.True(TranscriptQuality.IsFaithfulCleanup(
            "so um i think uh we should probably just ship it you know",
            "So I think we should probably just ship it."));
    }

    [Fact]
    public void AnEmptyReplyIsRefused() =>
        Assert.False(TranscriptQuality.IsFaithfulCleanup(RawWindow, "   "));

    [Fact]
    public void AnEmptyWindowNeedsNoCheck() =>
        Assert.True(TranscriptQuality.IsFaithfulCleanup("   ", "anything"));
}
