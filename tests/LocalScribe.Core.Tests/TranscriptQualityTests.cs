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
}
