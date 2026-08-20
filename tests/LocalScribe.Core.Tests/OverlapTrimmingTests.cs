using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// Repairing the seam between two windows that transcribed the same speech differently.
/// <para>
/// Whole-segment equality only catches the case where both passes divided the audio identically.
/// They usually do not, and the result is a visible stutter at every boundary.
/// </para>
/// </summary>
public class OverlapTrimmingTests
{
    /// <summary>Taken from a real run: this is what the seam actually looked like.</summary>
    [Fact]
    public void TheRepeatedTailOfThePreviousSpanIsRemoved()
    {
        var trimmed = TranscriptStitcher.TrimLeadingOverlap(
            "If the stitching works correctly,",
            "works correctly, this sentence will appear exactly once");

        Assert.Equal("this sentence will appear exactly once", trimmed);
    }

    [Fact]
    public void ASingleRepeatedWordIsRemoved()
    {
        var trimmed = TranscriptStitcher.TrimLeadingOverlap(
            "one at a time until",
            "until it reaches the end of the segment.");

        Assert.Equal("it reaches the end of the segment.", trimmed);
    }

    /// <summary>
    /// The two passes rarely agree on punctuation or casing at a boundary, so the comparison
    /// must not either.
    /// </summary>
    [Fact]
    public void PunctuationAndCasingDoNotDefeatTheMatch()
    {
        var trimmed = TranscriptStitcher.TrimLeadingOverlap(
            "and it is worth stating plainly.",
            "Is worth stating plainly — thank you for listening.");

        Assert.Equal("thank you for listening.", trimmed);
    }

    /// <summary>
    /// The longest overlap wins, so a word that happens to repeat inside the overlap does not
    /// shadow the real match and leave half the duplicate behind.
    /// </summary>
    [Fact]
    public void TheLongestOverlapWinsRatherThanTheFirstFound()
    {
        var trimmed = TranscriptStitcher.TrimLeadingOverlap(
            "the model runs on the model runs on the NPU",
            "the model runs on the NPU and stays there");

        Assert.Equal("and stays there", trimmed);
    }

    [Fact]
    public void TextWithNoOverlapIsLeftAlone()
    {
        const string candidate = "a completely different sentence";

        Assert.Equal(
            candidate,
            TranscriptStitcher.TrimLeadingOverlap("nothing in common here", candidate));
    }

    [Fact]
    public void AFullyContainedRepeatTrimsToNothing()
    {
        Assert.Equal(
            string.Empty,
            TranscriptStitcher.TrimLeadingOverlap("say that again please", "again please"));
    }

    [Fact]
    public void EmptyInputIsHandled()
    {
        Assert.Equal("hello", TranscriptStitcher.TrimLeadingOverlap(string.Empty, "hello"));
        Assert.Equal(string.Empty, TranscriptStitcher.TrimLeadingOverlap("hello", string.Empty));
    }

    /// <summary>
    /// The seam repair must not reach across a long gap. A phrase genuinely said again later is
    /// not an artefact of stitching.
    /// </summary>
    [Fact]
    public void RepetitionFarFromASeamSurvivesStitching()
    {
        var stitched = new TranscriptStitcher(boundaryToleranceSeconds: 2.5).Stitch(
        [
            [new TranscriptSegment("so that is the plan", 0, 3)],
            [new TranscriptSegment("that is the plan for next year", 600, 604)],
        ]);

        Assert.Equal(2, stitched.Count);
        Assert.Equal("that is the plan for next year", stitched[1].Text);
    }

    [Fact]
    public void RepetitionAtASeamIsTrimmed()
    {
        var stitched = new TranscriptStitcher(boundaryToleranceSeconds: 2.5).Stitch(
        [
            [new TranscriptSegment("so that is the plan", 0, 3)],
            [new TranscriptSegment("that is the plan for next year", 4, 8)],
        ]);

        Assert.Equal(2, stitched.Count);
        Assert.Equal("for next year", stitched[1].Text);
    }

    /// <summary>
    /// A pass re-reads its whole window, so the words it repeats routinely run past the end of
    /// any one previous segment. Comparing against only the most recent one left most of the
    /// repetition in place.
    /// </summary>
    [Fact]
    public void AnOverlapLongerThanOneSegmentIsStillFound()
    {
        const string previous =
            "so that chunking and stitching are both exercised. Whisper divides audio into";
        const string candidate =
            "so that chunking and stitching are both exercised. Whisper divides audio into "
            + "fixed windows and the decoder emits tokens one at a time.";

        var trimmed = TranscriptStitcher.TrimLeadingOverlap(previous, candidate);

        Assert.Equal("fixed windows and the decoder emits tokens one at a time.", trimmed);
    }

    /// <summary>
    /// The search is bounded, so an absurdly long repeat is left alone rather than matched by
    /// accident. Bounded is the point: past a certain length, repetition is speech, not a seam.
    /// </summary>
    [Fact]
    public void TheSearchIsBounded()
    {
        var words = string.Join(" ", Enumerable.Range(0, 80).Select(i => $"word{i}"));

        Assert.Equal(words, TranscriptStitcher.TrimLeadingOverlap(words, words, maxWords: 5));
    }
}
