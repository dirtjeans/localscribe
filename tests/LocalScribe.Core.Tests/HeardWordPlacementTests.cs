using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class HeardWordPlacementTests
{
    private static List<WordTimings.Word> Heard(params (string Text, double At)[] words) =>
        [.. words.Select(w => new WordTimings.Word(w.Text, w.At, w.At))];

    /// <summary>
    /// Only the times are borrowed. Cleanup and the stitcher change capitals and punctuation,
    /// and the words on screen are the segment's, so a rewrite of that kind must still pair.
    /// </summary>
    [Fact]
    public void PunctuationAndCapitalsDoNotStopThePairing()
    {
        var segment = new TranscriptSegment("Well, let's begin.", 10, 12);
        var heard = Heard(("well", 10.2), ("lets", 10.6), ("begin", 11.1));

        var words = HeardWordPlacement.Place(segment, heard);

        Assert.NotNull(words);
        Assert.Equal(["Well,", "let's", "begin."], words!.Select(w => w.Text));
        Assert.Equal([10.2, 10.6, 11.1], words.Select(w => w.StartSeconds));
        Assert.Equal([0, 6, 12], words.Select(w => w.Offset));
    }

    /// <summary>
    /// Wrongly timed is worse than untimed: an untimed line is grey and says "wait", a wrongly
    /// timed one plays the wrong sentence when clicked. Too few paired words is a refusal.
    /// </summary>
    [Fact]
    public void ASegmentMostlyUnheardStaysUntimed()
    {
        var segment = new TranscriptSegment("completely different words entirely here", 10, 12);
        var heard = Heard(("completely", 10.1), ("other", 10.5), ("things", 11.0));

        Assert.Null(HeardWordPlacement.Place(segment, heard));
    }

    /// <summary>
    /// A common word said a minute earlier is not this segment's word. Pairing searches only
    /// near the segment's own stamps.
    /// </summary>
    [Fact]
    public void WordsHeardFarAwayAreNotBorrowed()
    {
        var segment = new TranscriptSegment("and then", 60, 61);
        var heard = Heard(("and", 5.0), ("then", 5.3));

        Assert.Null(HeardWordPlacement.Place(segment, heard));
    }

    /// <summary>
    /// A word the decoder never produced — a filler cleanup restored, a number written out —
    /// sits between its neighbours, and the run never steps back in time, or the highlight
    /// would jump backwards mid-sentence.
    /// </summary>
    [Fact]
    public void AnUnheardWordSitsBetweenItsNeighbours()
    {
        var segment = new TranscriptSegment("one two three four", 0, 4);
        var heard = Heard(("one", 0.5), ("four", 2.5));

        var words = HeardWordPlacement.Place(segment, heard)!;

        Assert.Equal(0.5, words[0].StartSeconds);
        Assert.Equal(2.5, words[3].StartSeconds);
        Assert.InRange(words[1].StartSeconds, 0.5, 2.5);
        Assert.InRange(words[2].StartSeconds, words[1].StartSeconds, 2.5);
    }

    /// <summary>
    /// Windows overlap by two seconds, so the seam's words are heard twice, slightly out of
    /// order. Either copy will do; what must not happen is the second copy dragging a word
    /// earlier than the one before it.
    /// </summary>
    [Fact]
    public void TheSeamHeardTwiceStillGivesOneTimeline()
    {
        var segment = new TranscriptSegment("at the seam we go on", 28, 32);
        var heard = Heard(
            ("at", 28.1), ("the", 28.4), ("seam", 28.8), ("we", 29.3),
            ("the", 28.45), ("seam", 28.85), ("we", 29.25), ("go", 29.8), ("on", 30.2));

        var words = HeardWordPlacement.Place(segment, heard)!;

        for (var i = 1; i < words.Count; i++)
        {
            Assert.True(words[i].StartSeconds >= words[i - 1].StartSeconds);
        }

        Assert.Equal(30.2, words[^1].StartSeconds);
    }

    /// <summary>
    /// A pause is not spent lit on the word before it: each word runs to the next, but no
    /// longer than any word anybody says.
    /// </summary>
    [Fact]
    public void AWordDoesNotSwallowThePauseAfterIt()
    {
        var segment = new TranscriptSegment("stop. later", 0, 10);
        var heard = Heard(("stop", 0.2), ("later", 8.0));

        var words = HeardWordPlacement.Place(segment, heard)!;

        Assert.True(words[0].EndSeconds - words[0].StartSeconds <= WordTimings.LongestWordSeconds);
        Assert.True(words[0].EndSeconds <= words[1].StartSeconds);
    }

    [Fact]
    public void NothingHeardMeansNothingPlaced()
    {
        Assert.Null(HeardWordPlacement.Place(new TranscriptSegment("hello", 0, 1), []));
        Assert.Null(HeardWordPlacement.Place(new TranscriptSegment("", 0, 1), Heard(("hello", 0.2))));
    }
}
