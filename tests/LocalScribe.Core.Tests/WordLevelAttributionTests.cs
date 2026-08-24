using LocalScribe.Core.Diarization;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class WordLevelAttributionTests
{
    /// <summary>Builds a segment whose words are evenly spaced, one per second from <paramref name="from"/>.</summary>
    private static TimedSegment Spoken(string text, double from)
    {
        var words = new List<WordTimings.Word>();
        var offset = 0;
        var at = from;

        foreach (var word in text.Split(' '))
        {
            words.Add(new WordTimings.Word(word, at, at + 0.9) { Offset = offset });
            offset += word.Length + 1;
            at += 1;
        }

        return new TimedSegment(new TranscriptSegment(text, from, at), words);
    }

    private static SpeakerTurn Turn(int who, double from, double to) => new(who, from, to);

    /// <summary>
    /// The case the whole change exists for: a change inside a segment, with no sentence end to
    /// cut at. This used to hand the entire segment to whoever held most of it.
    /// </summary>
    [Fact]
    public void ASegmentIsCutWhereTheVoiceChanges()
    {
        var timed = Spoken("one two three four five six", from: 0);

        var pieces = WordLevelAttribution.Apply([timed], [Turn(0, 0, 3), Turn(1, 3, 6)]);

        Assert.Equal(2, pieces.Count);
        Assert.Equal("one two three", pieces[0].Segment.Text);
        Assert.Equal("Speaker 1", pieces[0].Segment.Speaker);
        Assert.Equal("four five six", pieces[1].Segment.Text);
        Assert.Equal("Speaker 2", pieces[1].Segment.Speaker);
    }

    /// <summary>Each piece begins and ends where its own words do, not where its parent did.</summary>
    [Fact]
    public void EachPieceTakesTheTimesOfItsOwnWords()
    {
        var pieces = WordLevelAttribution.Apply(
            [Spoken("one two three four", from: 10)], [Turn(0, 10, 12), Turn(1, 12, 14)]);

        Assert.Equal(10, pieces[0].Segment.StartSeconds, 3);
        Assert.Equal(12, pieces[1].Segment.StartSeconds, 3);
    }

    /// <summary>
    /// A word cut out of the middle keeps the punctuation around it, because the text is sliced
    /// from the original rather than rebuilt by joining words.
    /// </summary>
    [Fact]
    public void PunctuationSurvivesTheCut()
    {
        var timed = Spoken("Well, obviously not. That's absurd.", from: 0);

        var pieces = WordLevelAttribution.Apply([timed], [Turn(0, 0, 3), Turn(1, 3, 5)]);

        Assert.Equal("Well, obviously not.", pieces[0].Segment.Text);
        Assert.Equal("That's absurd.", pieces[1].Segment.Text);
    }

    /// <summary>Highlighting reads offsets against the text it is given, so they must be rebased.</summary>
    [Fact]
    public void OffsetsAreRelativeToThePieceNotTheParent()
    {
        var pieces = WordLevelAttribution.Apply(
            [Spoken("one two three four", from: 0)], [Turn(0, 0, 2), Turn(1, 2, 4)]);

        Assert.Equal(0, pieces[1].Words[0].Offset);
        Assert.Equal("three", pieces[1].Segment.Text[..5]);
    }

    /// <summary>
    /// The window vote flickers at a real boundary. A fifth of a second is not a turn, and
    /// cutting on one divides a sentence around a speaker who was never there.
    /// </summary>
    [Fact]
    public void AFlickerIsNotATurn()
    {
        var pieces = WordLevelAttribution.Apply(
            [Spoken("one two three four five six", from: 0)],
            [Turn(0, 0, 6), Turn(1, 3.0, 3.2)]);

        Assert.Single(pieces);
        Assert.Equal("Speaker 1", pieces[0].Segment.Speaker);
    }

    /// <summary>One word against both its neighbours is a boundary landing early, not an interjection.</summary>
    [Fact]
    public void ALoneWordIsNotAnInterjection()
    {
        var pieces = WordLevelAttribution.Apply(
            [Spoken("one two three four five", from: 0)],
            [Turn(0, 0, 2), Turn(1, 2, 3), Turn(0, 3, 5)]);

        Assert.Single(pieces);
    }

    [Fact]
    public void ASegmentInOneVoiceIsLeftWhole()
    {
        var pieces = WordLevelAttribution.Apply(
            [Spoken("one two three", from: 0)], [Turn(0, 0, 3)]);

        Assert.Single(pieces);
        Assert.Equal("one two three", pieces[0].Segment.Text);
        Assert.Equal("Speaker 1", pieces[0].Segment.Speaker);
    }

    /// <summary>With no measured words there is nothing to cut on, so the old rule stands.</summary>
    [Fact]
    public void AnUntimedSegmentGoesToWhoeverHeldMostOfIt()
    {
        var bare = new TimedSegment(new TranscriptSegment("no words here", 0, 6), []);

        var pieces = WordLevelAttribution.Apply([bare], [Turn(0, 0, 1), Turn(1, 1, 6)]);

        Assert.Single(pieces);
        Assert.Equal("Speaker 2", pieces[0].Segment.Speaker);
    }

    [Fact]
    public void NoTurnsWorthCuttingOnChangesNothing()
    {
        var timed = Spoken("one two three", from: 0);

        var pieces = WordLevelAttribution.Apply([timed], [Turn(0, 0, 0.2)]);

        Assert.Single(pieces);
        Assert.Null(pieces[0].Segment.Speaker);
    }

    [Fact]
    public void NothingAtAllIsNotAFailure() =>
        Assert.Empty(WordLevelAttribution.Apply([], [Turn(0, 0, 5)]));
}
