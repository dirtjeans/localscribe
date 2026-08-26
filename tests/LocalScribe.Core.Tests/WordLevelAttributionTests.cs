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

    /// <summary>
    /// A word that has swallowed the pause in front of it must not be handed to whoever was
    /// talking during that pause. On the debate recording "Atheists" was measured across 5.1
    /// seconds reaching back over the previous speaker, and went to them.
    /// </para>
    /// <para>
    /// Only the swallowed part is discounted. A word whose measured end is also in the wrong
    /// place is a misalignment and no attribution rule can rescue it — that is what the shorter
    /// backward search is for.
    /// </para>
    /// </summary>
    [Fact]
    public void AWordThatSwallowedTheSilenceBeforeItStillBelongsToItsSpeaker()
    {
        // One long first word, then two ordinary ones, all really said after 10s.
        IReadOnlyList<WordTimings.Word> words =
        [
            new WordTimings.Word("Atheists", 5.0, 11.5) { Offset = 0 },
            new WordTimings.Word("reject", 11.6, 12.1) { Offset = 9 },
            new WordTimings.Word("God.", 12.2, 12.7) { Offset = 16 },
        ];

        var pieces = WordLevelAttribution.Apply(
            [new TimedSegment(new TranscriptSegment("Atheists reject God.", 5, 12.7), words)],
            [Turn(0, 0, 10), Turn(1, 10, 13)]);

        Assert.Single(pieces);
        Assert.Equal("Speaker 2", pieces[0].Segment.Speaker);
    }

    /// <summary>
    /// The transcriber's order survives, even when the placed times disagree with it. Sorting by
    /// placement was tried and it scrambled text: where a repeated line made placements locally
    /// wrong by a second or two, segments leapfrogged each other and the outro was spliced into
    /// the middle of an answer. The decoder emits segments in the order they were spoken, and
    /// that order is the one thing it is always right about.
    /// </summary>
    [Fact]
    public void TheTranscribersOrderIsNeverRederivedFromTimes()
    {
        var first = Spoken("the first thing said", from: 21);
        var second = Spoken("the second thing said", from: 20);

        var pieces = WordLevelAttribution.Apply([first, second], [Turn(0, 0, 40)]);

        Assert.Equal("the first thing said", pieces[0].Segment.Text);
        Assert.Equal("the second thing said", pieces[1].Segment.Text);
    }

    /// <summary>
    /// From the debate recording, and the reason this rule exists: the sentence "…the context
    /// with which I am using the term God." was cut before its last word, and "God." went to the
    /// other speaker. A new speaker does not finish the last one's sentence for them.
    /// </summary>
    [Fact]
    public void TheEndOfASentenceGoesBackToWhoeverBeganIt()
    {
        var timed = Spoken("using the term God.", from: 0);

        var pieces = WordLevelAttribution.Apply([timed], [Turn(1, 0, 3), Turn(0, 3, 5)]);

        Assert.Single(pieces);
        Assert.Equal("Speaker 2", pieces[0].Segment.Speaker);
    }

    /// <summary>A whole sentence after a finished one is a real turn, not a tail.</summary>
    [Fact]
    public void AnAnswerAfterAFinishedSentenceIsLeftAlone()
    {
        var timed = Spoken("That is absurd. In what way?", from: 0);

        var pieces = WordLevelAttribution.Apply([timed], [Turn(0, 0, 3), Turn(1, 3, 8)]);

        Assert.Equal(2, pieces.Count);
        Assert.Equal("That is absurd.", pieces[0].Segment.Text);
        Assert.Equal("Speaker 2", pieces[1].Segment.Speaker);
    }

    /// <summary>A long stretch beginning mid-clause is more likely a real turn than a tail.</summary>
    [Fact]
    public void ALongContinuationIsNotTreatedAsATail()
    {
        var timed = Spoken("I was saying that and then you interrupted me completely", from: 0);

        var pieces = WordLevelAttribution.Apply([timed], [Turn(0, 0, 4), Turn(1, 4, 12)]);

        Assert.Equal(2, pieces.Count);
        Assert.Equal("Speaker 2", pieces[1].Segment.Speaker);
    }

    /// <summary>
    /// A segment opening with one word in another voice is the same fault at the other end, and
    /// the lone-word rule cannot see it: the first word has only one neighbour.
    /// </summary>
    [Fact]
    public void ALoneFirstWordJoinsWhatFollowsIt()
    {
        var timed = Spoken("All right, so that is interesting, but it goes on", from: 0);

        var pieces = WordLevelAttribution.Apply([timed], [Turn(0, 0, 1), Turn(1, 1, 12)]);

        Assert.Single(pieces);
        Assert.Equal("Speaker 2", pieces[0].Segment.Speaker);
    }
}
