using LocalScribe.Core.Diarization;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class SpeakerAttributionTests
{
    private static TranscriptSegment Segment(string text, double start, double end) =>
        new(text, start, end);

    [Fact]
    public void WithNoTurnsTheTranscriptIsUnchanged()
    {
        var segments = new[] { Segment("Hello there.", 0, 2) };

        Assert.Same(segments, SpeakerAttribution.Apply(segments, []));
    }

    [Fact]
    public void ASegmentInsideOneTurnIsLabelled()
    {
        var result = SpeakerAttribution.Apply(
            [Segment("Hello there.", 1, 3)],
            [new SpeakerTurn(0, 0, 10)]);

        Assert.Single(result);
        Assert.Equal("Speaker 1", result[0].Speaker);
    }

    /// <summary>
    /// The failure this exists to fix. Whisper returns a quick exchange as one segment, and
    /// labelling it whole hands four turns to one person — a perfectly diarized recording
    /// reading as somebody talking to themselves.
    /// </summary>
    [Fact]
    public void ASegmentSpanningTurnsIsSplitAtSentences()
    {
        var result = SpeakerAttribution.Apply(
            [Segment("Did you get the report? Yes, this morning.", 0, 4)],
            [
                new SpeakerTurn(0, 0, 2),
                new SpeakerTurn(1, 2, 4),
            ]);

        Assert.Equal(2, result.Count);
        Assert.Equal("Did you get the report?", result[0].Text);
        Assert.Equal("Speaker 1", result[0].Speaker);
        Assert.Equal("Yes, this morning.", result[1].Text);
        Assert.Equal("Speaker 2", result[1].Speaker);
    }

    [Fact]
    public void TheSplitPiecesCarryTheirOwnTimes()
    {
        var result = SpeakerAttribution.Apply(
            [Segment("One two three. Four five six.", 0, 6)],
            [
                new SpeakerTurn(0, 0, 3),
                new SpeakerTurn(1, 3, 6),
            ]);

        Assert.Equal(0, result[0].StartSeconds);
        Assert.True(result[1].EndSeconds <= 6.0001);
        Assert.True(result[0].EndSeconds <= result[1].StartSeconds + 0.0001);
    }

    /// <summary>
    /// A transcript with one line per sentence is as hard to read as one with none, so
    /// consecutive sentences from the same speaker are rejoined.
    /// </summary>
    [Fact]
    public void ConsecutiveSentencesFromOneSpeakerStayTogether()
    {
        var result = SpeakerAttribution.Apply(
            [Segment("One. Two. Three. Four.", 0, 8)],
            [
                new SpeakerTurn(0, 0, 6),
                new SpeakerTurn(1, 6, 8),
            ]);

        Assert.Equal(2, result.Count);
        Assert.Equal("One. Two. Three.", result[0].Text);
        Assert.Equal("Four.", result[1].Text);
    }

    /// <summary>
    /// Nothing to cut along. One span given to whoever spoke most of it beats a sentence chopped
    /// in half at an arbitrary character.
    /// </summary>
    [Fact]
    public void ASingleSentenceIsNotCutUp()
    {
        var result = SpeakerAttribution.Apply(
            [Segment("a continuous run of speech with no full stop", 0, 4)],
            [
                new SpeakerTurn(0, 0, 3),
                new SpeakerTurn(1, 3, 4),
            ]);

        Assert.Single(result);
        Assert.Equal("Speaker 1", result[0].Speaker);
    }

    [Fact]
    public void QuestionsAndExclamationsEndSentencesToo()
    {
        var result = SpeakerAttribution.Apply(
            [Segment("Really? Yes! Fine.", 0, 6)],
            [
                new SpeakerTurn(0, 0, 2),
                new SpeakerTurn(1, 2, 4),
                new SpeakerTurn(0, 4, 6),
            ]);

        Assert.Equal(3, result.Count);
    }

    [Fact]
    public void ASegmentOutsideEveryTurnKeepsNoSpeaker()
    {
        var result = SpeakerAttribution.Apply(
            [Segment("Off on its own.", 20, 22)],
            [new SpeakerTurn(0, 0, 5)]);

        Assert.Null(result[0].Speaker);
    }

    [Fact]
    public void NoTextIsLostInTheSplit()
    {
        const string text = "First one. Second one. Third one. Fourth one.";

        var result = SpeakerAttribution.Apply(
            [Segment(text, 0, 8)],
            [
                new SpeakerTurn(0, 0, 2),
                new SpeakerTurn(1, 2, 4),
                new SpeakerTurn(0, 4, 6),
                new SpeakerTurn(1, 6, 8),
            ]);

        Assert.Equal(text, string.Join(" ", result.Select(r => r.Text)));
    }

    /// <summary>
    /// The interruption from the debate recording. The diarizer put the change at 16.6s; Whisper
    /// ended a segment at 16.0s, where "Not in the least." begins. Left alone, the segment
    /// straddles the boundary, goes to whichever side holds more of it, and the interruption is
    /// credited to the person being interrupted.
    /// </summary>
    [Fact]
    public void AnInterruptionGoesToWhoeverInterrupted()
    {
        IReadOnlyList<TranscriptSegment> segments =
        [
            Segment("I am using the term God in belief, and then you--", 14.2, 16.0),
            Segment("Not in the least.", 16.0, 16.9),
            Segment("I don't understand how you're using it in the least.", 16.9, 19.3),
        ];

        IReadOnlyList<SpeakerTurn> turns =
        [
            new(1, 8.6, 16.6),
            new(0, 16.6, 28.9),
        ];

        var result = SpeakerAttribution.Apply(segments, turns);

        var interruption = result.Single(s => s.Text.StartsWith("Not in the least", StringComparison.Ordinal));

        Assert.Equal("Speaker 1", interruption.Speaker);
    }

    /// <summary>A boundary in open ground is the diarizer's to place, and stays put.</summary>
    [Fact]
    public void ABoundaryWithNoSegmentGapNearbyIsLeftAlone()
    {
        IReadOnlyList<TranscriptSegment> segments =
        [
            Segment("A long uninterrupted stretch of one person talking", 0, 20),
            Segment("and then somebody else entirely", 20, 30),
        ];

        IReadOnlyList<SpeakerTurn> turns = [new(0, 0, 10), new(1, 10, 30)];

        var moved = SpeakerAttribution.SnapToSegmentBoundaries(turns, segments);

        Assert.Equal(10, moved[1].StartSeconds, 3);
    }

    [Fact]
    public void SnappingLeavesNoGapOrOverlapBetweenTurns()
    {
        IReadOnlyList<TranscriptSegment> segments =
        [
            Segment("one", 0, 5.0),
            Segment("two", 5.0, 10),
            Segment("three", 10, 15),
        ];

        IReadOnlyList<SpeakerTurn> turns = [new(0, 0, 5.4), new(1, 5.4, 10.2), new(0, 10.2, 15)];

        var moved = SpeakerAttribution.SnapToSegmentBoundaries(turns, segments);

        for (var i = 1; i < moved.Count; i++)
        {
            Assert.Equal(moved[i - 1].EndSeconds, moved[i].StartSeconds, 3);
        }
    }

    [Fact]
    public void NothingToSnapToIsNotAFailure()
    {
        IReadOnlyList<SpeakerTurn> turns = [new(0, 0, 5), new(1, 5, 10)];

        Assert.Equal(turns, SpeakerAttribution.SnapToSegmentBoundaries(turns, []));
    }
}
