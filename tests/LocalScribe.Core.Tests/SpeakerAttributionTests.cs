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
}
