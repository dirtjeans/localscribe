using LocalScribe.Core.Diarization;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class CrosstalkMarksTests
{
    private static TimedSegment Piece(string speaker, params (double From, double To)[] words) =>
        new(
            new TranscriptSegment(
                string.Join(' ', words.Select((_, i) => $"w{i}")),
                words[0].From,
                words[^1].To,
                Speaker: speaker),
            [.. words.Select((w, i) => new WordTimings.Word($"w{i}", w.From, w.To))]);

    [Fact]
    public void TwoVoicesSoundingTogetherAreBothMarked()
    {
        var marked = CrosstalkMarks.Apply(
        [
            Piece("A", (10.0, 11.0), (11.0, 12.0), (12.0, 13.0)),
            Piece("B", (11.5, 12.5), (12.5, 13.5)),
        ]);

        Assert.True(marked[0].Segment.Overlapped);
        Assert.True(marked[1].Segment.Overlapped);
    }

    [Fact]
    public void ABoundaryBrushIsNotCrosstalk()
    {
        // A quarter-second graze at the handover between turns: the aligner works in
        // twenty-millisecond frames, and this is how a clean exchange routinely measures.
        var marked = CrosstalkMarks.Apply(
        [
            Piece("A", (10.0, 12.0)),
            Piece("B", (11.75, 14.0)),
        ]);

        Assert.False(marked[0].Segment.Overlapped);
        Assert.False(marked[1].Segment.Overlapped);
    }

    [Fact]
    public void TheSameVoiceCannotCrosstalkWithItself()
    {
        // A self-restart or an aside places two of one speaker's segments over shared time.
        // That is one person, not a contest.
        var marked = CrosstalkMarks.Apply(
        [
            Piece("A", (10.0, 13.0)),
            Piece("A", (12.0, 14.0)),
        ]);

        Assert.False(marked[0].Segment.Overlapped);
        Assert.False(marked[1].Segment.Overlapped);
    }

    [Fact]
    public void SilenceWhileTheOtherTalksIsNotCrosstalk()
    {
        // One speaker's words bracket a long gap the other speaks into. The envelope overlaps
        // for seconds; the voices never do.
        var marked = CrosstalkMarks.Apply(
        [
            Piece("A", (10.0, 10.5), (16.0, 16.5)),
            Piece("B", (11.0, 15.5)),
        ]);

        Assert.False(marked[0].Segment.Overlapped);
        Assert.False(marked[1].Segment.Overlapped);
    }

    [Fact]
    public void MarkingChangesNothingButTheFlag()
    {
        var input = new[]
        {
            Piece("A", (10.0, 12.0)),
            Piece("B", (10.5, 12.5)),
        };

        var marked = CrosstalkMarks.Apply(input);

        Assert.Equal(input.Length, marked.Count);
        Assert.Equal(input[0].Segment.Text, marked[0].Segment.Text);
        Assert.Equal(input[0].Segment.StartSeconds, marked[0].Segment.StartSeconds);
        Assert.Equal(input[0].Segment.EndSeconds, marked[0].Segment.EndSeconds);
        Assert.Same(input[0].Words, marked[0].Words);
    }
}
