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
    public void ALineSpokenThroughContestedTimeIsMarked()
    {
        var marked = CrosstalkMarks.Apply(
            [Piece("A", (10.0, 11.0), (11.0, 12.0), (12.0, 13.0))],
            [(11.0, 12.5)]);

        Assert.True(marked[0].Segment.Overlapped);
    }

    [Fact]
    public void ABoundaryBrushIsNotCrosstalk()
    {
        // A clean handover grazes the overlap classes for a couple of tenths of a second.
        var marked = CrosstalkMarks.Apply(
            [Piece("A", (10.0, 12.0))],
            [(11.8, 12.1)]);

        Assert.False(marked[0].Segment.Overlapped);
    }

    [Fact]
    public void SilenceUnderTheOtherVoiceIsNotCrosstalk()
    {
        // The speaker's words bracket a gap the contested stretch falls into: somebody else
        // was talking, but not over these words.
        var marked = CrosstalkMarks.Apply(
            [Piece("A", (10.0, 10.5), (16.0, 16.5))],
            [(11.0, 15.5)]);

        Assert.False(marked[0].Segment.Overlapped);
    }

    [Fact]
    public void ContestedMomentsAddUpAcrossOneLine()
    {
        // Three half-second interruptions across a line: no single stretch is noticeable, but
        // the line was fought over throughout, and the reader should know.
        var marked = CrosstalkMarks.Apply(
            [Piece("A", (10.0, 16.0))],
            [(10.5, 11.0), (12.5, 13.0), (14.5, 15.0)]);

        Assert.True(marked[0].Segment.Overlapped);
    }

    [Fact]
    public void ALineWithoutMeasuredWordsFallsBackToItsBounds()
    {
        var bare = new TimedSegment(
            new TranscriptSegment("never aligned", 10.0, 14.0, Speaker: "A"), []);

        var marked = CrosstalkMarks.Apply([bare], [(11.0, 13.0)]);

        Assert.True(marked[0].Segment.Overlapped);
    }

    [Fact]
    public void MarkingChangesNothingButTheFlag()
    {
        var input = new[] { Piece("A", (10.0, 12.0)) };

        var marked = CrosstalkMarks.Apply(input, [(10.0, 12.0)]);

        Assert.True(marked[0].Segment.Overlapped);
        Assert.Equal(input[0].Segment.Text, marked[0].Segment.Text);
        Assert.Equal(input[0].Segment.StartSeconds, marked[0].Segment.StartSeconds);
        Assert.Equal(input[0].Segment.EndSeconds, marked[0].Segment.EndSeconds);
        Assert.Same(input[0].Words, marked[0].Words);
    }
}
