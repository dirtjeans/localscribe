using LocalScribe.Core.Diarization;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class SpeakerLabelsTests
{
    private static TranscriptSegment Line(double at, string? speaker) =>
        new("words", at, at + 1, Speaker: speaker);

    /// <summary>
    /// The reason this class exists: cluster order is not speaking order, and a transcript
    /// that opens with "Speaker 2" reads as a bug even when the separation is right.
    /// </summary>
    [Fact]
    public void TheFirstVoiceHeardBecomesSpeakerOne()
    {
        var relabelled = SpeakerLabels.RenumberByAppearance(
        [
            Line(0, "Speaker 2"),
            Line(1, "Speaker 1"),
            Line(2, "Speaker 2"),
        ]);

        Assert.Equal("Speaker 1", relabelled[0].Speaker);
        Assert.Equal("Speaker 2", relabelled[1].Speaker);
        Assert.Equal("Speaker 1", relabelled[2].Speaker);
    }

    /// <summary>
    /// A name the user gave means every number left is theirs too: renumbering around a
    /// rename could collide with a number the user is still using to mean someone specific.
    /// </summary>
    [Fact]
    public void AnyRenamedSpeakerFreezesTheNumbering()
    {
        var segments = new[]
        {
            Line(0, "Speaker 3"),
            Line(1, "Kim"),
            Line(2, "Speaker 3"),
        };

        var relabelled = SpeakerLabels.RenumberByAppearance(segments);

        Assert.Same(segments[0], relabelled[0]);
        Assert.Equal("Speaker 3", relabelled[0].Speaker);
    }

    /// <summary>Labels already in speaking order come back as the same list, not a copy.</summary>
    [Fact]
    public void AnAlreadyOrderedTranscriptIsLeftAlone()
    {
        var segments = new[] { Line(0, "Speaker 1"), Line(1, "Speaker 2") };

        Assert.Same(segments, SpeakerLabels.RenumberByAppearance(segments));
    }

    /// <summary>Unlabelled lines are carried through, not counted as a speaker.</summary>
    [Fact]
    public void UnlabelledLinesNeitherCountNorChange()
    {
        var relabelled = SpeakerLabels.RenumberByAppearance(
        [
            Line(0, null),
            Line(1, "Speaker 5"),
            Line(2, "Speaker 4"),
        ]);

        Assert.Null(relabelled[0].Speaker);
        Assert.Equal("Speaker 1", relabelled[1].Speaker);
        Assert.Equal("Speaker 2", relabelled[2].Speaker);
    }
}
