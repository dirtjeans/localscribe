using LocalScribe.Core.Diarization;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class SpeakerNamesTests
{
    private static TranscriptSegment Said(string speaker, double start, double end) =>
        new("words", start, end, Speaker: speaker);

    /// <summary>
    /// The point of the class: a name given mid-run survives the next rebuild, which arrives
    /// with fresh segment records and the diarizer's own labels on them again.
    /// </summary>
    [Fact]
    public void ANameSurvivesTheNextLabelling()
    {
        var names = new SpeakerNames();

        names.Apply(
            [Said("voice-7", 0, 2), Said("voice-3", 2, 4)],
            [Said("Speaker 1", 0, 2), Said("Speaker 2", 2, 4)]);

        Assert.True(names.NameVoice("Speaker 2", "Ada"));

        var later = names.Apply(
            [Said("voice-7", 0, 2), Said("voice-3", 2, 4), Said("voice-3", 4, 6)],
            [Said("Speaker 1", 0, 2), Said("Speaker 2", 2, 4), Said("Speaker 2", 4, 6)]);

        Assert.Equal(["Speaker 1", "Ada", "Ada"], later.Select(s => s.Speaker));
    }

    /// <summary>
    /// Keyed by the voice, not the number on screen. Renumbering by appearance can hand a
    /// voice a different number when the final assembly cuts the opening differently, and a
    /// name keyed by the number would jump to the wrong person.
    /// </summary>
    [Fact]
    public void TheNameFollowsTheVoiceWhenTheNumbersChange()
    {
        var names = new SpeakerNames();

        names.Apply(
            [Said("voice-7", 0, 2), Said("voice-3", 2, 4)],
            [Said("Speaker 1", 0, 2), Said("Speaker 2", 2, 4)]);
        names.NameVoice("Speaker 2", "Ada");

        var final = names.Apply(
            [Said("voice-3", 0, 1), Said("voice-7", 1, 4)],
            [Said("Speaker 1", 0, 1), Said("Speaker 2", 1, 4)]);

        Assert.Equal(["Ada", "Speaker 2"], final.Select(s => s.Speaker));
    }

    /// <summary>A part rename is kept against its time, and beats the voice-wide name there.</summary>
    [Fact]
    public void APartKeepsItsOwnName()
    {
        var names = new SpeakerNames();

        names.Apply([Said("v1", 0, 2), Said("v1", 2, 4)], [Said("Speaker 1", 0, 2), Said("Speaker 1", 2, 4)]);
        names.NameVoice("Speaker 1", "Ada");
        names.NamePart(2, 4, "Grace");

        var relabelled = names.Apply(
            [Said("v1", 0, 2), Said("v1", 2, 4)],
            [Said("Speaker 1", 0, 2), Said("Speaker 1", 2, 4)]);

        Assert.Equal(["Ada", "Grace"], relabelled.Select(s => s.Speaker));
    }

    /// <summary>
    /// Renaming a name works too: the name on screen still leads back to its voice.
    /// </summary>
    [Fact]
    public void ANamedSpeakerCanBeRenamedAgain()
    {
        var names = new SpeakerNames();

        names.Apply([Said("v1", 0, 2)], [Said("Speaker 1", 0, 2)]);
        names.NameVoice("Speaker 1", "Ada");
        names.Apply([Said("v1", 0, 2)], [Said("Speaker 1", 0, 2)]);

        Assert.True(names.NameVoice("Ada", "Ada Lovelace"));
        Assert.Equal(
            ["Ada Lovelace"],
            names.Apply([Said("v1", 0, 2)], [Said("Speaker 1", 0, 2)]).Select(s => s.Speaker));
    }

    /// <summary>
    /// Wrongly named is worse than unnamed. When the two lists do not line up, no segment's
    /// voice can be known, so nothing is renamed.
    /// </summary>
    [Fact]
    public void MismatchedListsNameNobody()
    {
        var names = new SpeakerNames();

        names.Apply([Said("v1", 0, 2)], [Said("Speaker 1", 0, 2)]);
        names.NameVoice("Speaker 1", "Ada");

        var result = names.Apply([Said("v1", 0, 2)], [Said("Speaker 1", 0, 1), Said("Speaker 1", 1, 2)]);

        Assert.All(result, s => Assert.Equal("Speaker 1", s.Speaker));
        Assert.False(names.NameVoice("Speaker 1", "Grace"));
    }
}
