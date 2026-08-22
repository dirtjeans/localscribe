using LocalScribe.Core.Diarization;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class KeepSentencesWholeTests
{
    private static TranscriptSegment Said(string speaker, string text, double from, double to) =>
        new(text, from, to) { Speaker = speaker };

    private static List<string?> Speakers(IReadOnlyList<TranscriptSegment> segments) =>
        [.. segments.Select(s => s.Speaker)];

    /// <summary>
    /// The case from the debate recording: one sentence returned as three segments and handed to
    /// two speakers in turn.
    /// </summary>
    [Fact]
    public void ASentenceSplitAcrossSpeakersGoesBackToOne()
    {
        var attributed = SpeakerAttribution.KeepSentencesWhole(
        [
            Said("Speaker 1", "You accept conscience as a guide,", 51, 60),
            Said("Speaker 2", "and conscience is one of the defining characteristics", 60, 63),
            Said("Speaker 1", "of God in the Old Testament.", 63, 64),
        ]);

        Assert.Equal(["Speaker 1", "Speaker 1", "Speaker 1"], Speakers(attributed));
    }

    /// <summary>
    /// What a minimum turn length would destroy and this keeps. A second-long reply is the
    /// shortest real exchange there is, and the full stop and capital say it happened.
    /// </summary>
    [Fact]
    public void AQuickExchangeSurvives()
    {
        var attributed = SpeakerAttribution.KeepSentencesWhole(
        [
            Said("Speaker 2", "I think you're being intellectually disingenuous.", 64, 66),
            Said("Speaker 1", "In what way?", 66, 67),
            Said("Speaker 2", "Because you know better.", 67, 69),
        ]);

        Assert.Equal(["Speaker 2", "Speaker 1", "Speaker 2"], Speakers(attributed));
    }

    /// <summary>A real interruption is a new thought, and starts like one.</summary>
    [Fact]
    public void AnInterruptionStartingItsOwnSentenceIsBelieved()
    {
        var attributed = SpeakerAttribution.KeepSentencesWhole(
        [
            Said("Speaker 1", "The point I was trying to make is that", 10, 12),
            Said("Speaker 2", "No, that isn't what you said.", 12, 14),
        ]);

        Assert.Equal(["Speaker 1", "Speaker 2"], Speakers(attributed));
    }

    /// <summary>Across a long silence, "and" is as likely to be somebody starting a sentence.</summary>
    [Fact]
    public void ASpeakerChangeAcrossAPauseIsLeftAlone()
    {
        var attributed = SpeakerAttribution.KeepSentencesWhole(
        [
            Said("Speaker 1", "so that was the whole argument,", 10, 12),
            Said("Speaker 2", "and I never agreed with it.", 20, 23),
        ]);

        Assert.Equal(["Speaker 1", "Speaker 2"], Speakers(attributed));
    }

    /// <summary>A question mark ends a sentence as firmly as a full stop.</summary>
    [Theory]
    [InlineData("Do you actually believe that?")]
    [InlineData("That is absurd!")]
    [InlineData("I couldn't hear the rest…")]
    [InlineData("He said it plainly:")]
    public void AnyRealTerminatorEndsTheSentence(string first)
    {
        var attributed = SpeakerAttribution.KeepSentencesWhole(
        [
            Said("Speaker 1", first, 10, 12),
            Said("Speaker 2", "well that depends.", 12, 14),
        ]);

        Assert.Equal(["Speaker 1", "Speaker 2"], Speakers(attributed));
    }

    /// <summary>A closing quote does not stop a full stop from being one.</summary>
    [Fact]
    public void AQuotedSentenceStillEnds()
    {
        var attributed = SpeakerAttribution.KeepSentencesWhole(
        [
            Said("Speaker 1", "He called it \"a category error.\"", 10, 12),
            Said("Speaker 2", "which it plainly is not.", 12, 14),
        ]);

        Assert.Equal(["Speaker 1", "Speaker 2"], Speakers(attributed));
    }

    /// <summary>Segments already agreeing are left exactly as they are.</summary>
    [Fact]
    public void NothingChangesWhenTheSpeakerDoesNot()
    {
        IReadOnlyList<TranscriptSegment> spoken =
        [
            Said("Speaker 1", "and so it follows that", 10, 12),
            Said("Speaker 1", "the whole thing collapses.", 12, 14),
        ];

        Assert.Equal(["Speaker 1", "Speaker 1"], Speakers(SpeakerAttribution.KeepSentencesWhole(spoken)));
    }

    /// <summary>Unlabelled segments are not given labels by this.</summary>
    [Fact]
    public void SegmentsWithNoSpeakerAreLeftAlone()
    {
        var attributed = SpeakerAttribution.KeepSentencesWhole(
        [
            new TranscriptSegment("so that was the point,", 10, 12),
            new TranscriptSegment("which nobody disputed.", 12, 14),
        ]);

        Assert.Equal([null, null], Speakers(attributed));
    }

    [Fact]
    public void OneSegmentIsNotAFailure() =>
        Assert.Single(SpeakerAttribution.KeepSentencesWhole([Said("Speaker 1", "Alone.", 0, 1)]));

    [Fact]
    public void NothingAtAllIsNotAFailure() =>
        Assert.Empty(SpeakerAttribution.KeepSentencesWhole([]));
}
