using LocalScribe.Core.Diarization;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class UnfinishedSentencesTests
{
    private static TimedSegment Said(string speaker, string text, double from)
    {
        var words = new List<WordTimings.Word>();
        var at = from;
        var offset = 0;

        foreach (var word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            offset = text.IndexOf(word, offset, StringComparison.Ordinal);
            words.Add(new WordTimings.Word(word, at, at + 0.4) { Offset = offset });
            offset += word.Length;
            at += 0.5;
        }

        return new TimedSegment(
            new TranscriptSegment(text, from, at) { Speaker = speaker }, words);
    }

    private static string Words(string text) =>
        string.Join(" ", text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

    /// <summary>
    /// From the recording: only the first word of the next segment finishes the sentence, and
    /// the rest of it is a real turn that must survive.
    /// </summary>
    [Fact]
    public void OnlyTheWordsThatFinishTheSentenceMoveBack()
    {
        var pieces = UnfinishedSentences.Apply(
        [
            Said("Speaker 2", "you're kind of expanding the meaning of", 20),
            Said("Speaker 1", "God. No, I'm not.", 24),
        ]);

        Assert.Equal(2, pieces.Count);
        Assert.Equal("you're kind of expanding the meaning of God.", pieces[0].Segment.Text);
        Assert.Equal("Speaker 2", pieces[0].Segment.Speaker);
        Assert.Equal("No, I'm not.", pieces[1].Segment.Text);
        Assert.Equal("Speaker 1", pieces[1].Segment.Speaker);
    }

    /// <summary>
    /// And the other one, where all of it moves back. "characteristics" closes nothing — the
    /// sentence runs on past it — so requiring a tail to end a sentence would miss this.
    /// </summary>
    [Fact]
    public void AWholeFragmentCanMoveBack()
    {
        var pieces = UnfinishedSentences.Apply(
        [
            Said("Speaker 1", "and conscience is one of the defining", 49),
            Said("Speaker 2", "characteristics", 53),
        ]);

        Assert.Single(pieces);
        Assert.Equal("and conscience is one of the defining characteristics", pieces[0].Segment.Text);
        Assert.Equal("Speaker 1", pieces[0].Segment.Speaker);
    }

    [Fact]
    public void AFinishedSentenceIsLeftAlone()
    {
        var pieces = UnfinishedSentences.Apply(
        [
            Said("Speaker 1", "I think you're being disingenuous.", 10),
            Said("Speaker 2", "In what way?", 12),
        ]);

        Assert.Equal(2, pieces.Count);
        Assert.Equal("Speaker 2", pieces[1].Segment.Speaker);
    }

    /// <summary>A real turn beginning mid-sentence is longer than the end of a clause.</summary>
    [Fact]
    public void ALongTurnIsNotAFragment()
    {
        var pieces = UnfinishedSentences.Apply(
        [
            Said("Speaker 1", "and I was going to say that", 10),
            Said("Speaker 2", "no you were not going to say anything of the sort", 12),
        ]);

        Assert.Equal(2, pieces.Count);
        Assert.Equal("Speaker 2", pieces[1].Segment.Speaker);
    }

    /// <summary>Across a pause, the second speaker was starting rather than finishing.</summary>
    [Fact]
    public void APauseMeansTheyWereStartingNotFinishing()
    {
        var pieces = UnfinishedSentences.Apply(
        [
            Said("Speaker 1", "and conscience is one of the defining", 10),
            Said("Speaker 2", "characteristics", 20),
        ]);

        Assert.Equal(2, pieces.Count);
    }

    [Fact]
    public void TheSameSpeakerTwiceIsNotTouched()
    {
        var pieces = UnfinishedSentences.Apply(
        [
            Said("Speaker 1", "one of the defining", 10),
            Said("Speaker 1", "characteristics", 12),
        ]);

        Assert.Equal(2, pieces.Count);
    }

    /// <summary>Moving words must not lose or duplicate any of them.</summary>
    [Theory]
    [InlineData("the meaning of", "God. No, I'm not.")]
    [InlineData("one of the defining", "characteristics")]
    [InlineData("I said that", "and then. Well.")]
    public void EveryWordSurvivesTheMove(string first, string second)
    {
        var pieces = UnfinishedSentences.Apply(
            [Said("Speaker 1", first, 10), Said("Speaker 2", second, 12)]);

        Assert.Equal(
            Words($"{first} {second}"),
            Words(string.Join(" ", pieces.Select(p => p.Segment.Text))));
    }

    /// <summary>Offsets must address the text each word now sits in.</summary>
    [Fact]
    public void OffsetsFollowTheWordsToTheirNewSegment()
    {
        var pieces = UnfinishedSentences.Apply(
        [
            Said("Speaker 2", "the meaning of", 20),
            Said("Speaker 1", "God. No, I'm not.", 21),
        ]);

        foreach (var piece in pieces)
        {
            foreach (var word in piece.Words)
            {
                Assert.Equal(
                    word.Text,
                    piece.Segment.Text.Substring(word.Offset, word.Text.Length));
            }
        }
    }
}
