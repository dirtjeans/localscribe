using LocalScribe.Core.Diarization;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// Dividing a segment between speakers must not lose any of it.
/// <para>
/// Cutting is done by slicing the original text between word offsets, which keeps the
/// punctuation the reader expects and makes losing a word entirely possible: a piece that comes
/// out empty, or offsets that do not reach the end of the text, and the words are simply gone
/// from the transcript with nothing to say they were.
/// </para>
/// </summary>
public class NothingIsDroppedTests
{
    private static TimedSegment Spoken(string text, double from = 0, double step = 1)
    {
        var words = new List<WordTimings.Word>();
        var at = from;
        var offset = 0;

        foreach (var word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            offset = text.IndexOf(word, offset, StringComparison.Ordinal);
            words.Add(new WordTimings.Word(word, at, at + (step * 0.9)) { Offset = offset });
            offset += word.Length;
            at += step;
        }

        return new TimedSegment(new TranscriptSegment(text, from, at), words);
    }

    private static string Words(string text) =>
        string.Join(" ", text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

    [Theory]
    [InlineData("one two three four five six")]
    [InlineData("Well, obviously not. That's absurd.")]
    [InlineData("I do think there are lots of interesting ways to define God, actually.")]
    [InlineData("No. No! Absolutely not — and you know it.")]
    [InlineData("a")]
    [InlineData("two words")]
    [InlineData("Ends with an ellipsis…")]
    [InlineData("Multiple   spaces   between   words")]
    public void EveryWordSurvivesBeingDivided(string text)
    {
        var timed = Spoken(text);
        var turns = new List<SpeakerTurn>();

        // A change every two seconds, so most of these get cut more than once.
        for (var i = 0; i < 20; i += 2)
        {
            turns.Add(new SpeakerTurn(i / 2 % 2, i, i + 2));
        }

        var pieces = WordLevelAttribution.Apply([timed], turns);

        Assert.Equal(Words(text), Words(string.Join(" ", pieces.Select(p => p.Segment.Text))));
    }

    /// <summary>Each piece's offsets must address its own text, or the marker lands elsewhere.</summary>
    [Theory]
    [InlineData("one two three four five six")]
    [InlineData("Well, obviously not. That's absurd.")]
    [InlineData("Multiple   spaces   between   words")]
    public void EveryWordCanStillBeFoundInItsOwnPiece(string text)
    {
        var pieces = WordLevelAttribution.Apply(
            [Spoken(text)],
            [new SpeakerTurn(0, 0, 2), new SpeakerTurn(1, 2, 4), new SpeakerTurn(0, 4, 20)]);

        foreach (var piece in pieces)
        {
            foreach (var word in piece.Words)
            {
                Assert.InRange(word.Offset, 0, piece.Segment.Text.Length - 1);

                Assert.Equal(
                    word.Text,
                    piece.Segment.Text.Substring(word.Offset, word.Text.Length));
            }
        }
    }

    /// <summary>
    /// The word list and the text disagreeing is the shape that truncates silently. Cutting to
    /// the next word's offset rather than to the end of this word's text makes the pieces tile
    /// whatever is between them, so a wrong length costs a marker position and never a word.
    /// </summary>
    [Fact]
    public void AWordWhoseTextDoesNotMatchTheTextCannotTruncateThePiece()
    {
        const string said = "one two three four";

        IReadOnlyList<WordTimings.Word> words =
        [
            new WordTimings.Word("on", 0, 0.9) { Offset = 0 },       // a character short
            new WordTimings.Word("two", 1, 1.9) { Offset = 4 },
            new WordTimings.Word("thre", 2, 2.9) { Offset = 8 },     // and another
            new WordTimings.Word("four", 3, 3.9) { Offset = 14 },
        ];

        var pieces = WordLevelAttribution.Apply(
            [new TimedSegment(new TranscriptSegment(said, 0, 4), words)],
            [new SpeakerTurn(0, 0, 2), new SpeakerTurn(1, 2, 4)]);

        Assert.Equal(Words(said), Words(string.Join(" ", pieces.Select(p => p.Segment.Text))));
    }

    /// <summary>
    /// A division that cannot account for every word leaves the segment whole rather than
    /// publishing a transcript with words missing from it.
    /// </summary>
    [Fact]
    public void AnImpossibleDivisionLeavesTheSegmentWhole()
    {
        const string said = "one two three";

        // Offsets past the end of the text: no piece can be sliced out of this.
        IReadOnlyList<WordTimings.Word> words =
        [
            new WordTimings.Word("one", 0, 0.9) { Offset = 40 },
            new WordTimings.Word("two", 1, 1.9) { Offset = 44 },
            new WordTimings.Word("three", 2, 2.9) { Offset = 48 },
        ];

        var pieces = WordLevelAttribution.Apply(
            [new TimedSegment(new TranscriptSegment(said, 0, 3), words)],
            [new SpeakerTurn(0, 0, 1.5), new SpeakerTurn(1, 1.5, 3)]);

        Assert.Single(pieces);
        Assert.Equal(said, pieces[0].Segment.Text);
    }
}
