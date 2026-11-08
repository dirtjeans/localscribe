using LocalScribe.Core.Pipeline;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// Correcting a word the transcriber got wrong: it must reach what it should and nothing else,
/// survive the transcript being rebuilt, and keep the word clickable at the right moment.
/// </summary>
public class WordCorrectionsTests
{
    private static TranscriptSegment Segment(string text, double start = 0, double end = 10, string? speaker = null) =>
        new(text, start, end, Speaker: speaker);

    private static string Text(WordCorrections corrections, params TranscriptSegment[] segments) =>
        string.Join(" | ", corrections.Apply(segments).Shown.Select(segment => segment.Text));

    private static WordCorrections Everywhere(string wrong, string right)
    {
        var corrections = new WordCorrections();
        corrections.Everywhere(wrong, right);
        return corrections;
    }

    private static List<WordTimings.Word> Timed(string text, double start = 0, double each = 1)
    {
        var tokens = text.Split(' ');
        return [.. tokens.Select((token, i) => new WordTimings.Word(token, start + (i * each), start + (i * each) + (each / 2)))];
    }

    [Fact]
    public void ACapitalisedNameLeavesTheOrdinaryWordAlone()
    {
        Assert.Equal(
            "Wil said he will come.",
            Text(Everywhere("Will", "Wil"), Segment("Will said he will come.")));
    }

    [Fact]
    public void ALowerCaseCorrectionKeepsTheCapitalAtTheStartOfASentence()
    {
        Assert.Equal(
            "Colour is fine and colour is cheap.",
            Text(Everywhere("color", "colour"), Segment("Color is fine and color is cheap.")));
    }

    [Fact]
    public void OnlyWholeWordsMatch()
    {
        Assert.Equal(
            "Rags Ragu's Ragged",
            Text(Everywhere("Rag", "Rug"), Segment("Rags Ragu's Ragged")));
    }

    [Fact]
    public void PunctuationAndPossessivesStayWhereTheyWere()
    {
        Assert.Equal(
            "(Raghu), Raghu's point, and \"Raghu's.\"",
            Text(Everywhere("Ragu", "Raghu"), Segment("(Ragu), Ragu's point, and \"Ragu's.\"")));
    }

    [Fact]
    public void TypingThePunctuationIntoTheCorrectionDoesNotDoubleIt()
    {
        Assert.Equal("Yes, Raghu, yes", Text(Everywhere("Ragu", "Raghu,"), Segment("Yes, Ragu, yes")));
    }

    [Fact]
    public void ANameCanChangeHowManyWordsItIs()
    {
        Assert.Equal(
            "Hello Raghu Nandakumara, welcome.",
            Text(Everywhere("Ragu Nanda Kumara", "Raghu Nandakumara"), Segment("Hello Ragu Nanda Kumara, welcome.")));
    }

    [Fact]
    public void ANameCutInTwoBySegmentsIsStillFound()
    {
        var corrected = Everywhere("Ragu Nanda", "Raghu").Apply(
            [Segment("Hello Ragu", 0, 5), Segment("Nanda, welcome", 5, 10)]);

        Assert.Equal(["Hello Raghu,", "welcome"], corrected.Shown.Select(segment => segment.Text));
    }

    [Fact]
    public void ASegmentThatWasOnlyTheEndOfANameLeavesNoEmptyRow()
    {
        var corrected = Everywhere("Ragu Nanda", "Raghu").Apply(
            [Segment("Hello Ragu", 0, 5), Segment("Nanda", 5, 6), Segment("welcome", 6, 10)]);

        Assert.Equal(["Hello Raghu", "welcome"], corrected.Shown.Select(segment => segment.Text));
    }

    [Fact]
    public void ACorrectionAppliedToEveryRebuildNeverGrows()
    {
        var corrections = Everywhere("Ken", "Ken Brown");
        var heard = Segment("Ken spoke.");

        // The caller hands over the transcript as heard each time, so three rebuilds are one result.
        corrections.Apply([heard]);
        corrections.Apply([heard]);

        Assert.Equal("Ken Brown spoke.", Text(corrections, heard));
    }

    [Fact]
    public void TheCorrectionReachesTextThatArrivesLater()
    {
        var corrections = Everywhere("Ragu", "Raghu");

        Assert.Equal("Raghu one", Text(corrections, Segment("Ragu one", 0, 5)));
        Assert.Equal(
            "Raghu one | Raghu two",
            Text(corrections, Segment("Ragu one", 0, 5), Segment("Ragu two", 5, 10)));
    }

    [Fact]
    public void CorrectingACorrectionWorks()
    {
        var corrections = Everywhere("Ragu", "Rahgu");
        corrections.Everywhere("Rahgu", "Raghu");

        Assert.Equal("Raghu", Text(corrections, Segment("Ragu")));
    }

    [Fact]
    public void HereChangesOnlyTheOccurrenceNearThatMoment()
    {
        var corrections = new WordCorrections();
        corrections.Here("Ragu", "Raghu", 42);

        Assert.Equal(
            "Ragu early | Raghu here | Ragu late",
            Text(corrections, Segment("Ragu early", 0, 5), Segment("Ragu here", 40, 45), Segment("Ragu late", 90, 95)));
    }

    [Fact]
    public void HereChoosesWithinASegmentByPosition()
    {
        var corrections = new WordCorrections();
        corrections.Here("Ragu", "Raghu", 9);

        Assert.Equal("Ragu and then Raghu", Text(corrections, Segment("Ragu and then Ragu", 0, 10)));
    }

    [Fact]
    public void HereSurvivesTheSegmentsBeingCutDifferently()
    {
        var corrections = new WordCorrections();
        corrections.Here("Ragu", "Raghu", 12);

        // As a 30-second streamed window, then as the finished transcript's shorter segments.
        Assert.Equal("Well Raghu spoke", Text(corrections, Segment("Well Ragu spoke", 0, 30)));
        Assert.Equal("Well | Raghu spoke", Text(corrections, Segment("Well", 0, 11), Segment("Ragu spoke", 11, 14)));
    }

    [Fact]
    public void HereWithNothingNearbyChangesNothing()
    {
        var corrections = new WordCorrections();
        corrections.Here("Ragu", "Raghu", 500);

        Assert.Equal("Ragu", Text(corrections, Segment("Ragu", 0, 10)));
    }

    [Fact]
    public void AnUnchangedSegmentIsTheSameRecordSoItsWordTimesStillFindIt()
    {
        var plain = Segment("Nothing to fix", 0, 5);
        var corrected = Everywhere("Ragu", "Raghu").Apply([plain, Segment("Ragu", 5, 10)]);

        Assert.Same(plain, corrected.Shown[0]);
    }

    [Fact]
    public void TheWayBackFindsTheSegmentAsHeard()
    {
        var heard = Segment("Ragu spoke");
        var corrected = Everywhere("Ragu", "Raghu").Apply([heard]);

        Assert.Same(heard, corrected.HeardOf(corrected.Shown[0]));
    }

    [Fact]
    public void WordsNotCorrectedKeepTheirExactTimes()
    {
        var heard = Segment("one Ragu three");
        var times = Timed(heard.Text);
        var corrected = Everywhere("Ragu", "Raghu").Apply([heard]);
        var words = corrected.Retimed(corrected.Shown[0], times);

        Assert.Equal(["one", "Raghu", "three"], words.Select(word => word.Text));

        for (var i = 0; i < 3; i++)
        {
            Assert.Equal(times[i].StartSeconds, words[i].StartSeconds, 6);
            Assert.Equal(times[i].EndSeconds, words[i].EndSeconds, 6);
        }
    }

    [Fact]
    public void FewerWordsCoverTheSameStretchOfTime()
    {
        var heard = Segment("hello Ragu Nanda Kumara welcome");
        var times = Timed(heard.Text);
        var corrected = Everywhere("Ragu Nanda Kumara", "Raghu Nandakumara").Apply([heard]);
        var words = corrected.Retimed(corrected.Shown[0], times);

        Assert.Equal(["hello", "Raghu", "Nandakumara", "welcome"], words.Select(word => word.Text));
        Assert.Equal(times[1].StartSeconds, words[1].StartSeconds, 6);
        Assert.Equal(times[3].EndSeconds, words[2].EndSeconds, 6);
        Assert.Equal(times[4].StartSeconds, words[3].StartSeconds, 6);
        Assert.True(words[2].StartSeconds > words[1].StartSeconds);
    }

    [Fact]
    public void MoreWordsShareTheTimeOfTheOneTheyReplace()
    {
        var heard = Segment("hello Ken welcome");
        var times = Timed(heard.Text);
        var corrected = Everywhere("Ken", "Ken Brown").Apply([heard]);
        var words = corrected.Retimed(corrected.Shown[0], times);

        Assert.Equal(times[1].StartSeconds, words[1].StartSeconds, 6);
        Assert.Equal(times[1].EndSeconds, words[2].EndSeconds, 6);
        Assert.True(words[1].EndSeconds <= words[2].StartSeconds + 1e-9);
        Assert.Equal(times[2].StartSeconds, words[3].StartSeconds, 6);
    }

    [Fact]
    public void WordsAreTimedInOrderAndOffsetsPointIntoTheirText()
    {
        var heard = Segment("a Ragu b");
        var corrected = Everywhere("Ragu", "Raghu Nandakumara").Apply([heard]);
        var shown = corrected.Shown[0];
        var words = corrected.Retimed(shown, Timed(heard.Text));

        foreach (var word in words)
        {
            Assert.Equal(word.Text, shown.Text.Substring(word.Offset, word.Text.Length));
        }

        Assert.True(words.Zip(words.Skip(1)).All(pair => pair.First.StartSeconds <= pair.Second.StartSeconds));
    }

    [Fact]
    public void TimesThatDoNotCorrespondFallBackToTheSegmentsSpan()
    {
        var heard = Segment("one Ragu three", 0, 9);
        var corrected = Everywhere("Ragu", "Raghu").Apply([heard]);
        var words = corrected.Retimed(corrected.Shown[0], Timed("just two"));

        Assert.Equal(3, words.Count);
        Assert.Equal(3, words[1].StartSeconds, 6);
        Assert.Equal(6, words[1].EndSeconds, 6);
    }

    [Theory]
    [InlineData("", "Raghu")]
    [InlineData("Ragu", "  ")]
    [InlineData("Ragu", "Ragu")]
    [InlineData("...", "Raghu")]
    public void NothingToChangeIsNotACorrection(string wrong, string right)
    {
        var corrections = new WordCorrections();

        Assert.False(corrections.Everywhere(wrong, right));
        Assert.True(corrections.IsEmpty);
    }

    [Fact]
    public void ClearForgetsEverything()
    {
        var corrections = Everywhere("Ragu", "Raghu");
        corrections.Clear();

        Assert.Equal("Ragu", Text(corrections, Segment("Ragu")));
    }

    [Fact]
    public void OccurrencesCountsWhatEverywhereWouldChange()
    {
        IReadOnlyList<TranscriptSegment> segments =
            [Segment("Ragu and Ragu's, not ragu"), Segment("then Ragu"), Segment("Nanda")];

        Assert.Equal(3, WordCorrections.Occurrences("Ragu", segments));
        Assert.Equal(1, WordCorrections.Occurrences("Ragu Nanda", segments));
        Assert.Equal(0, WordCorrections.Occurrences("Raghu", segments));
    }

    [Theory]
    [InlineData("Hello Ragu, welcome", 7, 7, 6, "Ragu")]     // A point in a word.
    [InlineData("Hello Ragu, welcome", 8, 9, 6, "Ragu")]     // A drag inside one.
    [InlineData("Hello Ragu Nanda, welcome", 8, 12, 6, "Ragu Nanda")]
    [InlineData("Hello (Ragu's), welcome", 8, 8, 7, "Ragu")]
    [InlineData("Hello", 5, 5, 0, "Hello")]                  // Past the last character.
    public void ASelectionBecomesTheWordsItTouches(string text, int from, int to, int start, string words)
    {
        Assert.Equal((start, words), WordCorrections.WordsAround(text, from, to));
    }

    [Fact]
    public void ASelectionOfNothingButPunctuationIsNotAWord()
    {
        Assert.Null(WordCorrections.WordsAround("Hello - there", 6, 7));
    }

    [Theory]
    [InlineData(0, "0:00")]
    [InlineData(38, "0:38")]
    [InlineData(59.6, "1:00")]
    [InlineData(252, "4:12")]
    [InlineData(3723, "1:02:03")]
    public void TimesReadAsAClockDoes(double seconds, string clock)
    {
        Assert.Equal(clock, RunTimes.Clock(seconds));
    }

    [Fact]
    public void TheSummaryNamesEachStepThatDidAnything()
    {
        var text = RunTimes.Describe(
            252,
            443,
            [("Raw transcript", 65), ("Figuring out who said what", 0.1), ("Cleanup", 170), ("Never started", null)]);

        Assert.Equal("Done in 4:12 for 7:23 of audio. Raw transcript 1:05 · Cleanup 2:50.", text);
    }

    [Fact]
    public void WithNoStepsTheHeadlineStandsAlone()
    {
        Assert.Equal("Done in 0:40.", RunTimes.Describe(40, 0, []));
    }
}
