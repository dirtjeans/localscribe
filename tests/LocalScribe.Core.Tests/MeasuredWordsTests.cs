using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class MeasuredWordsTests
{
    /// <summary>The segment's own words, as the caller splits them.</summary>
    private static IReadOnlyList<WordTimings.Word> Own(params string[] words) =>
        [.. words.Select(w => new WordTimings.Word(w, 0, 0))];

    private static IReadOnlyList<WordTimings.Word> Placed(double from, double step, params string[] words) =>
        [.. words.Select((w, i) => new WordTimings.Word(w, from + (i * step), from + ((i + 1) * step)))];

    [Fact]
    public void MeasuredTimesReplaceTheEstimate()
    {
        var paired = MeasuredWords.Pair(Placed(0, 1, "one", "two", "three"), Own("one", "two", "three"));

        Assert.NotNull(paired);
        Assert.Equal(3, paired.Count);
        Assert.Equal(1, paired[1].StartSeconds, 3);
    }

    /// <summary>The text is the reader's; only the times are borrowed.</summary>
    [Fact]
    public void TheSegmentsOwnWordsAreKept()
    {
        var paired = MeasuredWords.Pair(Placed(0, 1, "dont", "stop"), Own("Don't", "stop."));

        Assert.NotNull(paired);
        Assert.Equal("Don't", paired[0].Text);
        Assert.Equal("stop.", paired[1].Text);
    }

    /// <summary>
    /// Where the words come from a different reading of the same speech, they cannot be paired at
    /// all. Refusing is the point: a segment timed against the wrong words is worse than one timed
    /// roughly, and the estimate is still there to fall back on.
    /// </summary>
    [Fact]
    public void TwiceTheWordsIsRefused()
    {
        var once = Placed(0, 1, "one", "two", "three");
        var twice = once.Concat(Placed(0.1, 1, "one", "two", "three")).ToList();

        Assert.Null(MeasuredWords.Pair(twice, Own("one", "two", "three")));
    }

    [Fact]
    public void AWordShortIsRefused() =>
        Assert.Null(MeasuredWords.Pair(Placed(0, 1, "one", "two"), Own("one", "two", "three")));

    /// <summary>Measurements are used exactly as given, including ones outside the old bounds.</summary>
    [Fact]
    public void TimesAreTakenAsMeasured()
    {
        var paired = MeasuredWords.Pair(Placed(51.4, 0.5, "you", "accept"), Own("You", "accept"));

        Assert.NotNull(paired);
        Assert.Equal(51.4, paired[0].StartSeconds, 3);
        Assert.Equal(51.9, paired[1].StartSeconds, 3);
    }

    /// <summary>Where each word sits in its segment's text is the caller's, not the aligner's.</summary>
    [Fact]
    public void TheOffsetsSurvivePairing()
    {
        IReadOnlyList<WordTimings.Word> own =
        [
            new WordTimings.Word("You", 0, 0) { Offset = 0 },
            new WordTimings.Word("accept", 0, 0) { Offset = 4 },
        ];

        var paired = MeasuredWords.Pair(Placed(10, 1, "you", "accept"), own);

        Assert.NotNull(paired);
        Assert.Equal([0, 4], paired.Select(w => w.Offset));
    }

    [Fact]
    public void NothingMeasuredMeansNoAnswer() =>
        Assert.Null(MeasuredWords.Pair([], Own("one")));

    [Fact]
    public void ASegmentWithNoWordsHasNoAnswer() =>
        Assert.Null(MeasuredWords.Pair(Placed(0, 1, "one"), Own()));
}
