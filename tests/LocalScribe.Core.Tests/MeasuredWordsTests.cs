using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class MeasuredWordsTests
{
    private static TranscriptSegment Segment(string text, double from, double to) =>
        new(text, from, to);

    /// <summary>The segment's own words, as the caller splits them.</summary>
    private static IReadOnlyList<WordTimings.Word> Own(params string[] words) =>
        [.. words.Select(w => new WordTimings.Word(w, 0, 0))];

    private static IReadOnlyList<WordTimings.Word> Placed(double from, double step, params string[] words) =>
        [.. words.Select((w, i) => new WordTimings.Word(w, from + (i * step), from + ((i + 1) * step)))];

    [Fact]
    public void MeasuredTimesReplaceTheEstimate()
    {
        var pool = Placed(0, 1, "one", "two", "three");

        var paired = MeasuredWords.Pair(Segment("one two three", 0, 3), pool, Own("one", "two", "three"));

        Assert.NotNull(paired);
        Assert.Equal(3, paired.Count);
        Assert.Equal(1, paired[1].StartSeconds, 3);
    }

    /// <summary>The text is the reader's; only the times are borrowed.</summary>
    [Fact]
    public void TheSegmentsOwnWordsAreKept()
    {
        var pool = Placed(0, 1, "dont", "stop");

        var paired = MeasuredWords.Pair(Segment("Don't stop.", 0, 2), pool, Own("Don't", "stop."));

        Assert.NotNull(paired);
        Assert.Equal("Don't", paired[0].Text);
        Assert.Equal("stop.", paired[1].Text);
    }

    /// <summary>
    /// The bug this rule exists to make impossible. Two alignments of one recording cover the
    /// same seconds, so a lookup by time finds both and the segment is timed against an
    /// interleaving of the two — early in places, late in others, with no pattern to it. It must
    /// refuse instead, and let the estimate stand.
    /// </summary>
    [Fact]
    public void TwoAlignmentsOfTheSameSecondsAreRefused()
    {
        var once = Placed(0, 1, "one", "two", "three");
        var again = Placed(0.1, 1, "one", "two", "three");

        var pool = once.Concat(again).ToList();

        Assert.Null(MeasuredWords.Pair(Segment("one two three", 0, 3), pool, Own("one", "two", "three")));
    }

    [Fact]
    public void AWordShortIsRefused()
    {
        var pool = Placed(0, 1, "one", "two");

        Assert.Null(MeasuredWords.Pair(Segment("one two three", 0, 3), pool, Own("one", "two", "three")));
    }

    /// <summary>
    /// A word starting on a boundary belongs to the segment beginning there, and to that one
    /// only. Claimed by both, the earlier segment has a word too many; claimed by neither, the
    /// later one is a word short — and either way both fall back to the estimate.
    /// </summary>
    [Fact]
    public void ABoundaryWordBelongsToExactlyOneSegment()
    {
        var pool = Placed(0, 1, "one", "two", "three", "four");

        var first = MeasuredWords.Pair(Segment("one two", 0, 2), pool, Own("one", "two"));
        var second = MeasuredWords.Pair(Segment("three four", 2, 4), pool, Own("three", "four"));

        Assert.NotNull(first);
        Assert.NotNull(second);
        Assert.Equal(2, second[0].StartSeconds, 3);
    }

    /// <summary>Floating point does not always agree with itself about a shared boundary.</summary>
    [Fact]
    public void AWordAMomentEarlyStillCounts()
    {
        var pool = Placed(1.998, 1, "three", "four");

        var paired = MeasuredWords.Pair(Segment("three four", 2, 4), pool, Own("three", "four"));

        Assert.NotNull(paired);
    }

    [Fact]
    public void NothingMeasuredMeansNoAnswer() =>
        Assert.Null(MeasuredWords.Pair(Segment("one", 0, 1), [], Own("one")));

    [Fact]
    public void ASegmentWithNoWordsHasNoAnswer() =>
        Assert.Null(MeasuredWords.Pair(Segment("", 0, 1), Placed(0, 1, "one"), Own()));
}
