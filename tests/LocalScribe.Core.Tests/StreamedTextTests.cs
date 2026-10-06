using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// The text streamed during a file run should say what the finished transcript will: the seam
/// between windows shown once, as the stitcher will leave it, not twice until the end.
/// </summary>
public class StreamedTextTests
{
    [Fact]
    public void TheSeamIsShownOnce()
    {
        var trimmed = StreamedText.Trim(
            "We started the project in March. If the stitching works correctly,",
            "works correctly, this sentence will appear once.");

        Assert.Equal("this sentence will appear once.", trimmed);
    }

    [Fact]
    public void TheFirstWindowHasNothingToTrimAgainst()
    {
        Assert.Equal("Hello and welcome.", StreamedText.Trim(null, "Hello and welcome."));
    }

    [Fact]
    public void ALoopInsideTheWindowIsShownOnce()
    {
        var trimmed = StreamedText.Trim(
            null,
            "They cannot be victims to this. They cannot be victims to this. And so we moved on.");

        Assert.Equal("They cannot be victims to this. And so we moved on.", trimmed);
    }

    [Fact]
    public void AWindowThatOnlyRepeatsIsEmpty()
    {
        // Nothing new was said in it, so nothing new should appear.
        Assert.Equal(string.Empty, StreamedText.Trim("and that was the end of it.", "the end of it."));
    }

    [Fact]
    public void StreamingAgreesWithTheStitchedTranscript()
    {
        // Two windows overlapping by two seconds, as the chunker cuts them.
        IReadOnlyList<TranscriptSegment> first =
        [
            new("So the first thing to say is that the plan changed.", 0, 14),
            new("Nobody expected the budget to hold, and it did not hold", 14, 30),
        ];
        IReadOnlyList<TranscriptSegment> second =
        [
            new("it did not hold for long. By April it was gone.", 28.5, 40),
        ];

        var stitched = new TranscriptStitcher().Stitch([first, second]);
        var finished = string.Join(" ", stitched.Select(s => s.Text.Trim()));

        var window1 = StreamedText.Trim(null, string.Join(" ", first.Select(s => s.Text)));
        var window2 = StreamedText.Trim(window1, string.Join(" ", second.Select(s => s.Text)));

        Assert.Equal((0, 0), StreamedText.Compare($"{window1} {window2}", finished));
    }

    [Fact]
    public void ComparisonCountsWhatAReaderSawVanish()
    {
        var (extra, missing) = StreamedText.Compare(
            "it did not hold, it did not hold for long",
            "It did not hold for long.");

        Assert.Equal((4, 0), (extra, missing));
    }
}
