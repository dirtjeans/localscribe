using LocalScribe.Core.Audio;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class WordTimingsTests
{
    private const int Rate = 16000;

    /// <summary>Audio that is loud over the given spans and silent everywhere else.</summary>
    private static PcmAudio Speech(double seconds, params (double From, double To)[] loud)
    {
        var samples = new float[(int)(seconds * Rate)];

        foreach (var (from, to) in loud)
        {
            for (var i = (int)(from * Rate); i < (int)(to * Rate) && i < samples.Length; i++)
            {
                samples[i] = (float)(Math.Sin(2 * Math.PI * 200 * i / Rate) * 0.5);
            }
        }

        return new PcmAudio(samples, Rate);
    }

    [Fact]
    public void EveryWordGetsATime()
    {
        var audio = Speech(4, (0, 4));
        var segment = new TranscriptSegment("one two three four", 0, 4);

        var words = WordTimings.For(audio, segment);

        Assert.Equal(4, words.Count);
        Assert.Equal(["one", "two", "three", "four"], words.Select(w => w.Text));
    }

    [Fact]
    public void TheWordsRunInOrderAndStayInsideTheSegment()
    {
        var audio = Speech(6, (1, 5));
        var segment = new TranscriptSegment("the quick brown fox jumps", 1, 5);

        var words = WordTimings.For(audio, segment);

        Assert.All(words, w => Assert.InRange(w.StartSeconds, 1, 5));
        for (var i = 1; i < words.Count; i++)
        {
            Assert.True(words[i].StartSeconds >= words[i - 1].StartSeconds,
                "words must not run backwards");
        }
    }

    /// <summary>
    /// The reason for weighing by loudness at all. A segment with a two-second pause in the
    /// middle must not put words inside the pause: dividing the clock evenly would place the
    /// third of four words in silence.
    /// </summary>
    [Fact]
    public void WordsAreNotPlacedInSilence()
    {
        // Talking for a second, two seconds of nothing, then talking again.
        var audio = Speech(4, (0, 1), (3, 4));
        var segment = new TranscriptSegment("aaa bbb ccc ddd", 0, 4);

        var words = WordTimings.For(audio, segment);

        var inThePause = words.Count(w => w.StartSeconds > 1.3 && w.StartSeconds < 2.7);

        Assert.True(inThePause <= 1, $"{inThePause} words landed in the silence");
    }

    /// <summary>
    /// And the consequence that matters: the second half of the sentence should be found in the
    /// second burst of speech, not adrift in the middle.
    /// </summary>
    [Fact]
    public void TheSecondHalfLandsInTheSecondBurst()
    {
        var audio = Speech(4, (0, 1), (3, 4));
        var segment = new TranscriptSegment("aaa bbb ccc ddd", 0, 4);

        var words = WordTimings.For(audio, segment);

        Assert.True(words[^1].StartSeconds > 2.5,
            $"the last word was placed at {words[^1].StartSeconds:F2}s, before the speech resumed");
    }

    [Fact]
    public void OffsetsPointAtTheWordInTheText()
    {
        var audio = Speech(3, (0, 3));
        var segment = new TranscriptSegment("alpha beta gamma", 0, 3);

        var words = WordTimings.For(audio, segment);

        Assert.Equal(0, words[0].Offset);
        Assert.Equal(6, words[1].Offset);
        Assert.Equal(11, words[2].Offset);
        Assert.All(words, w => Assert.Equal(w.Text, segment.Text.Substring(w.Offset, w.Text.Length)));
    }

    [Fact]
    public void OneWordTakesTheWholeSegment()
    {
        var words = WordTimings.For(Speech(2, (0, 2)), new TranscriptSegment("Hello", 0, 2));

        Assert.Single(words);
        Assert.Equal(0, words[0].StartSeconds, 2);
        Assert.Equal(2, words[0].EndSeconds, 2);
    }

    [Fact]
    public void SilenceThroughoutStillDividesTheWordsUp()
    {
        var words = WordTimings.For(Speech(4), new TranscriptSegment("one two three four", 0, 4));

        Assert.Equal(4, words.Count);
        Assert.True(words[^1].StartSeconds > words[0].StartSeconds,
            "even in silence the words must not pile up at one instant");
    }

    [Fact]
    public void AnEmptySegmentHasNoWords() =>
        Assert.Empty(WordTimings.For(Speech(1, (0, 1)), new TranscriptSegment("   ", 0, 1)));

    [Fact]
    public void PunctuationStaysWithItsWord()
    {
        var words = WordTimings.For(Speech(3, (0, 3)), new TranscriptSegment("Yes, I agree.", 0, 3));

        Assert.Equal(["Yes,", "I", "agree."], words.Select(w => w.Text));
    }
}
