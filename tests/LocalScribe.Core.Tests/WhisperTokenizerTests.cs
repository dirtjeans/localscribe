using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public sealed class WhisperTokenizerTests
{
    /// <summary>
    /// A miniature vocabulary shaped like a real Whisper one: byte-level text tokens first,
    /// then the special markers above them.
    /// </summary>
    private static WhisperTokenizer BuildTokenizer()
    {
        var map = new Dictionary<string, int>
        {
            // "Ġ" is the byte-level stand-in for a leading space.
            ["Hello"] = 0,
            ["Ġworld"] = 1,
            ["Ġthere"] = 2,
            ["!"] = 3,
            ["Ġcaf"] = 4,
            // "é" is two UTF-8 bytes, which byte-level encoding represents as two characters.
            ["Ã©"] = 5,
            ["<|startoftranscript|>"] = 100,
            ["<|transcribe|>"] = 101,
            ["<|notimestamps|>"] = 102,
            ["<|nospeech|>"] = 103,
            ["<|endoftext|>"] = 104,
            ["<|0.00|>"] = 105,
        };

        return WhisperTokenizer.FromTokenMap(map);
    }

    [Fact]
    public void SpecialTokenIdsComeFromTheVocabularyNotFromConstants()
    {
        // Hard-coded ids are wrong across model variants, so they must be looked up.
        var tokenizer = BuildTokenizer();

        Assert.Equal(100, tokenizer.Special.StartOfTranscript);
        Assert.Equal(104, tokenizer.Special.EndOfText);
        Assert.Equal(105, tokenizer.Special.FirstTimestamp);
    }

    [Fact]
    public void LeadingSpaceMarkersDecodeBackToSpaces()
    {
        var tokenizer = BuildTokenizer();

        Assert.Equal("Hello world", tokenizer.Decode([0, 1]));
    }

    [Fact]
    public void MultiByteCharactersSurviveTheRoundTrip()
    {
        // Byte-level tokens split é across two entries. Decoding per-character would corrupt it.
        var tokenizer = BuildTokenizer();

        Assert.Equal(" café", tokenizer.Decode([4, 5]));
    }

    [Fact]
    public void SpecialTokensAreStrippedFromTheText()
    {
        var tokenizer = BuildTokenizer();

        var text = tokenizer.Decode([100, 101, 102, 0, 1, 104]);

        Assert.Equal("Hello world", text);
    }

    [Fact]
    public void UnknownIdsAreSkippedRatherThanThrowing()
    {
        // A truncated or mismatched vocabulary should degrade, not crash mid-transcription.
        var tokenizer = BuildTokenizer();

        Assert.Equal("Hello", tokenizer.Decode([0, 9999]));
    }

    [Fact]
    public void TimestampTokensConvertToSeconds()
    {
        var tokenizer = BuildTokenizer();

        Assert.Equal(0.0, tokenizer.TimestampToSeconds(105), precision: 6);
        Assert.Equal(1.0, tokenizer.TimestampToSeconds(155), precision: 6);
    }

    [Fact]
    public void NonTimestampIdsAreRejectedByTheTimestampConverter()
    {
        var tokenizer = BuildTokenizer();

        Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.TimestampToSeconds(0));
    }

    [Fact]
    public void PromptOpensWithStartOfTranscriptAndSelectsTranscription()
    {
        var tokenizer = BuildTokenizer();

        var prompt = tokenizer.BuildPrompt(withTimestamps: true);

        Assert.Equal(100, prompt[0]);
        Assert.Contains(101, prompt);
        Assert.DoesNotContain(102, prompt);
    }

    [Fact]
    public void SuppressingTimestampsAddsTheMarker()
    {
        var tokenizer = BuildTokenizer();

        Assert.Contains(102, tokenizer.BuildPrompt(withTimestamps: false));
    }

    [Fact]
    public void EnglishOnlyModelsOmitTheLanguageToken()
    {
        // The .en models have no language token at all, so passing none must be valid.
        var tokenizer = BuildTokenizer();

        var prompt = tokenizer.BuildPrompt(withTimestamps: true, languageToken: -1);

        Assert.Equal([100, 101], prompt);
    }

    [Fact]
    public void AVocabularyWithoutWhisperMarkersIsRejectedClearly()
    {
        // Pointing this at a plain GPT-2 vocab.json is an easy mistake with a confusing failure.
        var map = new Dictionary<string, int> { ["Hello"] = 0 };

        var exception = Assert.Throws<InvalidOperationException>(() => WhisperTokenizer.FromTokenMap(map));

        Assert.Contains("Whisper export", exception.Message, StringComparison.Ordinal);
    }
}
