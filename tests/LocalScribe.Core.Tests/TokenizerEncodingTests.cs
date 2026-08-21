using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// Turning text back into tokens, so the decoder can be given a prompt.
/// <para>
/// The tokenizer has only ever decoded. Encoding is needed for one thing: conditioning a decode
/// on example text, which is the standard remedy for Whisper dropping into its unpunctuated
/// mode. An encoder that is subtly wrong produces a prompt of plausible nonsense, so the test
/// that matters is that text survives the round trip.
/// </para>
/// </summary>
public class TokenizerEncodingTests
{
    /// <summary>
    /// A byte-level vocabulary of single characters, built the way Whisper's is: printable
    /// ASCII stands for itself, and a space is the character at 256 + 32.
    /// </summary>
    private static WhisperTokenizer Build(params string[] extraTokens)
    {
        var map = new Dictionary<string, int>(StringComparer.Ordinal);
        var next = 0;

        for (var c = '!'; c <= '~'; c++)
        {
            map[c.ToString()] = next++;
        }

        map[((char)(256 + 32)).ToString()] = next++;   // space

        foreach (var token in extraTokens)
        {
            map[token] = next++;
        }

        // Specials must sort above every text token, as they do in the real vocabulary.
        map["<|startoftranscript|>"] = 50000;
        map["<|endoftext|>"] = 50001;
        map["<|transcribe|>"] = 50002;
        map["<|startofprev|>"] = 50003;
        map["<|en|>"] = 50004;

        return WhisperTokenizer.FromTokenMap(map);
    }

    [Theory]
    [InlineData("Hello.")]
    [InlineData("Hello, and thank you.")]
    [InlineData("Yes - that's right.")]
    [InlineData("One two three four five.")]
    public void TextSurvivesTheRoundTrip(string text)
    {
        var tokenizer = Build();

        Assert.Equal(text, tokenizer.Decode(tokenizer.Encode(text)));
    }

    /// <summary>
    /// A leading space belongs to the word after it, which is what byte-level vocabularies are
    /// built around. Encoding a space as its own ordinary character would give a token sequence
    /// no model was trained on.
    /// </summary>
    [Fact]
    public void ASpaceIsEncodedAsTheVocabularysStandIn()
    {
        var tokenizer = Build();

        Assert.Equal(" ", tokenizer.Decode(tokenizer.Encode(" ")));
    }

    /// <summary>Longer tokens win, or the encoding is a character at a time and needlessly long.</summary>
    [Fact]
    public void TheLongestMatchIsPreferred()
    {
        var tokenizer = Build("the", "ing");

        var single = tokenizer.Encode("the");
        var characters = tokenizer.Encode("t") .Concat(tokenizer.Encode("h")).Concat(tokenizer.Encode("e"));

        Assert.Single(single);
        Assert.Equal(3, characters.Count());
        Assert.Equal("the", tokenizer.Decode(single));
    }

    [Fact]
    public void NothingInNothingOut() => Assert.Empty(Build().Encode(string.Empty));

    /// <summary>
    /// A prompt is conditioning, not speech, so it goes after the marker that says so and before
    /// the one that starts the transcript.
    /// </summary>
    [Fact]
    public void APromptIsPlacedBeforeTheStartMarker()
    {
        var tokenizer = Build();
        var prior = tokenizer.Encode("Hello.");

        var prompt = tokenizer.BuildPrompt(withTimestamps: true, languageToken: -1, priorTokens: prior);

        Assert.Equal(tokenizer.Special.StartOfPrev, prompt[0]);
        Assert.Equal(prior, prompt.Skip(1).Take(prior.Count).ToList());
        Assert.Equal(tokenizer.Special.StartOfTranscript, prompt[prior.Count + 1]);
    }

    [Fact]
    public void WithoutAPromptTheOpeningIsUnchanged()
    {
        var tokenizer = Build();

        var prompt = tokenizer.BuildPrompt();

        Assert.Equal(tokenizer.Special.StartOfTranscript, prompt[0]);
        Assert.DoesNotContain(tokenizer.Special.StartOfPrev, prompt);
    }

    /// <summary>
    /// Special markers must never be encodable as text: one appearing inside a prompt would end
    /// it early and be read as an instruction rather than as words.
    /// </summary>
    [Fact]
    public void SpecialMarkersAreNotEncodableAsText()
    {
        var tokenizer = Build();

        var encoded = tokenizer.Encode("<|startoftranscript|>");

        Assert.DoesNotContain(tokenizer.Special.StartOfTranscript, encoded);
    }
}
