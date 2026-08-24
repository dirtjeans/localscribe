using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class SpeechTaskTests
{
    private static WhisperTokenizer Multilingual() =>
        WhisperTokenizer.FromTokenMap(new Dictionary<string, int>
        {
            ["<|startoftranscript|>"] = 50258,
            ["<|endoftext|>"] = 50257,
            ["<|transcribe|>"] = 50360,
            ["<|translate|>"] = 50359,
            ["<|en|>"] = 50259,
            ["<|pt|>"] = 50270,
            ["<|notimestamps|>"] = 50364,
            ["<|0.00|>"] = 50365,
            ["hello"] = 1,
        });

    private static WhisperTokenizer EnglishOnly() =>
        WhisperTokenizer.FromTokenMap(new Dictionary<string, int>
        {
            ["<|startoftranscript|>"] = 50258,
            ["<|endoftext|>"] = 50257,
            ["<|transcribe|>"] = 50359,
            ["<|notimestamps|>"] = 50363,
            ["<|0.00|>"] = 50364,
            ["hello"] = 1,
        });

    [Fact]
    public void TranscribingIsWhatHappensByDefault()
    {
        var tokenizer = Multilingual();

        Assert.Equal(tokenizer.Special.Transcribe, tokenizer.TaskToken(SpeechTask.Transcribe));
        Assert.DoesNotContain(tokenizer.Special.Translate, tokenizer.BuildPrompt());
    }

    [Fact]
    public void AskingForEnglishUsesTheTranslateTask()
    {
        var tokenizer = Multilingual();

        var prompt = tokenizer.BuildPrompt(task: SpeechTask.TranslateToEnglish);

        Assert.Contains(tokenizer.Special.Translate, prompt);
        Assert.DoesNotContain(tokenizer.Special.Transcribe, prompt);
    }

    /// <summary>
    /// The source language is still named. Translation reads the speech as what it is and writes
    /// English; leaving the slot empty is the fault that made a live transcript open with
    /// "Gracias." and it is no less wrong here.
    /// </summary>
    [Fact]
    public void TheSpokenLanguageIsStillDeclaredWhenTranslating()
    {
        var tokenizer = Multilingual();
        var portuguese = tokenizer.Languages.First(l => l.Value == "pt").Key;

        var prompt = tokenizer.BuildPrompt(
            languageToken: portuguese, task: SpeechTask.TranslateToEnglish);

        Assert.Contains(portuguese, prompt);
        Assert.Contains(tokenizer.Special.Translate, prompt);
    }

    /// <summary>
    /// An English-only export has no translate task, having nothing to translate from. Asking
    /// for one gets a transcript rather than a refusal: the setting can be changed afterwards,
    /// and the language actually spoken is the safer of the two wrong answers.
    /// </summary>
    [Fact]
    public void AnEnglishOnlyModelQuietlyTranscribesInstead()
    {
        var tokenizer = EnglishOnly();

        Assert.False(tokenizer.CanTranslate);
        Assert.Equal(
            tokenizer.Special.Transcribe,
            tokenizer.TaskToken(SpeechTask.TranslateToEnglish));
    }

    [Fact]
    public void AMultilingualModelSaysItCanTranslate() =>
        Assert.True(Multilingual().CanTranslate);
}
