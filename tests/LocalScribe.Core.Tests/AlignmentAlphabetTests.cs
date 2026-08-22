using LocalScribe.Core.Alignment;
using Xunit;

namespace LocalScribe.Core.Tests;

public class AlignmentAlphabetTests
{
    /// <summary>The real vocabulary of the multilingual aligner, trimmed to what matters here.</summary>
    private static AlignmentAlphabet Real() => AlignmentAlphabet.Parse(
        """
        {"<blank>":0,"<pad>":1,"</s>":2,"<unk>":3,"a":4,"i":5,"e":6,"n":7,"o":8,"u":9,
         "t":10,"s":11,"r":12,"m":13,"k":14,"l":15,"d":16,"g":17,"h":18,"y":19,"b":20,
         "p":21,"w":22,"c":23,"v":24,"j":25,"z":26,"'":27,"q":28,"x":29,"f":30}
        """);

    [Fact]
    public void TheAlphabetIsReadFromTheModelsOwnVocabulary()
    {
        var alphabet = Real();

        Assert.Equal(0, alphabet.Blank);
        Assert.Equal(31, alphabet.Size);
    }

    [Fact]
    public void AWordSpellsToItsLetters()
    {
        var (tokens, words) = Real().Spell(["cat"]);

        Assert.Equal([23, 4, 10], tokens);
        Assert.Single(words);
        Assert.Equal(0, words[0].First);
        Assert.Equal(3, words[0].Count);
    }

    [Fact]
    public void EachWordKnowsWhichLettersAreIts()
    {
        var (tokens, words) = Real().Spell(["at", "on"]);

        Assert.Equal(4, tokens.Count);
        Assert.Equal((0, 2), (words[0].First, words[0].Count));
        Assert.Equal((2, 2), (words[1].First, words[1].Count));
    }

    /// <summary>Punctuation is not spoken, so it is not aligned.</summary>
    [Theory]
    [InlineData("Yes,", "yes")]
    [InlineData("agree.", "agree")]
    [InlineData("\"quoted\"", "quoted")]
    [InlineData("half-time", "halftime")]
    [InlineData("don't", "don't")]
    [InlineData("don’t", "don't")]
    public void PunctuationIsDroppedButApostrophesAreKept(string word, string expected) =>
        Assert.Equal(expected, AlignmentAlphabet.Fold(word));

    /// <summary>
    /// The point of a romanised alphabet: one model for every language, reached by folding each
    /// language into the same letters.
    /// </summary>
    [Theory]
    [InlineData("café", "cafe")]
    [InlineData("Ünterstützung", "unterstutzung")]
    [InlineData("naïve", "naive")]
    [InlineData("SEÑOR", "senor")]
    [InlineData("Ærø", "aero")]
    [InlineData("Straße", "strasse")]
    [InlineData("Þór", "thor")]
    public void AccentsAreFoldedAway(string word, string expected) =>
        Assert.Equal(expected, AlignmentAlphabet.Fold(word));

    /// <summary>
    /// A word in a script the folding cannot reach loses its timing and nothing else — it must
    /// not take the rest of the sentence down with it.
    /// </summary>
    [Fact]
    public void AWordThatCannotBeSpelledIsSkipped()
    {
        var (tokens, words) = Real().Spell(["the", "日本語", "word"]);

        Assert.Equal(2, words.Count);
        Assert.Equal("the", words[0].Word);
        Assert.Equal("word", words[1].Word);

        // And the one that survived still points at the right letters.
        Assert.Equal(3, words[1].First);
        Assert.Equal(4, words[1].Count);
        Assert.Equal(7, tokens.Count);
    }

    [Fact]
    public void TheWordKeepsItsPlaceInTheOriginalList()
    {
        var (_, words) = Real().Spell(["one", "!!!", "three"]);

        Assert.Equal(0, words[0].Index);
        Assert.Equal(2, words[1].Index);
    }

    [Fact]
    public void NothingToSpellIsNotAFailure()
    {
        var (tokens, words) = Real().Spell([]);

        Assert.Empty(tokens);
        Assert.Empty(words);
    }
}
