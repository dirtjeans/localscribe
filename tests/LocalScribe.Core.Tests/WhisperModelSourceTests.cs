using LocalScribe.Core.Models;
using Xunit;

namespace LocalScribe.Core.Tests;

public class WhisperModelSourceTests
{
    /// <summary>
    /// Every size the planner can return must be fetchable, or --fetch-models sends the user
    /// away empty-handed on exactly the machine the plan was built for.
    /// </summary>
    [Theory]
    [InlineData("tiny.en")]
    [InlineData("base.en")]
    [InlineData("small.en")]
    [InlineData("medium.en")]
    public void EverySizeThePlannerCanChooseIsFetchable(string size) =>
        Assert.True(WhisperModelSource.IsSupported(size));

    [Fact]
    public void UnknownSizesAreRejectedRatherThanGuessedAt()
    {
        Assert.False(WhisperModelSource.IsSupported("large-v3"));
        Assert.Throws<ArgumentOutOfRangeException>(() => WhisperModelSource.For("large-v3"));
    }

    [Fact]
    public void AnExportCarriesTheThreeFilesTheAppRequires()
    {
        var names = WhisperModelSource.For("base.en").Select(d => d.FileName).ToList();

        Assert.Contains("encoder.onnx", names);
        Assert.Contains("decoder.onnx", names);
        Assert.Contains("vocab.json", names);
    }

    /// <summary>
    /// WhisperTokenizer reads added_tokens.json only when it exists, so a missing one must not
    /// fail the fetch.
    /// </summary>
    [Fact]
    public void OnlyAddedTokensIsOptional()
    {
        var downloads = WhisperModelSource.For("base.en");

        Assert.All(
            downloads.Where(d => d.FileName != "added_tokens.json"),
            d => Assert.False(d.Optional));

        Assert.True(downloads.Single(d => d.FileName == "added_tokens.json").Optional);
    }

    /// <summary>
    /// The decode loop re-runs the whole prefix each step rather than carrying a key/value
    /// cache, so it needs the export that takes no past state. The _merged and _with_past
    /// variants sit in the same directory and would load without complaint.
    /// </summary>
    [Fact]
    public void TheDecoderIsThePastFreeExport()
    {
        var decoder = WhisperModelSource.For("small.en").Single(d => d.FileName == "decoder.onnx");

        Assert.EndsWith("decoder_model.onnx", decoder.Source.AbsolutePath, StringComparison.Ordinal);
        Assert.DoesNotContain("with_past", decoder.Source.AbsolutePath, StringComparison.Ordinal);
        Assert.DoesNotContain("merged", decoder.Source.AbsolutePath, StringComparison.Ordinal);
    }

    [Fact]
    public void SourcesAreHttpsAndSizeSpecific()
    {
        foreach (var download in WhisperModelSource.For("tiny.en"))
        {
            Assert.Equal(Uri.UriSchemeHttps, download.Source.Scheme);
            Assert.Contains("whisper-tiny.en", download.Source.ToString(), StringComparison.Ordinal);
        }
    }
}
