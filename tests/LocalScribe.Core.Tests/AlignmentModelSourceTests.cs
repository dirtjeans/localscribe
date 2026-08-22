using LocalScribe.Core.Models;
using Xunit;

namespace LocalScribe.Core.Tests;

public class AlignmentModelSourceTests
{
    /// <summary>
    /// The names the aligner looks for on disk. Fetching the right bytes under the wrong name
    /// leaves the model invisible, and the app's only symptom is that word times quietly go back
    /// to being estimated.
    /// </summary>
    [Fact]
    public void TheFilesAreNamedAsTheAlignerExpects()
    {
        var names = AlignmentModelSource.Files.Select(f => f.FileName).ToList();

        Assert.Contains("model_fp16.onnx", names);
        Assert.Contains("vocab.json", names);
    }

    /// <summary>
    /// Half precision, not one of the quantised builds. Those use ConvInteger, which ONNX Runtime
    /// cannot run on ARM64 at all — fetching one would fail at load rather than run slowly.
    /// </summary>
    [Fact]
    public void TheWeightsAreNotQuantised()
    {
        var weights = AlignmentModelSource.Files.Single(f => f.FileName.EndsWith(".onnx", StringComparison.Ordinal));

        Assert.DoesNotContain("int8", weights.Source.AbsoluteUri, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("quantized", weights.Source.AbsoluteUri, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("uint8", weights.Source.AbsoluteUri, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void EverythingComesFromOneOngatedRepository()
    {
        foreach (var file in AlignmentModelSource.Files)
        {
            Assert.Equal("huggingface.co", file.Source.Host);
            Assert.Equal(Uri.UriSchemeHttps, file.Source.Scheme);
        }
    }

    /// <summary>Only the weights and the vocabulary are needed to run; the rest is provenance.</summary>
    [Fact]
    public void OnlyWhatTheAlignerReadsIsRequired()
    {
        var required = AlignmentModelSource.Files.Where(f => !f.Optional).Select(f => f.FileName).ToList();

        Assert.Equal(["model_fp16.onnx", "vocab.json"], required);
    }

    [Fact]
    public void TheSizeWarnedAboutIsTheSizeItIs() =>
        Assert.InRange(AlignmentModelSource.ApproximateBytes, 500L * 1024 * 1024, 700L * 1024 * 1024);
}
