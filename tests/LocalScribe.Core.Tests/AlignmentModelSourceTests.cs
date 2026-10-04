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

    /// <summary>
    /// The 4-bit build keeps its published name — the aligner looks for it by that name — and
    /// brings the vocabulary with it, which no build of the model can do without.
    /// </summary>
    [Fact]
    public void TheQuantisedBuildKeepsItsNameAndItsVocabulary()
    {
        var required = AlignmentModelSource.QuantisedFiles.Where(f => !f.Optional).Select(f => f.FileName).ToList();

        Assert.Equal(["model_q4.onnx", "vocab.json"], required);
        Assert.EndsWith("model_q4.onnx", AlignmentModelSource.QuantisedFiles[0].Source.AbsolutePath, StringComparison.Ordinal);
    }

    /// <summary>The point of it: well under half the fp16 download.</summary>
    [Fact]
    public void TheQuantisedBuildIsTheSmallerDownload() =>
        Assert.True(AlignmentModelSource.QuantisedApproximateBytes < AlignmentModelSource.ApproximateBytes / 2);

    /// <summary>
    /// What this machine fetches and what it checks for must name the same file, or setup
    /// would download the 4-bit build and then report the aligner missing forever after.
    /// </summary>
    [Fact]
    public void ThisMachineFetchesTheFileItLooksFor() =>
        Assert.Contains(
            AlignmentModelSource.FilesForThisMachine,
            file => file.FileName == AlignmentModelSource.ModelFileNameForThisMachine);
}
