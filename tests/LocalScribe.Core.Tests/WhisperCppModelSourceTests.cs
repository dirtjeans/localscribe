using LocalScribe.Core.Models;
using Xunit;

namespace LocalScribe.Core.Tests;

public class WhisperCppModelSourceTests
{
    /// <summary>
    /// The files keep their published names. whisper.cpp derives the Core ML encoder bundle's
    /// path from the GGML file's name, so a rename on either side breaks the pairing without
    /// an error message — the never-rename invariant, in its Core ML form.
    /// </summary>
    [Fact]
    public void FilesKeepTheirPublishedNames()
    {
        var files = WhisperCppModelSource.Files("large-v3-turbo", coreMl: true);

        Assert.Equal(2, files.Count);
        Assert.Equal("ggml-large-v3-turbo.bin", files[0].FileName);
        Assert.Equal("ggml-large-v3-turbo-encoder.mlmodelc.zip", files[1].FileName);
        Assert.EndsWith(files[0].FileName, files[0].Source.AbsolutePath, StringComparison.Ordinal);
        Assert.EndsWith(files[1].FileName, files[1].Source.AbsolutePath, StringComparison.Ordinal);
    }

    /// <summary>
    /// The Core ML encoder must never fail the fetch. A size the repository has no Core ML
    /// build for degrades to the Metal encoder; losing the whole model over the optional half
    /// would be the wrong trade.
    /// </summary>
    [Fact]
    public void TheCoreMlEncoderIsOptionalAndTheModelIsNot()
    {
        var files = WhisperCppModelSource.Files("large-v3-turbo", coreMl: true);

        Assert.False(files[0].Optional);
        Assert.True(files[1].Optional);
    }

    /// <summary>
    /// Off Apple silicon the Core ML bundle is dead weight: nothing there can run it, and it
    /// is a third of the download.
    /// </summary>
    [Fact]
    public void WithoutCoreMlOnlyTheModelIsFetched()
    {
        var files = WhisperCppModelSource.Files("base.en", coreMl: false);

        Assert.Single(files);
        Assert.Equal("ggml-base.en.bin", files[0].FileName);
    }

    /// <summary>
    /// Unpacking an encoder that was never downloaded reports the truth rather than
    /// pretending: the caller's message differs between "Neural Engine ready" and "Metal
    /// fallback", and only this return value tells them apart.
    /// </summary>
    [Fact]
    public void UnpackingWithNothingDownloadedSaysSo()
    {
        var directory = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(directory);

        try
        {
            Assert.False(WhisperCppModelSource.UnpackCoreMlEncoder(directory, "large-v3-turbo"));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
