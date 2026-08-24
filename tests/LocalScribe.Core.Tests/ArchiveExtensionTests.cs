using LocalScribe.Core.Archive;
using Xunit;

namespace LocalScribe.Core.Tests;

public class ArchiveExtensionTests
{
    [Fact]
    public void TheCurrentExtensionIsShortAndOurs() =>
        Assert.Equal(".scrb", TranscriptArchive.Extension);

    /// <summary>
    /// A file already on disk does not stop being readable because the name got shorter. The
    /// container never changed; only what it is called did.
    /// </summary>
    [Theory]
    [InlineData(@"C:\notes\interview.scrb")]
    [InlineData(@"C:\notes\interview.lscribe")]
    [InlineData("interview.SCRB")]
    [InlineData("interview.LScribe")]
    public void BothNamesOpen(string path) =>
        Assert.True(TranscriptArchive.IsArchive(path));

    [Theory]
    [InlineData("recording.wav")]
    [InlineData("transcript.txt")]
    [InlineData("notes.scr")]
    [InlineData("book.scrbl")]
    [InlineData("noextension")]
    public void OtherFilesAreNotArchives(string path) =>
        Assert.False(TranscriptArchive.IsArchive(path));

    /// <summary>Saving offers one name; opening offers every name, current first.</summary>
    [Fact]
    public void TheCurrentNameLeadsTheList()
    {
        Assert.Equal(TranscriptArchive.Extension, TranscriptArchive.Extensions[0]);
        Assert.Contains(".lscribe", TranscriptArchive.Extensions);
    }

    /// <summary>Every name in the list must be one the file dialogs will accept.</summary>
    [Fact]
    public void EveryNameIsAWellFormedExtension()
    {
        foreach (var extension in TranscriptArchive.Extensions)
        {
            Assert.StartsWith(".", extension, StringComparison.Ordinal);
            Assert.Equal(extension.ToLowerInvariant(), extension);
            Assert.DoesNotContain(" ", extension, StringComparison.Ordinal);
        }
    }
}
