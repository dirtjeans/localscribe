using LocalScribe.Core.Diarization;
using Xunit;

namespace LocalScribe.Core.Tests;

public class DiarizationChoiceTests
{
    private sealed class Scratch : IDisposable
    {
        public Scratch() => Directory.CreateDirectory(Path);

        public string Path { get; } = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "localscribe-choice-" + Guid.NewGuid().ToString("N"));

        public void Dispose()
        {
            try
            {
                Directory.Delete(Path, recursive: true);
            }
            catch (IOException)
            {
            }
        }
    }

    /// <summary>
    /// Tracking unless told otherwise. The failure it prevents — one voice split into nineteen —
    /// is worse than the one it causes, so an install that has never chosen gets it.
    /// </summary>
    [Fact]
    public void TrackingIsWhatYouGetWithoutChoosing()
    {
        using var scratch = new Scratch();

        Assert.Equal(DiarizationMethod.Tracking, DiarizationChoice.Read(scratch.Path));
    }

    [Fact]
    public void AChoiceSurvivesBeingWrittenAndReadBack()
    {
        using var scratch = new Scratch();

        DiarizationChoice.Write(scratch.Path, DiarizationMethod.Voices);

        Assert.Equal(DiarizationMethod.Voices, DiarizationChoice.Read(scratch.Path));
    }

    [Fact]
    public void SwitchingBackWorks()
    {
        using var scratch = new Scratch();

        DiarizationChoice.Write(scratch.Path, DiarizationMethod.Voices);
        DiarizationChoice.Write(scratch.Path, DiarizationMethod.Tracking);

        Assert.Equal(DiarizationMethod.Tracking, DiarizationChoice.Read(scratch.Path));
    }

    /// <summary>
    /// A file nobody can parse must not stop a transcription. Falling back costs accuracy on some
    /// recordings; failing costs the whole run.
    /// </summary>
    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("nonsense")]
    [InlineData("Voices\n")]
    public void AnythingUnreadableFallsBackRatherThanFailing(string written)
    {
        using var scratch = new Scratch();

        File.WriteAllText(Path.Combine(scratch.Path, DiarizationChoice.FileName), written);

        var read = DiarizationChoice.Read(scratch.Path);

        Assert.Equal(
            written.Trim().Equals("voices", StringComparison.OrdinalIgnoreCase)
                ? DiarizationMethod.Voices
                : DiarizationMethod.Tracking,
            read);
    }

    [Fact]
    public void TheNamesAreTheOnesTheCommandTakes()
    {
        Assert.Equal("tracking", DiarizationChoice.Name(DiarizationMethod.Tracking));
        Assert.Equal("voices", DiarizationChoice.Name(DiarizationMethod.Voices));
    }
}
