using LocalScribe.Core.Archive;
using LocalScribe.Core.Audio;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class TranscriptArchiveTests
{
    private static PcmAudio Tone(double seconds = 2.0, int sampleRate = 16000)
    {
        var samples = new float[(int)(seconds * sampleRate)];
        for (var i = 0; i < samples.Length; i++)
        {
            samples[i] = (float)(Math.Sin(2 * Math.PI * 440 * i / sampleRate) * 0.5);
        }

        return new PcmAudio(samples, sampleRate);
    }

    private static IReadOnlyList<TranscriptSegment> Segments() =>
    [
        new("Morning, shall we start with the budget?", 0.0, 2.4, -0.2, 0.01, "Kim"),
        new("Yes, I looked at the numbers last night.", 2.6, 5.1, -0.3, 0.02, "Sam"),
        new("And they seem reasonable to me.", 5.1, 7.0, -0.1, 0.00, "Sam"),
    ];

    private static TranscriptArchive.Contents RoundTrip(
        PcmAudio audio,
        IReadOnlyList<TranscriptSegment> segments,
        string name = "Team catch-up")
    {
        using var stream = new MemoryStream();
        TranscriptArchive.Write(stream, audio, segments, name);

        stream.Position = 0;
        return TranscriptArchive.Read(stream);
    }

    [Fact]
    public void TheWordsAndTheirTimingsSurvive()
    {
        var result = RoundTrip(Tone(), Segments());

        Assert.Equal(3, result.Segments.Count);
        Assert.Equal("Morning, shall we start with the budget?", result.Segments[0].Text);
        Assert.Equal(2.6, result.Segments[1].StartSeconds, 3);
        Assert.Equal(7.0, result.Segments[2].EndSeconds, 3);
    }

    /// <summary>
    /// The point of saving them together. Without the speakers a reader cannot tell who said
    /// what, and without the timings nothing can be clicked to hear it again.
    /// </summary>
    [Fact]
    public void SpeakersAndConfidencesSurvive()
    {
        var result = RoundTrip(Tone(), Segments());

        Assert.Equal("Kim", result.Segments[0].Speaker);
        Assert.Equal("Sam", result.Segments[1].Speaker);
        Assert.Equal(-0.2, result.Segments[0].AverageLogProbability, 3);
        Assert.Equal(0.02, result.Segments[1].NoSpeechProbability, 3);
    }

    [Fact]
    public void TheRecordingSurvives()
    {
        var original = Tone();
        var result = RoundTrip(original, Segments());

        Assert.Equal(original.SampleRate, result.Audio.SampleRate);
        Assert.Equal(original.Samples.Length, result.Audio.Samples.Length);

        // Sixteen-bit quantisation, so near enough rather than identical.
        for (var i = 0; i < original.Samples.Length; i += 97)
        {
            Assert.Equal(original.Samples[i], result.Audio.Samples[i], 0.001);
        }
    }

    [Fact]
    public void TheManifestSaysWhatIsInside()
    {
        var result = RoundTrip(Tone(3.0), Segments(), "Budget call");

        Assert.Equal(TranscriptArchive.CurrentVersion, result.Manifest.Version);
        Assert.Equal("Budget call", result.Manifest.SourceName);
        Assert.Equal(3, result.Manifest.SegmentCount);
        Assert.Equal(2, result.Manifest.SpeakerCount);
        Assert.Equal(3.0, result.Manifest.DurationSeconds, 2);
    }

    /// <summary>
    /// An archive from a future version must announce itself rather than be half-read. Silently
    /// dropping whatever a newer build added is how a file loses the part that mattered.
    /// </summary>
    [Fact]
    public void AnArchiveFromANewerVersionIsRefusedPlainly()
    {
        using var stream = new MemoryStream();
        TranscriptArchive.Write(stream, Tone(), Segments(), "Later");

        // Rewrite the manifest as if a newer build had made it.
        stream.Position = 0;
        using var zip = new System.IO.Compression.ZipArchive(stream, System.IO.Compression.ZipArchiveMode.Update, leaveOpen: true);
        var entry = zip.GetEntry("manifest.json")!;

        string text;
        using (var reader = new StreamReader(entry.Open()))
        {
            text = reader.ReadToEnd();
        }

        entry.Delete();
        using (var writer = new StreamWriter(zip.CreateEntry("manifest.json").Open()))
        {
            writer.Write(text.Replace("\"version\": 1", "\"version\": 99"));
        }

        zip.Dispose();
        stream.Position = 0;

        var failure = Assert.Throws<InvalidDataException>(() => TranscriptArchive.Read(stream));
        Assert.Contains("newer version", failure.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void AnEmptyTranscriptIsStillAValidArchive()
    {
        var result = RoundTrip(Tone(1.0), []);

        Assert.Empty(result.Segments);
        Assert.Equal(0, result.Manifest.SegmentCount);
        Assert.NotEmpty(result.Audio.Samples);
    }

    [Fact]
    public void SomethingThatIsNotAnArchiveIsRejected()
    {
        using var stream = new MemoryStream("not a zip, not even close"u8.ToArray());

        Assert.ThrowsAny<Exception>(() => TranscriptArchive.Read(stream));
    }
}
