using LocalScribe.Core.Audio;
using LocalScribe.Core.Pipeline;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// Where the next window starts, and why it is not a fixed stride.
/// </summary>
public class WindowSeekTests
{
    /// <summary>
    /// Transcribes a fixed span of each window and records where it was asked to look. Real
    /// Whisper stops wherever it finds a natural end, which is routinely well before the end of
    /// the window; this reproduces that without a model.
    /// </summary>
    private sealed class StopsEarlyTranscriber(double transcribedSeconds) : ITranscriber
    {
        public List<double> WindowStarts { get; } = [];

        public string Description => "stops early";

        public Task<IReadOnlyList<TranscriptSegment>> TranscribeChunkAsync(
            AudioChunk chunk,
            CancellationToken cancellationToken = default)
        {
            WindowStarts.Add(chunk.StartSeconds);

            var end = Math.Min(transcribedSeconds, chunk.ContentSeconds);
            if (end <= 0)
            {
                return Task.FromResult<IReadOnlyList<TranscriptSegment>>([]);
            }

            return Task.FromResult<IReadOnlyList<TranscriptSegment>>(
            [
                new TranscriptSegment(
                    $"heard {chunk.StartSeconds:F0}",
                    chunk.StartSeconds,
                    chunk.StartSeconds + end),
            ]);
        }

        public void Dispose()
        {
        }
    }

    private static PcmAudio Silence(double seconds) =>
        new(new float[(int)(seconds * PcmAudio.WhisperSampleRate)]);

    /// <summary>
    /// The failure this exists to prevent. A window transcribed only to 25.8s of its 30, then a
    /// fixed 28s stride, leaves 25.8 to 28 heard by nobody — which on a real recording removed
    /// "the segment. If the stitching" from the middle of a sentence.
    /// </summary>
    [Fact]
    public async Task NoAudioIsSkippedWhenTheModelStopsEarly()
    {
        using var transcriber = new StopsEarlyTranscriber(transcribedSeconds: 25.8);
        var chunker = new AudioChunker(overlapSeconds: 2.0);

        await new TranscriptionPipeline(transcriber, chunker: chunker).RunAsync(Silence(63));

        // Each window must begin no later than where the previous one stopped hearing.
        for (var i = 1; i < transcriber.WindowStarts.Count; i++)
        {
            var previousHeardTo = transcriber.WindowStarts[i - 1] + 25.8;

            Assert.True(
                transcriber.WindowStarts[i] <= previousHeardTo,
                $"Window {i} starts at {transcriber.WindowStarts[i]:F2}s but the one before it "
                + $"stopped hearing at {previousHeardTo:F2}s, so that gap is transcribed by nobody.");
        }
    }

    [Fact]
    public async Task TheWholeRecordingIsCovered()
    {
        using var transcriber = new StopsEarlyTranscriber(transcribedSeconds: 25.8);

        await new TranscriptionPipeline(transcriber).RunAsync(Silence(63));

        Assert.True(transcriber.WindowStarts[^1] + 30 >= 63, "The last window stops short of the audio.");
    }

    /// <summary>
    /// A window that hears nothing gives nothing to seek to, so it has to fall back to a stride
    /// rather than sit still.
    /// </summary>
    [Fact]
    public async Task SilenceStillMakesProgress()
    {
        using var transcriber = new StopsEarlyTranscriber(transcribedSeconds: 0);

        await new TranscriptionPipeline(transcriber).RunAsync(Silence(90));

        Assert.True(transcriber.WindowStarts.Count is > 1 and < 20);
        Assert.True(transcriber.WindowStarts.Zip(transcriber.WindowStarts.Skip(1)).All(p => p.Second > p.First));
    }

    /// <summary>
    /// A model that reports a timestamp past the audio it was given is describing its own
    /// padding. Trusting it would seek beyond real speech and skip it.
    /// </summary>
    [Fact]
    public async Task ATimestampBeyondTheAudioDoesNotSkipAhead()
    {
        using var transcriber = new StopsEarlyTranscriber(transcribedSeconds: 30);
        var chunker = new AudioChunker(overlapSeconds: 2.0);

        await new TranscriptionPipeline(transcriber, chunker: chunker).RunAsync(Silence(40));

        // The second window covers 28-40; a claim to have heard to 58 must not move the third
        // window past the end of the recording's real audio.
        Assert.All(transcriber.WindowStarts, start => Assert.True(start <= 40));
    }

    [Fact]
    public async Task ShortAudioIsOneWindow()
    {
        using var transcriber = new StopsEarlyTranscriber(transcribedSeconds: 5);

        await new TranscriptionPipeline(transcriber).RunAsync(Silence(6));

        Assert.Single(transcriber.WindowStarts);
    }
}
