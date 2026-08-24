using LocalScribe.Core.Audio;
using LocalScribe.Core.Pipeline;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// A recording must not inherit what was learned from the one before it.
/// <para>
/// Whisper settles the language once and then asserts it on every window, which is right within
/// a recording and wrong across two. The transcriber is loaded once and reused, so left to
/// itself it carries English from one file into the next — and Whisper asked to transcribe
/// Portuguese as English does not refuse, it translates.
/// </para>
/// </summary>
public class RecordingBoundaryTests
{
    private sealed class CountingTranscriber : ITranscriber
    {
        public int Beginnings { get; private set; }

        public SpeechTask LastTask { get; private set; } = SpeechTask.Transcribe;

        public int Windows { get; private set; }

        public string Description => "Counting";

        public void BeginRecording(SpeechTask task = SpeechTask.Transcribe)
        {
            Beginnings++;
            LastTask = task;
        }

        public Task<IReadOnlyList<TranscriptSegment>> TranscribeChunkAsync(
            AudioChunk chunk,
            CancellationToken cancellationToken = default)
        {
            Windows++;

            IReadOnlyList<TranscriptSegment> segments =
                [new TranscriptSegment("hello", chunk.StartSeconds, chunk.StartSeconds + 1)];

            return Task.FromResult(segments);
        }

        public void Dispose()
        {
        }
    }

    private static PcmAudio Silence(double seconds) =>
        new(new float[(int)(seconds * PcmAudio.WhisperSampleRate)], PcmAudio.WhisperSampleRate);

    [Fact]
    public async Task EachRunIsToldItIsANewRecording()
    {
        var transcriber = new CountingTranscriber();
        var pipeline = new TranscriptionPipeline(transcriber);

        await pipeline.RunAsync(Silence(3));
        await pipeline.RunAsync(Silence(3));

        Assert.Equal(2, transcriber.Beginnings);
    }

    /// <summary>Once per recording, not once per window: within one it must be left alone.</summary>
    [Fact]
    public async Task ItIsNotSaidAgainForEveryWindow()
    {
        var transcriber = new CountingTranscriber();
        var pipeline = new TranscriptionPipeline(transcriber);

        await pipeline.RunAsync(Silence(90));

        Assert.True(transcriber.Windows > 1, "the recording should span more than one window");
        Assert.Equal(1, transcriber.Beginnings);
    }

    [Fact]
    public void ALiveSessionIsARecordingToo()
    {
        var transcriber = new CountingTranscriber();

        _ = new LiveTranscriptionSession(transcriber);

        Assert.Equal(1, transcriber.Beginnings);
    }

    /// <summary>Writing it down as spoken is what happens unless asked otherwise.</summary>
    [Fact]
    public async Task TranscribingIsTheDefault()
    {
        var transcriber = new CountingTranscriber();

        await new TranscriptionPipeline(transcriber).RunAsync(Silence(3));

        Assert.Equal(SpeechTask.Transcribe, transcriber.LastTask);
    }

    /// <summary>And the other way round only when it is.</summary>
    [Fact]
    public async Task TranslationIsCarriedToTheBackend()
    {
        var transcriber = new CountingTranscriber();

        await new TranscriptionPipeline(transcriber)
            .RunAsync(Silence(3), task: SpeechTask.TranslateToEnglish);

        Assert.Equal(SpeechTask.TranslateToEnglish, transcriber.LastTask);
    }

    /// <summary>
    /// The app transcribes through this method rather than through RunAsync, so the boundary has
    /// to be on it. Put on RunAsync alone, the fix for a recording inheriting the last one's
    /// language would never have run in the app at all.
    /// </summary>
    [Fact]
    public async Task TranscribingDirectlyIsStillARecording()
    {
        var transcriber = new CountingTranscriber();

        await new TranscriptionPipeline(transcriber).TranscribeAsync(Silence(3));

        Assert.Equal(1, transcriber.Beginnings);
    }
}
