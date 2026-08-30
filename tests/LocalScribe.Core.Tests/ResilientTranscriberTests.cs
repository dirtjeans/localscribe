using LocalScribe.Core.Audio;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class ResilientTranscriberTests
{
    private sealed class FlakyEngine(int failuresBeforeWorking) : ITranscriber
    {
        private int _failuresLeft = failuresBeforeWorking;

        public string Description => "flaky";
        public SpeechTask? BeganWith { get; private set; }
        public bool Disposed { get; private set; }

        public void BeginRecording(SpeechTask task = SpeechTask.Transcribe) => BeganWith = task;

        public Task<IReadOnlyList<TranscriptSegment>> TranscribeChunkAsync(
            AudioChunk chunk, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (_failuresLeft-- > 0)
            {
                // The costume the real failure wore: a cancellation nobody asked for.
                throw new OperationCanceledException("engine gave up");
            }

            return Task.FromResult<IReadOnlyList<TranscriptSegment>>(
                [new TranscriptSegment("heard", chunk.StartSeconds, chunk.StartSeconds + 1)]);
        }

        public void Dispose() => Disposed = true;
    }

    private static AudioChunk Chunk() => new(new float[16000], 0, 1);

    /// <summary>
    /// The point of the class: one engine failure costs a restart and a retry of the same
    /// window, not the recording.
    /// </summary>
    [Fact]
    public async Task AFailedWindowIsRetriedOnAFreshEngine()
    {
        var engines = new Queue<ITranscriber>([new FlakyEngine(1), new FlakyEngine(0)]);
        var transcriber = new ResilientTranscriber(engines.Dequeue);

        var segments = await transcriber.TranscribeChunkAsync(Chunk());

        Assert.Single(segments);
        Assert.Equal("heard", segments[0].Text);
        Assert.Equal(1, transcriber.Restarts);
        Assert.NotNull(transcriber.LastFailure);
    }

    /// <summary>
    /// A cancellation the user asked for is a decision, not a failure, and restarting the
    /// engine to defeat it would turn the stop button into a suggestion.
    /// </summary>
    [Fact]
    public async Task ARealCancellationIsNotRetried()
    {
        var opened = 0;
        var transcriber = new ResilientTranscriber(() =>
        {
            opened++;
            return new FlakyEngine(0);
        });

        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => transcriber.TranscribeChunkAsync(Chunk(), cancelled.Token));

        Assert.Equal(1, opened);
        Assert.Equal(0, transcriber.Restarts);
    }

    /// <summary>
    /// Two engines agreeing something is wrong is systemic. Skipping the window instead would
    /// silently drop speech, which is the worse lie.
    /// </summary>
    [Fact]
    public async Task AWindowThatFailsTheFreshEngineTooFailsHonestly()
    {
        var transcriber = new ResilientTranscriber(() => new FlakyEngine(9));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => transcriber.TranscribeChunkAsync(Chunk()));

        Assert.Equal(1, transcriber.Restarts);
    }

    /// <summary>
    /// The replacement engine must resume the recording's own task, or the transcript would
    /// change language convention at the restart.
    /// </summary>
    [Fact]
    public async Task TheReplacementResumesTheSameTask()
    {
        var engines = new Queue<FlakyEngine>([new FlakyEngine(1), new FlakyEngine(0)]);
        var built = new List<FlakyEngine>();

        var transcriber = new ResilientTranscriber(() =>
        {
            var engine = engines.Dequeue();
            built.Add(engine);
            return engine;
        });

        transcriber.BeginRecording(SpeechTask.TranslateToEnglish);
        await transcriber.TranscribeChunkAsync(Chunk());

        Assert.True(built[0].Disposed);
        Assert.Equal(SpeechTask.TranslateToEnglish, built[1].BeganWith);
    }
}
