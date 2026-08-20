using LocalScribe.Core.Audio;
using LocalScribe.Core.Pipeline;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// What happens when audio and shutdown arrive at the same time.
/// <para>
/// Separate from the happy-path tests because these need a pass that can be held open. The
/// failure they cover is only reachable while one is mid-flight, and a transcriber that returns
/// immediately never leaves that window open.
/// </para>
/// </summary>
public sealed class LiveTranscriptionShutdownTests
{
    /// <summary>A transcriber returning a scripted sequence of readings, one per pass.</summary>
    private sealed class ScriptedTranscriber(params string[] readings) : ITranscriber
    {
        private int _pass;

        public string Description => "scripted";

        public Task<IReadOnlyList<TranscriptSegment>> TranscribeChunkAsync(
            AudioChunk chunk,
            CancellationToken cancellationToken = default)
        {
            var index = Math.Min(_pass++, readings.Length - 1);

            return Task.FromResult<IReadOnlyList<TranscriptSegment>>(
                [new TranscriptSegment(readings[index], chunk.StartSeconds, chunk.StartSeconds + 2)]);
        }

        public void Dispose()
        {
        }
    }

    /// <summary>
    /// The reported failure, at the level it actually occurs. Whisper delivers the same speech
    /// formatted on one pass and as a bare lowercase run of words on the next, and the final
    /// pass is the one that reaches the transcript. Arriving last is not a reason to win.
    /// </summary>
    [Fact]
    public async Task ADegeneratePassDoesNotOverwriteAFormattedOne()
    {
        const string formatted = "Okay, here we go. Let's try the transcription again.";
        const string degenerate = "okay here we go let's try the transcription again";

        using var transcriber = new ScriptedTranscriber(formatted, degenerate);
        await using var session = new LiveTranscriptionSession(transcriber);

        await session.PushAsync(Audio(2));   // formatted
        await session.PushAsync(Audio(2));   // degenerate

        var committed = await session.FinishAsync();
        var text = string.Join(" ", committed.Select(s => s.Text));

        Assert.Contains("Okay, here we go.", text, StringComparison.Ordinal);
        Assert.DoesNotContain("okay here we go", text, StringComparison.Ordinal);
    }

    /// <summary>A transcriber whose pass blocks until released.</summary>
    private sealed class HoldableTranscriber : ITranscriber
    {
        private readonly SemaphoreSlim _release = new(0);

        public string Description => "holdable";

        public bool HoldPasses { get; init; }

        public int Started;

        public async Task<IReadOnlyList<TranscriptSegment>> TranscribeChunkAsync(
            AudioChunk chunk,
            CancellationToken cancellationToken = default)
        {
            Interlocked.Increment(ref Started);

            if (HoldPasses)
            {
                await _release.WaitAsync(cancellationToken).ConfigureAwait(false);
            }

            return [new TranscriptSegment("hello world", chunk.StartSeconds, chunk.StartSeconds + 1)];
        }

        public void ReleaseOne() => _release.Release();

        public void Dispose() => _release.Dispose();
    }

    private static float[] Audio(double seconds) =>
        new float[(int)(seconds * PcmAudio.WhisperSampleRate)];

    private static async Task WaitForPassToStart(HoldableTranscriber transcriber)
    {
        while (Volatile.Read(ref transcriber.Started) == 0)
        {
            await Task.Delay(5);
        }
    }

    /// <summary>
    /// The reported failure. Buffers already captured still raise their event after stop, and a
    /// handler that is mid-await is still running, so pushes land after disposal. That used to
    /// hit a disposed semaphore and surface to the user as "Live transcription failed: Cannot
    /// access a disposed object".
    /// </summary>
    [Fact]
    public async Task AudioArrivingAfterDisposalIsIgnoredRatherThanThrowing()
    {
        using var transcriber = new HoldableTranscriber();
        var session = new LiveTranscriptionSession(transcriber);

        await session.PushAsync(Audio(2));
        await session.DisposeAsync();

        Assert.Null(await session.PushAsync(Audio(2)));
        Assert.Null(await session.PushAsync(Audio(2)));
    }

    [Fact]
    public async Task DisposingTwiceIsHarmless()
    {
        using var transcriber = new HoldableTranscriber();
        var session = new LiveTranscriptionSession(transcriber);

        await session.PushAsync(Audio(2));
        await session.DisposeAsync();
        await session.DisposeAsync();
    }

    /// <summary>
    /// Disposal must not walk away from a pass that is reading the window and appending to the
    /// committed list.
    /// </summary>
    [Fact]
    public async Task DisposalWaitsForAPassAlreadyRunning()
    {
        using var transcriber = new HoldableTranscriber { HoldPasses = true };
        var session = new LiveTranscriptionSession(transcriber);

        var pass = session.PushAsync(Audio(2));
        await WaitForPassToStart(transcriber);

        var disposal = session.DisposeAsync().AsTask();

        Assert.False(disposal.IsCompleted, "Disposal returned while a pass was still running.");

        transcriber.ReleaseOne();

        await pass;
        await disposal;
    }

    /// <summary>
    /// Finish waits its turn rather than skipping. A pass is very likely running at the moment
    /// the user hits stop, and skipping drops exactly the trailing words Finish exists to keep.
    /// </summary>
    [Fact]
    public async Task FinishCommitsEvenWhenAPassIsAlreadyRunning()
    {
        using var transcriber = new HoldableTranscriber { HoldPasses = true };
        var session = new LiveTranscriptionSession(transcriber);

        var pass = session.PushAsync(Audio(2));
        await WaitForPassToStart(transcriber);

        var finish = session.FinishAsync();

        Assert.False(finish.IsCompleted, "Finish skipped instead of waiting for the running pass.");

        transcriber.ReleaseOne();   // release the streaming pass
        await pass;
        transcriber.ReleaseOne();   // release Finish's own pass

        Assert.NotEmpty(await finish);

        await session.DisposeAsync();
    }

    /// <summary>
    /// Audio arrives on the capture thread while a pass reads the window on a worker. A
    /// List&lt;float&gt; touched from both without a lock corrupts rather than failing cleanly.
    /// </summary>
    [Fact]
    public async Task ConcurrentPushesDoNotCorruptTheWindow()
    {
        using var transcriber = new HoldableTranscriber();
        await using var session = new LiveTranscriptionSession(transcriber);

        var pushes = Enumerable.Range(0, 64)
            .Select(_ => Task.Run(async () => await session.PushAsync(Audio(0.5))))
            .ToArray();

        await Task.WhenAll(pushes);

        Assert.NotNull(await session.FinishAsync());
    }
}
