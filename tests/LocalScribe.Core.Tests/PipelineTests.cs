using LocalScribe.Core.Audio;
using LocalScribe.Core.Pipeline;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>A transcriber that returns scripted results, so the orchestration can be tested alone.</summary>
public class StreamedStampTests
{
    /// <summary>
    /// Streamed text must carry the window's real time range. Windows overlap, so counting
    /// windows and multiplying by the window length overstates position by the overlap per
    /// window — ninety seconds by the end of a twenty-one minute recording, which pushed
    /// every streamed anchor out of the aligner's reach while a run was in progress.
    /// </summary>
    [Xunit.Fact]
    public async System.Threading.Tasks.Task ProgressCarriesTheWindowsRealRange()
    {
        var audio = new LocalScribe.Core.Audio.PcmAudio(
            new float[16000 * 70], 16000);

        // Recorded synchronously: Progress<T> posts through the thread pool, and waiting for
        // it to land made this test fail once under load for no reason of its own.
        var reports = new System.Collections.Generic.List<TranscriptionProgress>();
        var progress = new InlineProgress<TranscriptionProgress>(reports.Add);
        var transcriber = new FakeTranscriber((_, i) => [new TranscriptSegment($"w{i}", 0, 1)]);

        await new TranscriptionPipeline(transcriber).TranscribeAsync(audio, progress);

        var spoken = reports.Where(r => r.LatestText.Length > 0).ToList();

        Xunit.Assert.True(spoken.Count >= 2);

        // The second window begins at the stride (window minus overlap), not at the window
        // length — 28 seconds with the defaults, not 30.
        Xunit.Assert.True(spoken[1].LatestStartSeconds < LocalScribe.Core.Audio.AudioChunker.WindowSeconds);
        Xunit.Assert.True(spoken[1].LatestStartSeconds > 0);
        Xunit.Assert.True(spoken[1].LatestEndSeconds <= audio.DurationSeconds + 0.01);
    }
}

/// <summary>Reports on the calling thread, so a test can read them the moment the call returns.</summary>
internal sealed class InlineProgress<T>(System.Action<T> report) : System.IProgress<T>
{
    public void Report(T value) => report(value);
}

internal sealed class FakeTranscriber : ITranscriber
{
    private readonly Func<AudioChunk, int, IReadOnlyList<TranscriptSegment>> _respond;
    private int _callCount;

    public FakeTranscriber(Func<AudioChunk, int, IReadOnlyList<TranscriptSegment>> respond)
    {
        _respond = respond;
    }

    public string Description => "fake";

    public int CallCount => _callCount;

    public Task<IReadOnlyList<TranscriptSegment>> TranscribeChunkAsync(
        AudioChunk chunk,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(_respond(chunk, _callCount++));
    }

    public void Dispose()
    {
    }
}

public sealed class TranscriptStitcherTests
{
    [Fact]
    public void OverlappingWindowsDoNotRepeatTheSameWords()
    {
        // The chunker overlaps by design, so both windows see the words at the boundary.
        var stitched = new TranscriptStitcher().Stitch(
        [
            [new TranscriptSegment("the meeting starts at ten", 0, 3)],
            [
                new TranscriptSegment("the meeting starts at ten", 0, 3),
                new TranscriptSegment("and runs for an hour", 3, 6),
            ],
        ]);

        Assert.Equal(2, stitched.Count);
        Assert.Equal("the meeting starts at ten and runs for an hour", new Transcript(stitched).FullText);
    }

    [Fact]
    public void PunctuationAndCasingDifferencesStillCountAsDuplicates()
    {
        // The two passes over an overlap rarely punctuate identically.
        var stitched = new TranscriptStitcher().Stitch(
        [
            [new TranscriptSegment("Right, so — we ship on Friday.", 0, 3)],
            [new TranscriptSegment("right so we ship on friday", 0.2, 3.1)],
        ]);

        Assert.Single(stitched);
    }

    [Fact]
    public void APhraseRepeatedMuchLaterIsKept()
    {
        // Genuine repetition is not a duplicate. Only near-boundary repeats are.
        var stitched = new TranscriptStitcher(boundaryToleranceSeconds: 2.5).Stitch(
        [
            [new TranscriptSegment("any questions", 0, 2)],
            [new TranscriptSegment("any questions", 600, 602)],
        ]);

        Assert.Equal(2, stitched.Count);
    }

    [Fact]
    public void HallucinatedSegmentsAreDropped()
    {
        var stitched = new TranscriptStitcher().Stitch(
        [
            [
                new TranscriptSegment("real speech here", 0, 2, AverageLogProbability: -0.2),
                new TranscriptSegment(
                    "Thank you for watching!",
                    2,
                    4,
                    AverageLogProbability: -1.8,
                    NoSpeechProbability: 0.95),
            ],
        ]);

        var segment = Assert.Single(stitched);
        Assert.Equal("real speech here", segment.Text);
    }

    [Fact]
    public void EmptySegmentsAreDropped()
    {
        var stitched = new TranscriptStitcher().Stitch(
            [[new TranscriptSegment("   ", 0, 2), new TranscriptSegment("words", 2, 4)]]);

        Assert.Single(stitched);
    }

    /// <summary>
    /// Windows arrive in order and the decoder writes in order, so the order written is the
    /// order said; a stamp that disagrees is the weaker evidence and is moved, not obeyed.
    /// </summary>
    [Fact]
    public void OutputKeepsTheOrderItWasWrittenIn()
    {
        var stitched = new TranscriptStitcher().Stitch(
        [
            [new TranscriptSegment("first", 10, 12)],
            [new TranscriptSegment("second", 9, 14)],
        ]);

        Assert.Equal(["first", "second"], stitched.Select(s => s.Text));
        Assert.True(stitched[1].StartSeconds >= stitched[0].EndSeconds);
    }

    [Fact]
    public void NoInputProducesAnEmptyTranscript()
    {
        Assert.Empty(new TranscriptStitcher().Stitch([]));
    }
}

public sealed class TranscriptionPipelineTests
{
    private static PcmAudio Audio(double seconds) =>
        new(new float[(int)(seconds * PcmAudio.WhisperSampleRate)]);

    [Fact]
    public async Task RunsEveryWindowAndJoinsTheResults()
    {
        var transcriber = new FakeTranscriber((chunk, index) =>
            [new TranscriptSegment($"window {index}", chunk.StartSeconds, chunk.StartSeconds + 1)]);

        var result = await new TranscriptionPipeline(transcriber).TranscribeAsync(Audio(90));

        Assert.Equal(4, transcriber.CallCount);
        Assert.Contains("window 0", result.FullText, StringComparison.Ordinal);
        Assert.Contains("window 3", result.FullText, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ProgressIsReportedForEveryWindow()
    {
        var reports = new List<TranscriptionProgress>();
        var progress = new Progress<TranscriptionProgress>(reports.Add);
        var transcriber = new FakeTranscriber((_, i) => [new TranscriptSegment($"w{i}", 0, 1)]);

        await new TranscriptionPipeline(transcriber).TranscribeAsync(Audio(90), progress: progress);

        // Progress<T> marshals through the synchronisation context, so allow it to drain.
        await Task.Delay(100);

        Assert.NotEmpty(reports);
        Assert.Equal(1.0, reports[^1].Fraction, precision: 3);
    }

    [Fact]
    public async Task PaddingOnlyWindowsAreNeverSentToTheModel()
    {
        // Asking the model about silence is how transcripts acquire "Thank you for watching".
        var transcriber = new FakeTranscriber((_, _) => [new TranscriptSegment("invented", 0, 1)]);
        var chunker = new AudioChunker(overlapSeconds: 0);

        await new TranscriptionPipeline(transcriber, chunker: chunker).TranscribeAsync(Audio(30.5));

        Assert.Equal(1, transcriber.CallCount);
    }

    [Fact]
    public async Task CancellationStopsTheRun()
    {
        using var cts = new CancellationTokenSource();
        var transcriber = new FakeTranscriber((_, index) =>
        {
            if (index == 1)
            {
                cts.Cancel();
            }

            return [new TranscriptSegment($"w{index}", 0, 1)];
        });

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            new TranscriptionPipeline(transcriber).TranscribeAsync(Audio(300), cancellationToken: cts.Token));
    }

    [Fact]
    public async Task WrongSampleRateFailsBeforeAnyWorkHappens()
    {
        var transcriber = new FakeTranscriber((_, _) => []);

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            new TranscriptionPipeline(transcriber).TranscribeAsync(new PcmAudio(new float[16000], 8000)));

        Assert.Equal(0, transcriber.CallCount);
    }
}

public sealed class LiveTranscriptionSessionTests
{
    private static float[] Seconds(double seconds) =>
        new float[(int)(seconds * PcmAudio.WhisperSampleRate)];

    [Fact]
    public async Task SmallPushesAccumulateBeforeTriggeringAPass()
    {
        var transcriber = new FakeTranscriber((_, _) => []);
        await using var session = new LiveTranscriptionSession(transcriber);

        var update = await session.PushAsync(Seconds(0.2));

        Assert.Null(update);
        Assert.Equal(0, transcriber.CallCount);
    }

    [Fact]
    public async Task EnoughAudioTriggersAPass()
    {
        var transcriber = new FakeTranscriber((_, _) => [new TranscriptSegment("hello", 0, 1)]);
        await using var session = new LiveTranscriptionSession(transcriber);

        var update = await session.PushAsync(Seconds(1.5));

        Assert.NotNull(update);
        Assert.Equal(1, transcriber.CallCount);
    }

    [Fact]
    public async Task RecentSpeechStaysProvisionalRatherThanBeingCommittedEarly()
    {
        // The model cannot know where a sentence is heading, so the tail must stay revisable.
        var transcriber = new FakeTranscriber((_, _) => [new TranscriptSegment("still talking", 0, 2)]);
        await using var session = new LiveTranscriptionSession(transcriber);

        var update = await session.PushAsync(Seconds(2));

        Assert.NotNull(update);
        Assert.False(update.IsFinal);
        Assert.Equal("still talking", update.Text);
        Assert.Empty(session.CommittedSegments);
    }

    [Fact]
    public async Task SpeechOlderThanTheCommitWindowIsCommitted()
    {
        var transcriber = new FakeTranscriber((_, _) => [new TranscriptSegment("settled words", 0, 2)]);
        await using var session = new LiveTranscriptionSession(transcriber);

        // Push well past the commit horizon so the early span can no longer change.
        await session.PushAsync(Seconds(12));

        Assert.Single(session.CommittedSegments);
        Assert.Equal("settled words", session.CommittedSegments[0].Text);
    }

    [Fact]
    public async Task RepeatedPassesDoNotCommitTheSameWordsTwice()
    {
        // Every pass re-transcribes the whole window, so most output is already known.
        var transcriber = new FakeTranscriber((_, _) => [new TranscriptSegment("settled words", 0, 2)]);
        await using var session = new LiveTranscriptionSession(transcriber);

        await session.PushAsync(Seconds(12));
        await session.PushAsync(Seconds(2));
        await session.PushAsync(Seconds(2));

        Assert.Single(session.CommittedSegments);
    }

    [Fact]
    public async Task FinishingCommitsTheTrailingWords()
    {
        // Without this, the last thing the user said never reaches the transcript.
        var transcriber = new FakeTranscriber((_, _) => [new TranscriptSegment("final thought", 0, 2)]);
        await using var session = new LiveTranscriptionSession(transcriber);

        await session.PushAsync(Seconds(2));
        Assert.Empty(session.CommittedSegments);

        var committed = await session.FinishAsync();

        Assert.Single(committed);
        Assert.Equal("final thought", committed[0].Text);
    }

    [Fact]
    public async Task HallucinationsAreNeverCommitted()
    {
        var transcriber = new FakeTranscriber((_, _) =>
        [
            new TranscriptSegment("Thank you for watching!", 0, 2, -1.9, 0.97),
        ]);
        await using var session = new LiveTranscriptionSession(transcriber);

        await session.PushAsync(Seconds(12));
        var committed = await session.FinishAsync();

        Assert.Empty(committed);
    }

    [Fact]
    public async Task LongSessionsDoNotGrowTheAudioBufferWithoutBound()
    {
        // A two-hour recording must not accumulate two hours of samples in memory.
        var transcriber = new FakeTranscriber((chunk, _) =>
        {
            Assert.Equal(
                (int)(LiveTranscriptionSession.WindowSeconds * PcmAudio.WhisperSampleRate),
                chunk.Samples.Length);
            return [];
        });

        await using var session = new LiveTranscriptionSession(transcriber);

        for (var i = 0; i < 60; i++)
        {
            await session.PushAsync(Seconds(2));
        }

        Assert.Equal(60, transcriber.CallCount);
    }
}
