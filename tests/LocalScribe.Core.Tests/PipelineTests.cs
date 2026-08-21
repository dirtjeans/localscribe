using LocalScribe.Core.Audio;
using LocalScribe.Core.Pipeline;
using LocalScribe.Core.Refinement;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>A transcriber that returns scripted results, so the orchestration can be tested alone.</summary>
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

/// <summary>A language model that echoes a scripted transformation.</summary>
internal sealed class FakeLanguageModel : ILanguageModel
{
    private readonly Func<string, string, string> _respond;

    public FakeLanguageModel(Func<string, string, string>? respond = null)
    {
        _respond = respond ?? ((_, user) => user.ToUpperInvariant());
    }

    public List<string> SystemPrompts { get; } = [];

    public string Description => "fake model";

    public Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default) => Task.FromResult(true);

    public Task<string> CompleteAsync(
        string systemPrompt,
        string userPrompt,
        int maxTokens = 1024,
        CancellationToken cancellationToken = default)
    {
        SystemPrompts.Add(systemPrompt);
        return Task.FromResult(_respond(systemPrompt, userPrompt));
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

    [Fact]
    public void OutputIsOrderedByTime()
    {
        var stitched = new TranscriptStitcher().Stitch(
        [
            [new TranscriptSegment("second", 10, 12)],
            [new TranscriptSegment("first", 0, 2)],
        ]);

        Assert.Equal("first", stitched[0].Text);
        Assert.Equal("second", stitched[1].Text);
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

        var result = await new TranscriptionPipeline(transcriber).RunAsync(Audio(90));

        Assert.Equal(4, transcriber.CallCount);
        Assert.Contains("window 0", result.Transcript.FullText, StringComparison.Ordinal);
        Assert.Contains("window 3", result.Transcript.FullText, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ProgressIsReportedForEveryWindow()
    {
        var reports = new List<TranscriptionProgress>();
        var progress = new Progress<TranscriptionProgress>(reports.Add);
        var transcriber = new FakeTranscriber((_, i) => [new TranscriptSegment($"w{i}", 0, 1)]);

        await new TranscriptionPipeline(transcriber).RunAsync(Audio(90), progress: progress);

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

        await new TranscriptionPipeline(transcriber, chunker: chunker).RunAsync(Audio(30.5));

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
            new TranscriptionPipeline(transcriber).RunAsync(Audio(300), cancellationToken: cts.Token));
    }

    [Fact]
    public async Task WithoutARefinerTheRawTranscriptIsStillReturned()
    {
        var transcriber = new FakeTranscriber((_, _) => [new TranscriptSegment("hello there", 0, 1)]);

        var result = await new TranscriptionPipeline(transcriber).RunAsync(Audio(10));

        Assert.Null(result.Refinement);
        Assert.Equal("hello there", result.BestText);
    }

    [Fact]
    public async Task WithARefinerTheCleanedTextIsPreferred()
    {
        var transcriber = new FakeTranscriber((_, _) => [new TranscriptSegment("hello there", 0, 1)]);
        var refiner = new TranscriptRefiner(new FakeLanguageModel());

        var result = await new TranscriptionPipeline(transcriber, refiner).RunAsync(Audio(10));

        Assert.Equal("HELLO THERE", result.BestText);
    }

    [Fact]
    public async Task SilentRecordingSkipsCleanupEntirely()
    {
        // Nothing was said, so there is nothing to clean up and no reason to wake the model.
        var model = new FakeLanguageModel();
        var transcriber = new FakeTranscriber((_, _) => []);

        var result = await new TranscriptionPipeline(transcriber, new TranscriptRefiner(model))
            .RunAsync(Audio(10));

        Assert.Empty(model.SystemPrompts);
        Assert.Equal(string.Empty, result.BestText);
    }

    [Fact]
    public async Task WrongSampleRateFailsBeforeAnyWorkHappens()
    {
        var transcriber = new FakeTranscriber((_, _) => []);

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            new TranscriptionPipeline(transcriber).RunAsync(new PcmAudio(new float[16000], 8000)));

        Assert.Equal(0, transcriber.CallCount);
    }
}

public sealed class TranscriptRefinerTests
{
    [Fact]
    public void GlossaryTermsReachThePrompt()
    {
        var prompt = TranscriptRefiner.BuildCleanupSystemPrompt(
            ["Kubernetes", "Grafana", "Nadia Okonkwo"],
            RefinementOutputs.Default);

        Assert.Contains("Kubernetes", prompt, StringComparison.Ordinal);
        Assert.Contains("Nadia Okonkwo", prompt, StringComparison.Ordinal);
    }

    [Fact]
    public void ThePromptForbidsRewording()
    {
        // A transcript that reads well but is not what the speaker said is worse than useless.
        var prompt = TranscriptRefiner.BuildCleanupSystemPrompt(null, RefinementOutputs.Default);

        Assert.Contains("Do not reword", prompt, StringComparison.Ordinal);
    }

    [Fact]
    public void GlossaryInstructionsAreOmittedWhenThereIsNoGlossary()
    {
        var prompt = TranscriptRefiner.BuildCleanupSystemPrompt([], RefinementOutputs.Default);

        Assert.DoesNotContain("Correct misheard names", prompt, StringComparison.Ordinal);
    }

    [Fact]
    public void LongTranscriptsAreSplitIntoModelSizedWindows()
    {
        // Twenty segments of ten words: two hundred words at fifty per window.
        var segments = Enumerable.Range(0, 20)
            .Select(i => new TranscriptSegment(
                string.Join(' ', Enumerable.Range(0, 10).Select(w => $"w{i}_{w}")),
                i * 5.0,
                (i + 1) * 5.0))
            .ToList();

        var windows = TranscriptRefiner.SplitSegmentsIntoWindows(segments, 50);

        Assert.Equal(4, windows.Count);
        Assert.All(windows, window => Assert.Equal(5, window.Count));
    }

    /// <summary>
    /// Segments are never cut in half. A segment is the unit that carries a timing, and half of
    /// one has nowhere to put its words back afterwards.
    /// </summary>
    [Fact]
    public void SplittingLosesNoSegments()
    {
        var segments = Enumerable.Range(0, 37)
            .Select(i => new TranscriptSegment($"segment {i} text here", i, i + 1))
            .ToList();

        var windows = TranscriptRefiner.SplitSegmentsIntoWindows(segments, 40);

        Assert.Equal(segments, windows.SelectMany(window => window).ToList());
    }

    /// <summary>A segment longer than a whole window still travels intact, on its own.</summary>
    [Fact]
    public void AnOversizedSegmentGetsItsOwnWindow()
    {
        var segments = new List<TranscriptSegment>
        {
            new("short one", 0, 1),
            new(string.Join(' ', Enumerable.Range(0, 200).Select(w => $"w{w}")), 1, 20),
            new("short two", 20, 21),
        };

        var windows = TranscriptRefiner.SplitSegmentsIntoWindows(segments, 50);

        Assert.Equal(segments, windows.SelectMany(window => window).ToList());
        Assert.Contains(windows, window => window.Count == 1 && window[0] == segments[1]);
    }

    [Theory]
    [InlineData("- Ship the fix\n- Email Priya", 2)]
    [InlineData("1. Ship the fix\n2. Email Priya", 2)]
    [InlineData("• Ship the fix", 1)]
    [InlineData("NONE", 0)]
    [InlineData("", 0)]
    public void ActionItemParsingToleratesTheBulletStylesModelsDriftBetween(string raw, int expected)
    {
        Assert.Equal(expected, TranscriptRefiner.ParseActionItems(raw).Count);
    }

    [Fact]
    public void ActionItemsKeepTheirTextWithoutTheBullet()
    {
        var items = TranscriptRefiner.ParseActionItems("- Ship the fix by Friday");

        Assert.Equal("Ship the fix by Friday", Assert.Single(items));
    }

    [Fact]
    public async Task SummaryAndActionItemsAreOnlyRequestedWhenAskedFor()
    {
        var model = new FakeLanguageModel();
        var transcript = new Transcript([new TranscriptSegment("we agreed to ship on friday", 0, 3)]);

        await new TranscriptRefiner(model).RefineAsync(transcript, null, RefinementOutputs.Punctuation);

        Assert.All(model.SystemPrompts, prompt =>
            Assert.DoesNotContain("Summarise", prompt, StringComparison.Ordinal));
    }

    [Fact]
    public async Task EverythingModeAsksForAllFourOutputs()
    {
        var model = new FakeLanguageModel((system, user) =>
            system.Contains("action items", StringComparison.OrdinalIgnoreCase) ? "- Ship it" : user);
        var transcript = new Transcript([new TranscriptSegment("we agreed to ship on friday", 0, 3)]);

        var result = await new TranscriptRefiner(model)
            .RefineAsync(transcript, ["Friday"], RefinementOutputs.Everything);

        Assert.NotNull(result.Summary);
        Assert.NotNull(result.ActionItems);
        Assert.Single(result.ActionItems);
    }

    [Fact]
    public async Task AnEmptyTranscriptNeverReachesTheModel()
    {
        var model = new FakeLanguageModel();

        var result = await new TranscriptRefiner(model).RefineAsync(Transcript.Empty);

        Assert.Empty(model.SystemPrompts);
        Assert.Equal(string.Empty, result.CleanedText);
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
