using LocalScribe.Core.Audio;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Refinement;
using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Pipeline;

/// <summary>Progress for the UI: which window we are on, and the text so far.</summary>
/// <param name="ChunksCompleted">Windows finished.</param>
/// <param name="ChunksTotal">Windows in the recording.</param>
/// <param name="LatestText">Text from the window that just finished, for a live-updating view.</param>
/// <param name="Stage">
/// What the pipeline is doing. Diarization runs after every window is transcribed and can take
/// a while on a long recording, so a progress bar that only counts windows would appear to
/// finish and then hang.
/// </param>
public sealed record TranscriptionProgress(
    int ChunksCompleted,
    int ChunksTotal,
    string LatestText,
    PipelineStage Stage = PipelineStage.Transcribing)
{
    public double Fraction => ChunksTotal == 0 ? 0 : ChunksCompleted / (double)ChunksTotal;
}

/// <summary>Which part of the run is in progress.</summary>
public enum PipelineStage
{
    /// <summary>Turning audio windows into text.</summary>
    Transcribing,

    /// <summary>Working out who spoke when.</summary>
    Diarizing,

    /// <summary>Punctuation, glossary, and summaries.</summary>
    Refining,
}

/// <param name="Transcript">Raw output, before any cleanup.</param>
/// <param name="Refinement">Cleanup output, or <c>null</c> when no language model was available.</param>
/// <param name="SpeakerTurns">Speaker turns, or empty when diarization did not run.</param>
/// <param name="DiarizationError">
/// Why diarization did not produce turns, when it was asked to and failed. Kept rather than
/// thrown: a transcript without speaker labels is still the thing the user asked for.
/// </param>
public sealed record TranscriptionResult(
    Transcript Transcript,
    RefinementResult? Refinement,
    IReadOnlyList<SpeakerTurn>? SpeakerTurns = null,
    string? DiarizationError = null)
{
    /// <summary>The best text we have: cleaned when cleanup ran, raw otherwise.</summary>
    public string BestText => Refinement?.CleanedText ?? Transcript.FullText;

    /// <summary>The transcript as dialogue, when speakers are known.</summary>
    public string DialogueText => Transcript.HasSpeakers
        ? SpeakerAssigner.FormatAsDialogue(Transcript.Segments)
        : Transcript.FullText;
}

/// <summary>
/// Runs a recording end to end: chunk, transcribe, stitch, then optionally clean up.
/// <para>
/// Windows are transcribed one at a time rather than in parallel. That is deliberate — the NPU
/// is a single shared resource, and queueing several windows at it makes the whole run slower
/// while also stealing the responsiveness we were trying to protect.
/// </para>
/// </summary>
public sealed class TranscriptionPipeline
{
    private readonly ITranscriber _transcriber;
    private readonly TranscriptRefiner? _refiner;
    private readonly IDiarizer? _diarizer;
    private readonly AudioChunker _chunker;
    private readonly TranscriptStitcher _stitcher;
    private readonly SpeakerAssigner _assigner;

    public TranscriptionPipeline(
        ITranscriber transcriber,
        TranscriptRefiner? refiner = null,
        AudioChunker? chunker = null,
        TranscriptStitcher? stitcher = null,
        IDiarizer? diarizer = null,
        DiarizationOptions? diarizationOptions = null)
    {
        _transcriber = transcriber ?? throw new ArgumentNullException(nameof(transcriber));
        _refiner = refiner;
        _diarizer = diarizer;
        _chunker = chunker ?? new AudioChunker();
        _stitcher = stitcher ?? new TranscriptStitcher();
        _assigner = new SpeakerAssigner(diarizationOptions);
    }

    public async Task<TranscriptionResult> RunAsync(
        PcmAudio audio,
        IReadOnlyList<string>? glossary = null,
        RefinementOutputs outputs = RefinementOutputs.Default,
        IProgress<TranscriptionProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        audio.EnsureWhisperFormat();

        var chunks = _chunker.Chunk(audio);
        var perChunkSegments = new List<IReadOnlyList<TranscriptSegment>>(chunks.Count);

        for (var i = 0; i < chunks.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var chunk = chunks[i];

            // A window that is almost entirely padding gives Whisper nothing to work with, and
            // asking anyway is the fastest route to invented text.
            var segments = chunk.IsMostlyPadding
                ? []
                : await _transcriber.TranscribeChunkAsync(chunk, cancellationToken).ConfigureAwait(false);

            perChunkSegments.Add(segments);
            progress?.Report(new TranscriptionProgress(
                i + 1,
                chunks.Count,
                string.Join(" ", segments.Select(s => s.Text.Trim()))));
        }

        var transcript = new Transcript(_stitcher.Stitch(perChunkSegments));

        // Diarization runs after transcription rather than alongside it. The two use different
        // hardware and could in principle overlap, but running them at once doubles the CPU
        // draw at exactly the moment the machine is already busy, which is the thing the whole
        // accelerator policy exists to avoid.
        var (turns, diarizationError) = await DiarizeAsync(
            audio, chunks.Count, progress, cancellationToken).ConfigureAwait(false);

        if (turns.Count > 0)
        {
            transcript = transcript with { Segments = _assigner.Assign(transcript.Segments, turns) };
        }

        if (_refiner is null || transcript.Segments.Count == 0)
        {
            return new TranscriptionResult(transcript, null, turns, diarizationError);
        }

        progress?.Report(new TranscriptionProgress(
            chunks.Count, chunks.Count, string.Empty, PipelineStage.Refining));

        var refinement = await _refiner
            .RefineAsync(transcript, glossary, outputs, cancellationToken)
            .ConfigureAwait(false);

        return new TranscriptionResult(transcript, refinement, turns, diarizationError);
    }

    /// <summary>
    /// Runs diarization if one is configured, converting failure into a reported error rather
    /// than losing a transcript that is otherwise complete.
    /// </summary>
    private async Task<(IReadOnlyList<SpeakerTurn> Turns, string? Error)> DiarizeAsync(
        PcmAudio audio,
        int chunkCount,
        IProgress<TranscriptionProgress>? progress,
        CancellationToken cancellationToken)
    {
        if (_diarizer is null)
        {
            return ([], null);
        }

        progress?.Report(new TranscriptionProgress(
            chunkCount, chunkCount, string.Empty, PipelineStage.Diarizing));

        try
        {
            var turns = await _diarizer
                .DiarizeAsync(audio, progress: null, cancellationToken)
                .ConfigureAwait(false);

            return (turns, null);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception exception)
        {
            return ([], exception.Message);
        }
    }
}
