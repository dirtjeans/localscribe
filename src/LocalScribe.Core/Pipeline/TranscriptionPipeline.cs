using LocalScribe.Core.Audio;
using LocalScribe.Core.Refinement;
using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Pipeline;

/// <summary>Progress for the UI: which window we are on, and the text so far.</summary>
/// <param name="ChunksCompleted">Windows finished.</param>
/// <param name="ChunksTotal">Windows in the recording.</param>
/// <param name="LatestText">Text from the window that just finished, for a live-updating view.</param>
public sealed record TranscriptionProgress(int ChunksCompleted, int ChunksTotal, string LatestText)
{
    public double Fraction => ChunksTotal == 0 ? 0 : ChunksCompleted / (double)ChunksTotal;
}

/// <param name="Transcript">Raw output, before any cleanup.</param>
/// <param name="Refinement">Cleanup output, or <c>null</c> when no language model was available.</param>
public sealed record TranscriptionResult(Transcript Transcript, RefinementResult? Refinement)
{
    /// <summary>The best text we have: cleaned when cleanup ran, raw otherwise.</summary>
    public string BestText => Refinement?.CleanedText ?? Transcript.FullText;
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
    private readonly AudioChunker _chunker;
    private readonly TranscriptStitcher _stitcher;

    public TranscriptionPipeline(
        ITranscriber transcriber,
        TranscriptRefiner? refiner = null,
        AudioChunker? chunker = null,
        TranscriptStitcher? stitcher = null)
    {
        _transcriber = transcriber ?? throw new ArgumentNullException(nameof(transcriber));
        _refiner = refiner;
        _chunker = chunker ?? new AudioChunker();
        _stitcher = stitcher ?? new TranscriptStitcher();
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

        if (_refiner is null || transcript.Segments.Count == 0)
        {
            return new TranscriptionResult(transcript, Refinement: null);
        }

        var refinement = await _refiner
            .RefineAsync(transcript, glossary, outputs, cancellationToken)
            .ConfigureAwait(false);

        return new TranscriptionResult(transcript, refinement);
    }
}
