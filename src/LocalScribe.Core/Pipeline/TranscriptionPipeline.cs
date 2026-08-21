using LocalScribe.Core.Audio;
using LocalScribe.Core.Refinement;
using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Pipeline;

/// <summary>Which stage of the run is being reported.</summary>
public enum TranscriptionPhase
{
    /// <summary>Whisper, window by window.</summary>
    Transcribing,

    /// <summary>The language model repairing punctuation, names, and writing a summary.</summary>
    CleaningUp,
}

/// <summary>Progress for the UI: which stage, how far through it, and the text so far.</summary>
/// <param name="ChunksCompleted">Units of this stage finished.</param>
/// <param name="ChunksTotal">Units in this stage.</param>
/// <param name="LatestText">Text from the window that just finished, for a live-updating view.</param>
/// <param name="Phase">
/// Which stage this is. A run has more than one, and they take comparable amounts of time, so a
/// bar that reaches the end and then sits there for a minute is worse than no bar at all.
/// </param>
public sealed record TranscriptionProgress(
    int ChunksCompleted,
    int ChunksTotal,
    string LatestText,
    TranscriptionPhase Phase = TranscriptionPhase.Transcribing)
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

    /// <summary>
    /// Transcribes and then cleans up, one after the other.
    /// <para>
    /// The convenience for callers with nothing else to do. A caller that also wants to work out
    /// who spoke should run <see cref="TranscribeAsync"/> and then start
    /// <see cref="RefineAsync"/> alongside the diarizer, which reads only the audio and so has
    /// no reason to wait for either.
    /// </para>
    /// </summary>
    public async Task<TranscriptionResult> RunAsync(
        PcmAudio audio,
        IReadOnlyList<string>? glossary = null,
        RefinementOutputs outputs = RefinementOutputs.Default,
        IProgress<TranscriptionProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var transcript = await TranscribeAsync(audio, progress, cancellationToken)
            .ConfigureAwait(false);

        var refinement = await RefineAsync(transcript, glossary, outputs, progress, cancellationToken)
            .ConfigureAwait(false);

        return new TranscriptionResult(transcript, refinement);
    }

    /// <summary>Whisper only: the words and their timings, before any cleanup.</summary>
    public async Task<Transcript> TranscribeAsync(
        PcmAudio audio,
        IProgress<TranscriptionProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        audio.EnsureWhisperFormat();

        var perChunkSegments = new List<IReadOnlyList<TranscriptSegment>>();

        var position = 0.0;
        var completed = 0;

        while (position < audio.DurationSeconds)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var chunk = _chunker.WindowAt(audio, position);

            // A window that is almost entirely padding gives Whisper nothing to work with, and
            // asking anyway is the fastest route to invented text.
            var segments = chunk.IsMostlyPadding
                ? []
                : await _transcriber.TranscribeChunkAsync(chunk, cancellationToken).ConfigureAwait(false);

            perChunkSegments.Add(segments);
            completed++;

            // Once a window has reached the end of the audio there is nothing after it but
            // padding, and seeking back into speech already covered would only transcribe it
            // twice.
            var reachedEnd = chunk.StartSeconds + chunk.ContentSeconds >= audio.DurationSeconds;

            position = reachedEnd ? audio.DurationSeconds : NextPosition(position, chunk, segments);

            progress?.Report(new TranscriptionProgress(
                completed,
                EstimateTotal(completed, position, audio.DurationSeconds),
                string.Join(" ", segments.Select(s => s.Text.Trim()))));
        }

        return new Transcript(_stitcher.Stitch(perChunkSegments));
    }

    /// <summary>
    /// Cleanup only: punctuation, the glossary, and whatever else was asked for. Null when no
    /// cleanup model was found or there is nothing to clean.
    /// </summary>
    public async Task<RefinementResult?> RefineAsync(
        Transcript transcript,
        IReadOnlyList<string>? glossary = null,
        RefinementOutputs outputs = RefinementOutputs.Default,
        IProgress<TranscriptionProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(transcript);

        if (_refiner is null || transcript.Segments.Count == 0)
        {
            return null;
        }

        var cleanup = new Progress<double>(fraction => progress?.Report(new TranscriptionProgress(
            (int)Math.Round(fraction * 100),
            100,
            string.Empty,
            TranscriptionPhase.CleaningUp)));

        return await _refiner
            .RefineAsync(transcript, glossary, outputs, cleanup, cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Where the next window starts.
    /// <para>
    /// Not a fixed stride. Whisper stops transcribing wherever it finds a natural end, which is
    /// routinely well before the end of the window it was given — it would rather finish a
    /// sentence than cut one. A fixed stride assumes the gap between where it stopped and where
    /// the next window begins is never wider than the overlap, and when that assumption breaks
    /// the audio in between is transcribed by nobody and the words simply vanish. On a
    /// sixty-second recording that lost "the segment. If the stitching" from the middle of a
    /// sentence, with the seam reading as though the speaker had jumped.
    /// </para>
    /// <para>
    /// So the next window starts a little before wherever speech was last heard, and falls back
    /// to a fixed stride only when a window yielded nothing to go on.
    /// </para>
    /// </summary>
    private double NextPosition(
        double position,
        AudioChunk chunk,
        IReadOnlyList<TranscriptSegment> segments)
    {
        var fallback = position + _chunker.DefaultAdvanceSeconds;

        var lastEnd = segments.Count == 0 ? 0 : segments[^1].EndSeconds;
        if (lastEnd <= 0)
        {
            return fallback;
        }

        var resume = lastEnd - _chunker.OverlapSeconds;

        // Never past the audio the window actually held: a timestamp beyond it is the model
        // describing its own padding, and trusting it would skip real speech.
        var limit = chunk.StartSeconds + chunk.ContentSeconds;
        resume = Math.Min(resume, limit);

        // And never backwards, nor so little that the same window is transcribed forever.
        return resume <= position + MinimumAdvanceSeconds ? fallback : resume;
    }

    /// <summary>
    /// Windows still to come, at the rate they have been going. The count is not known in
    /// advance once the stride varies, so this is an estimate that stays honest by being
    /// recomputed rather than fixed at the start.
    /// </summary>
    private int EstimateTotal(int completed, double position, double duration)
    {
        if (position >= duration)
        {
            return completed;
        }

        var covered = Math.Max(position, 0.001);
        var perWindow = covered / completed;
        var remaining = (int)Math.Ceiling((duration - position) / Math.Max(perWindow, 0.001));

        return completed + Math.Max(remaining, 1);
    }

    /// <summary>
    /// The least a window may advance. Below this a recording that goes badly could re-read the
    /// same seconds indefinitely, which is worse than the gap this all exists to close.
    /// </summary>
    private const double MinimumAdvanceSeconds = 1.0;

}
