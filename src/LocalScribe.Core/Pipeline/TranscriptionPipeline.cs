using LocalScribe.Core.Audio;
using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Pipeline;

/// <summary>Progress for the UI: how far through the recording, and the text so far.</summary>
/// <param name="ChunksCompleted">Units of this stage finished.</param>
/// <param name="ChunksTotal">Units in this stage.</param>
/// <param name="LatestText">Text from the window that just finished, for a live-updating view.</param>
/// <param name="LatestStartSeconds">Where the finished window's audio actually begins.</param>
/// <param name="LatestEndSeconds">
/// Where it ends. Carried so a live view can stamp streamed text with the truth: windows
/// overlap, so window-number-times-window-length overstates position by the overlap per
/// window — an error that compounded to a minute and a half by the end of a long recording
/// and pushed every streamed anchor out of the aligner's reach.
/// </param>
public sealed record TranscriptionProgress(
    int ChunksCompleted,
    int ChunksTotal,
    string LatestText,
    double LatestStartSeconds = 0,
    double LatestEndSeconds = 0)
{
    public double Fraction => ChunksTotal == 0 ? 0 : ChunksCompleted / (double)ChunksTotal;
}

/// <summary>
/// Runs a recording end to end: chunk, transcribe, stitch.
/// <para>
/// Windows are transcribed one at a time rather than in parallel. That is deliberate — the NPU
/// is a single shared resource, and queueing several windows at it makes the whole run slower
/// while also stealing the responsiveness we were trying to protect.
/// </para>
/// </summary>
public sealed class TranscriptionPipeline
{
    private readonly ITranscriber _transcriber;
    private readonly AudioChunker _chunker;
    private readonly TranscriptStitcher _stitcher;

    public TranscriptionPipeline(
        ITranscriber transcriber,
        AudioChunker? chunker = null,
        TranscriptStitcher? stitcher = null)
    {
        _transcriber = transcriber ?? throw new ArgumentNullException(nameof(transcriber));
        _chunker = chunker ?? new AudioChunker();
        _stitcher = stitcher ?? new TranscriptStitcher();
    }

    /// <summary>The words and their timings, as Whisper heard them.</summary>
    public async Task<Transcript> TranscribeAsync(
        PcmAudio audio,
        IProgress<TranscriptionProgress>? progress = null,
        CancellationToken cancellationToken = default,
        SpeechTask task = SpeechTask.Transcribe)
    {
        // A different recording, so anything carried over from the last one is wrong now. Here
        // because this is the method every caller passes through.
        _transcriber.BeginRecording(task);

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

            // Stamps repaired to agree with the decoder's order for the transcript. Where the next
            // window starts is still read from the raw stamps below, which already ignores any
            // past the window's audio.
            perChunkSegments.Add(TranscriptStitcher.InDecoderOrder(
                segments, chunk.StartSeconds, chunk.StartSeconds + chunk.ContentSeconds));
            completed++;

            // Once a window has reached the end of the audio there is nothing after it but
            // padding, and seeking back into speech already covered would only transcribe it
            // twice.
            var reachedEnd = chunk.StartSeconds + chunk.ContentSeconds >= audio.DurationSeconds;

            position = reachedEnd ? audio.DurationSeconds : NextPosition(position, chunk, segments);

            progress?.Report(new TranscriptionProgress(
                completed,
                EstimateTotal(completed, position, audio.DurationSeconds),
                string.Join(" ", segments.Select(s => s.Text.Trim())),
                LatestStartSeconds: chunk.StartSeconds,
                LatestEndSeconds: Math.Min(
                    chunk.StartSeconds + chunk.ContentSeconds, audio.DurationSeconds)));
        }

        // Split where Whisper marked a change of speaker. It writes dialogue the way a script
        // does — a dash opening a line, another when somebody cuts in — from the same audio the
        // diarizer will see, and until now the marks survived into the transcript as stray
        // punctuation with nothing acting on them. Two people sharing one segment could not be
        // told apart afterwards however good the voices were, because there was only ever one
        // segment to attach a speaker to.
        return new Transcript(DialogueMarks.Split(_stitcher.Stitch(perChunkSegments)));
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
