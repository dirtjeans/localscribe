using LocalScribe.Core.Audio;
using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Pipeline;

/// <summary>Text emitted while listening.</summary>
/// <param name="Text">The words recognised.</param>
/// <param name="IsFinal">
/// False while the span may still change as more audio arrives; true once it has scrolled out
/// of the live window and been committed.
/// </param>
/// <param name="StartSeconds">Offset from the start of the session.</param>
public sealed record LiveUpdate(string Text, bool IsFinal, double StartSeconds);

/// <summary>
/// Transcribes a microphone as it records.
/// <para>
/// The hard part of live transcription is that the last few words are always unreliable: the
/// model cannot tell where a sentence is going. So this keeps a rolling window, re-transcribes
/// it as audio arrives, and only commits text once enough later audio exists to be confident
/// it will not change. Committed text is final; the tail is marked provisional so the UI can
/// show it in a lighter style rather than pretending it is settled.
/// </para>
/// </summary>
public sealed class LiveTranscriptionSession : IAsyncDisposable
{
    /// <summary>
    /// How much audio the rolling window holds. Matches the encoder's fixed input, so we never
    /// pay to transcribe padding we could have skipped.
    /// </summary>
    public const double WindowSeconds = AudioChunker.WindowSeconds;

    /// <summary>
    /// Audio to accumulate between passes. Shorter feels more responsive but re-runs the encoder
    /// more often; a second is a reasonable middle on this hardware.
    /// </summary>
    public const double PassIntervalSeconds = 1.0;

    /// <summary>
    /// Text older than this within the window is treated as settled. Whisper rarely revises a
    /// span once several seconds of later audio exist.
    /// </summary>
    public const double CommitAfterSeconds = 8.0;

    private readonly ITranscriber _transcriber;
    private readonly int _sampleRate;
    private readonly List<float> _window = [];
    private readonly List<TranscriptSegment> _committed = [];
    private readonly SemaphoreSlim _passLock = new(1, 1);

    /// <summary>
    /// Guards the rolling window. Audio arrives on the capture thread while a pass reads the
    /// window on a worker, so these genuinely overlap.
    /// </summary>
    private readonly object _windowLock = new();

    /// <summary>
    /// How far into the recording has already been committed. The rolling window is
    /// re-transcribed in full every pass, so this is what stops the same speech being appended
    /// again and again.
    /// </summary>
    private double _committedThroughSeconds;

    /// <summary>
    /// The best-formatted reading of the current window seen so far, kept so a later pass that
    /// degenerates does not become the transcript merely by arriving last.
    /// </summary>
    private List<TranscriptSegment>? _bestReading;

    /// <summary>
    /// Where the window started when <see cref="_bestReading"/> was taken. A reading is only a
    /// substitute for another reading of the same audio, and once the window has rolled on it is
    /// no longer that.
    /// </summary>
    private double _bestReadingWindowStart = double.NaN;

    private volatile bool _disposed;
    private double _windowStartSeconds;
    private double _sessionSeconds;
    private int _samplesSinceLastPass;

    public LiveTranscriptionSession(ITranscriber transcriber, int sampleRate = PcmAudio.WhisperSampleRate)
    {
        _transcriber = transcriber ?? throw new ArgumentNullException(nameof(transcriber));
        _sampleRate = sampleRate;

        // One session is one recording, so this is where the last one is forgotten.
        _transcriber.BeginRecording();
    }

    /// <summary>Everything committed so far, in order.</summary>
    public IReadOnlyList<TranscriptSegment> CommittedSegments => _committed;

    /// <summary>
    /// Feeds newly captured audio in. Returns an update when this push triggered a pass, and
    /// <c>null</c> when it is still accumulating.
    /// </summary>
    public async Task<LiveUpdate?> PushAsync(
        ReadOnlyMemory<float> samples,
        CancellationToken cancellationToken = default)
    {
        // Audio keeps arriving for a moment after the user stops: the capture thread has buffers
        // already queued, and a handler that is mid-await is still running. Dropping those
        // quietly is the whole contract here — throwing at one of them surfaces as a failure of
        // the transcription rather than of the shutdown.
        if (_disposed)
        {
            return null;
        }

        bool due;

        lock (_windowLock)
        {
            _window.AddRange(samples.Span);
            _samplesSinceLastPass += samples.Length;
            _sessionSeconds += samples.Length / (double)_sampleRate;

            TrimWindow();

            due = _samplesSinceLastPass >= PassIntervalSeconds * _sampleRate;
            if (due)
            {
                _samplesSinceLastPass = 0;
            }
        }

        return due ? await RunPassAsync(cancellationToken).ConfigureAwait(false) : null;
    }

    /// <summary>
    /// Stops listening and commits whatever is still provisional. Call this when the user hits
    /// stop, otherwise the last few seconds never make it into the transcript.
    /// </summary>
    public async Task<IReadOnlyList<TranscriptSegment>> FinishAsync(CancellationToken cancellationToken = default)
    {
        // Unlike a streaming pass this one waits its turn rather than skipping. A pass is very
        // likely already running when the user hits stop, and skipping here would drop exactly
        // the trailing words this method exists to keep.
        await RunPassAsync(cancellationToken, commitEverything: true, waitForTurn: true)
            .ConfigureAwait(false);

        return _committed;
    }

    private async Task<LiveUpdate?> RunPassAsync(
        CancellationToken cancellationToken,
        bool commitEverything = false,
        bool waitForTurn = false)
    {
        if (_disposed && !waitForTurn)
        {
            return null;
        }

        // A streaming pass skips rather than queues when one is already running. Audio keeps
        // arriving whether or not we keep up, and a backlog of stale passes helps nobody.
        if (waitForTurn)
        {
            await _passLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
        else if (!await _passLock.WaitAsync(0, cancellationToken).ConfigureAwait(false))
        {
            return null;
        }

        try
        {
            float[] padded;
            int count;
            double windowStart;

            lock (_windowLock)
            {
                if (_window.Count == 0)
                {
                    return null;
                }

                padded = new float[(int)(WindowSeconds * _sampleRate)];
                count = Math.Min(_window.Count, padded.Length);
                _window.CopyTo(0, padded, 0, count);
                windowStart = _windowStartSeconds;
            }

            var chunk = new AudioChunk(padded, windowStart, count / (double)_sampleRate);
            var segments = await _transcriber
                .TranscribeChunkAsync(chunk, cancellationToken)
                .ConfigureAwait(false);

            segments = PreferBestReading(segments, windowStart);

            var settledBefore = commitEverything
                ? double.MaxValue
                : _sessionSeconds - CommitAfterSeconds;

            var provisional = new List<string>();

            foreach (var segment in segments)
            {
                if (segment.LooksHallucinated || string.IsNullOrWhiteSpace(segment.Text))
                {
                    continue;
                }

                if (segment.EndSeconds <= settledBefore)
                {
                    Commit(segment);
                }
                else
                {
                    provisional.Add(segment.Text.Trim());
                }
            }

            return new LiveUpdate(
                string.Join(" ", provisional),
                IsFinal: commitEverything,
                windowStart);
        }
        finally
        {
            _passLock.Release();
        }
    }

    /// <summary>
    /// The end of the committed transcript, long enough to cover any overlap a single pass can
    /// produce and no longer, since every extra word is another chance at a false match.
    /// </summary>
    private string CommittedTail()
    {
        const int words = 60;

        var tail = new List<string>();

        for (var i = _committed.Count - 1; i >= 0 && tail.Count < words; i--)
        {
            var parts = _committed[i].Text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            tail.InsertRange(0, parts);
        }

        return string.Join(" ", tail.TakeLast(words));
    }

    /// <summary>
    /// Remembers the best-formatted reading of the window, and falls back to it when a pass
    /// returns the degenerate one.
    /// <para>
    /// Every pass re-transcribes the same rolling window, so a session normally holds several
    /// readings of identical audio. Whisper occasionally delivers one of them as a bare
    /// lowercase run of words, and on the final pass that reading would otherwise become the
    /// transcript purely because it came last. Nothing here rewrites the model's output; it only
    /// chooses between things the model itself said about the same seconds of speech.
    /// </para>
    /// </summary>
    private IReadOnlyList<TranscriptSegment> PreferBestReading(
        IReadOnlyList<TranscriptSegment> segments,
        double windowStart)
    {
        var text = string.Join(" ", segments.Select(s => s.Text.Trim()));

        if (!TranscriptQuality.LooksUnformatted(text))
        {
            _bestReading = segments.ToList();
            _bestReadingWindowStart = windowStart;
            return segments;
        }

        // Only a reading of the same window is a candidate. Once the window has rolled, an
        // earlier reading describes different seconds of audio, and substituting it drops
        // whatever was said in between — which is a far worse failure than the missing commas
        // it was meant to repair.
        if (_bestReading is null || Math.Abs(_bestReadingWindowStart - windowStart) > double.Epsilon)
        {
            return segments;
        }

        var best = string.Join(" ", _bestReading.Select(s => s.Text.Trim()));

        return TranscriptQuality.PreferCandidate(text, best) ? _bestReading : segments;
    }

    /// <summary>
    /// How far a segment may start before the committed mark and still count as new. Whisper
    /// shifts a boundary by a couple of hundred milliseconds between passes even when the words
    /// are identical, so an exact comparison would drop real speech.
    /// </summary>
    private const double BoundaryToleranceSeconds = 0.4;

    /// <summary>
    /// Appends a settled segment, unless the audio it covers has already been committed.
    /// <para>
    /// The test is the span it occupies, not the words it contains. Every pass re-transcribes
    /// the whole rolling window, and it does not divide that window the same way twice: the same
    /// speech comes back as "long enough to span more than one window" on one pass and "window,
    /// so that chunking and stitching are both exercised" on the next. Those are different
    /// strings covering overlapping audio, so matching on text appended both, and a minute of
    /// speech grew into several minutes of stuttering repetition.
    /// </para>
    /// <para>
    /// Time cannot drift like that. Audio committed once is committed, whatever words a later
    /// pass decides were in it.
    /// </para>
    /// </summary>
    private void Commit(TranscriptSegment segment)
    {
        if (segment.EndSeconds <= _committedThroughSeconds + BoundaryToleranceSeconds)
        {
            // Entirely inside audio we have already written down.
            return;
        }

        // A segment straddling the mark used to be deferred, on the reasoning that a later pass
        // would offer the same speech starting cleanly after it. Often one does. But once the
        // window rolls past, no pass ever offers it again, and the deferral is silently
        // permanent — a whole clause vanished mid-sentence, "the Orion cores stay" running
        // straight into the next paragraph. Losing words to avoid repeating them is the wrong
        // trade, so it is taken and its overlapping opening trimmed below.

        // Trimmed against the tail of everything committed, not just the last segment. A pass
        // re-reads its whole window, so the words it repeats routinely span several of the
        // segments already written down, and comparing with only the most recent one leaves the
        // rest of the repetition in place.
        var text = TranscriptStitcher.TrimLeadingOverlap(CommittedTail(), segment.Text);

        if (text.Trim().Length == 0)
        {
            _committedThroughSeconds = Math.Max(_committedThroughSeconds, segment.EndSeconds);
            return;
        }

        _committed.Add(segment with { Text = text });
        _committedThroughSeconds = Math.Max(_committedThroughSeconds, segment.EndSeconds);
    }

    /// <summary>Drops audio that has scrolled past the window, keeping memory flat over long sessions.</summary>
    private void TrimWindow()
    {
        var maxSamples = (int)(WindowSeconds * _sampleRate);
        var excess = _window.Count - maxSamples;
        if (excess <= 0)
        {
            return;
        }

        _window.RemoveRange(0, excess);
        _windowStartSeconds += excess / (double)_sampleRate;
    }

    /// <summary>
    /// Stops accepting audio and waits for any pass still running.
    /// <para>
    /// The semaphore is deliberately not disposed. <see cref="SemaphoreSlim"/> only requires
    /// disposal once its <c>AvailableWaitHandle</c> has been touched, which this never does, and
    /// disposing it turned every late audio callback into an <c>ObjectDisposedException</c>
    /// surfacing as "live transcription failed". Draining the pass and refusing further work is
    /// what shutdown actually needs.
    /// </para>
    /// </summary>
    public async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        // Waiting rather than abandoning: a pass in flight is reading the window and appending
        // to the committed list.
        await _passLock.WaitAsync().ConfigureAwait(false);
        _passLock.Release();
    }
}
