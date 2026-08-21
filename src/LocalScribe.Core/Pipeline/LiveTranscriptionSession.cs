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

    // Guards the window and the counters derived from it. Audio arrives on the capture thread
    // while a pass, or FinishAsync, reads the same list from another; a List<float> torn between
    // the two throws or yields garbage rather than failing loudly.
    private readonly object _windowLock = new();

    private double _windowStartSeconds;
    private double _sessionSeconds;
    private int _samplesSinceLastPass;

    public LiveTranscriptionSession(ITranscriber transcriber, int sampleRate = PcmAudio.WhisperSampleRate)
    {
        _transcriber = transcriber ?? throw new ArgumentNullException(nameof(transcriber));
        _sampleRate = sampleRate;
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
        lock (_windowLock)
        {
            _window.AddRange(samples.Span);
            _samplesSinceLastPass += samples.Length;
            _sessionSeconds += samples.Length / (double)_sampleRate;

            TrimWindow();

            if (_samplesSinceLastPass < PassIntervalSeconds * _sampleRate)
            {
                return null;
            }

            _samplesSinceLastPass = 0;
        }

        return await RunPassAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Stops listening and commits whatever is still provisional. Call this when the user hits
    /// stop, otherwise the last few seconds never make it into the transcript.
    /// </summary>
    public async Task<IReadOnlyList<TranscriptSegment>> FinishAsync(CancellationToken cancellationToken = default)
    {
        await RunPassAsync(cancellationToken, commitEverything: true).ConfigureAwait(false);
        return _committed;
    }

    private async Task<LiveUpdate?> RunPassAsync(
        CancellationToken cancellationToken,
        bool commitEverything = false)
    {
        // If a pass is already running, skip rather than queue. Audio keeps arriving whether or
        // not we keep up, and a backlog of stale passes helps nobody.
        if (!await _passLock.WaitAsync(0, cancellationToken).ConfigureAwait(false))
        {
            return null;
        }

        try
        {
            AudioChunk chunk;
            double sessionSeconds;

            // Copy the window out before transcribing, so the capture thread can keep appending
            // during a pass that takes longer than the interval between buffers.
            lock (_windowLock)
            {
                if (_window.Count == 0)
                {
                    return null;
                }

                var padded = new float[(int)(WindowSeconds * _sampleRate)];
                var count = Math.Min(_window.Count, padded.Length);
                _window.CopyTo(0, padded, 0, count);

                chunk = new AudioChunk(padded, _windowStartSeconds, count / (double)_sampleRate);
                sessionSeconds = _sessionSeconds;
            }

            var segments = await _transcriber
                .TranscribeChunkAsync(chunk, cancellationToken)
                .ConfigureAwait(false);

            var settledBefore = commitEverything
                ? double.MaxValue
                : sessionSeconds - CommitAfterSeconds;

            var provisional = new List<string>();

            foreach (var segment in segments)
            {
                if (segment.LooksHallucinated || string.IsNullOrWhiteSpace(segment.Text))
                {
                    continue;
                }

                if (segment.EndSeconds <= settledBefore)
                {
                    if (!IsAlreadyCommitted(segment))
                    {
                        _committed.Add(segment);
                    }
                }
                else
                {
                    provisional.Add(segment.Text.Trim());
                }
            }

            return new LiveUpdate(
                string.Join(" ", provisional),
                IsFinal: commitEverything,
                _windowStartSeconds);
        }
        finally
        {
            _passLock.Release();
        }
    }

    /// <summary>
    /// Each pass re-transcribes the whole window, so most of what comes back was already
    /// committed on an earlier pass. Compare on normalised text, since timings shift slightly
    /// between passes even when the words do not.
    /// </summary>
    private bool IsAlreadyCommitted(TranscriptSegment candidate)
    {
        var normalised = TranscriptStitcher.Normalise(candidate.Text);

        for (var i = _committed.Count - 1; i >= 0; i--)
        {
            if (candidate.StartSeconds - _committed[i].EndSeconds > WindowSeconds)
            {
                break;
            }

            if (TranscriptStitcher.Normalise(_committed[i].Text) == normalised)
            {
                return true;
            }
        }

        return false;
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

    public ValueTask DisposeAsync()
    {
        _passLock.Dispose();
        return ValueTask.CompletedTask;
    }
}
