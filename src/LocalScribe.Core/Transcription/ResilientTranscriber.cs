using LocalScribe.Core.Audio;

namespace LocalScribe.Core.Transcription;

/// <summary>
/// Restarts a failed engine and retries the window it failed on, so one engine hiccup costs a
/// moment instead of the recording.
/// <para>
/// The pipeline hands over audio one window at a time, which is what makes this possible: the
/// failed window is still in hand, so a rebuilt engine picks up exactly where the old one
/// stopped. A cancellation the user actually asked for passes straight through — the token
/// says so — while a cancellation nobody requested is treated as what it is, an engine
/// failure in costume (HttpClient reports timeouts that way, and a binding tearing down can
/// too).
/// </para>
/// <para>
/// One restart per window, and a window that fails the fresh engine too fails honestly: two
/// engines agreeing something is wrong is systemic, and silently skipping speech would be the
/// worse lie. The restart is counted and the failure kept, so a status line can say what
/// happened rather than pretending nothing did.
/// </para>
/// </summary>
public sealed class ResilientTranscriber : ITranscriber
{
    private readonly Func<ITranscriber> _open;
    private ITranscriber _inner;
    private SpeechTask _task = SpeechTask.Transcribe;

    public ResilientTranscriber(Func<ITranscriber> open)
    {
        _open = open ?? throw new ArgumentNullException(nameof(open));
        _inner = open();
    }

    public string Description => _inner.Description;

    public string? DetectedLanguage => _inner.DetectedLanguage;

    /// <summary>How many times the engine has been rebuilt. Zero is the usual answer.</summary>
    public int Restarts { get; private set; }

    /// <summary>What the last restart was recovering from, for diagnostics.</summary>
    public Exception? LastFailure { get; private set; }

    public void BeginRecording(SpeechTask task = SpeechTask.Transcribe)
    {
        // Remembered as well as forwarded: a replacement engine mid-recording must resume
        // the same task, or the transcript would change convention partway through.
        _task = task;
        _inner.BeginRecording(task);
    }

    public async Task<IReadOnlyList<TranscriptSegment>> TranscribeChunkAsync(
        AudioChunk chunk,
        CancellationToken cancellationToken = default)
    {
        try
        {
            return await _inner.TranscribeChunkAsync(chunk, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception failure)
        {
            LastFailure = failure;
            Restarts++;

            Restart();

            // The same window again, on the fresh engine. The language re-settles from the
            // same audio, so a mid-recording restart keeps its answer in practice.
            return await _inner.TranscribeChunkAsync(chunk, cancellationToken).ConfigureAwait(false);
        }
    }

    private void Restart()
    {
        try
        {
            _inner.Dispose();
        }
        catch (Exception)
        {
            // A failed engine failing to die cleanly is not worth losing the restart over.
        }

        _inner = _open();
        _inner.BeginRecording(_task);
    }

    public void Dispose() => _inner.Dispose();
}
