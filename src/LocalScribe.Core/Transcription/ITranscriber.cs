using LocalScribe.Core.Audio;

namespace LocalScribe.Core.Transcription;

/// <summary>
/// A speech-to-text backend. The pipeline depends on this rather than on ONNX Runtime directly,
/// which keeps the orchestration testable and leaves room for a different runtime later.
/// </summary>
public interface ITranscriber : IDisposable
{
    /// <summary>Human-readable description of what is actually running, e.g. "Whisper base.en on NPU".</summary>
    string Description { get; }

    /// <summary>
    /// Says that the windows from here on belong to a different recording.
    /// <para>
    /// A backend may carry something between windows that is true of one recording and false of
    /// the next. Whisper does: it is asked to name the language at the start of every decode and
    /// answers differently on different passes over the same audio, so the answer is settled once
    /// and asserted thereafter. Settled once <em>per recording</em> — held across recordings, a
    /// language detected from English speech is then forced onto the next file, and Whisper asked
    /// to transcribe Portuguese as English does not refuse. It translates.
    /// </para>
    /// <para>
    /// Empty by default, because a backend that remembers nothing has nothing to forget.
    /// </para>
    /// </summary>
    /// <param name="task">
    /// Whether to write the recording down as spoken or render it in English. Settled here
    /// rather than per window, so one transcript cannot end up with two conventions.
    /// </param>
    void BeginRecording(SpeechTask task = SpeechTask.Transcribe)
    {
    }

    /// <summary>Transcribes one encoder-sized window.</summary>
    /// <param name="chunk">A window produced by <see cref="AudioChunker"/>.</param>
    /// <param name="cancellationToken">Cancels a long-running window.</param>
    Task<IReadOnlyList<TranscriptSegment>> TranscribeChunkAsync(
        AudioChunk chunk,
        CancellationToken cancellationToken = default);
}
