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

    /// <summary>Transcribes one encoder-sized window.</summary>
    /// <param name="chunk">A window produced by <see cref="AudioChunker"/>.</param>
    /// <param name="cancellationToken">Cancels a long-running window.</param>
    Task<IReadOnlyList<TranscriptSegment>> TranscribeChunkAsync(
        AudioChunk chunk,
        CancellationToken cancellationToken = default);
}
