using LocalScribe.Core.Audio;

namespace LocalScribe.Core.Diarization;

/// <summary>
/// Splits a recording into speaker turns.
/// <para>
/// Diarization is a whole-recording problem, not a per-window one: deciding that the voice at
/// minute two is the same person as the voice at minute forty means comparing them, so the
/// interface takes the entire audio rather than the chunks transcription works on.
/// </para>
/// </summary>
public interface IDiarizer : IDisposable
{
    /// <summary>Human-readable description of the models in use, for the doctor and logs.</summary>
    string Description { get; }

    /// <summary>Finds speaker turns across a whole recording.</summary>
    Task<IReadOnlyList<SpeakerTurn>> DiarizeAsync(
        PcmAudio audio,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default);
}
