namespace LocalScribe.Core.Audio;

/// <summary>
/// Mono 16 kHz float samples, the only format Whisper accepts. Anything arriving from a file
/// or a microphone is resampled to this before it reaches the model.
/// </summary>
/// <param name="Samples">Sample values, nominally in [-1, 1].</param>
/// <param name="SampleRate">Samples per second. Whisper requires 16000.</param>
public sealed record PcmAudio(float[] Samples, int SampleRate = PcmAudio.WhisperSampleRate)
{
    public const int WhisperSampleRate = 16_000;

    /// <summary>Length of the audio in seconds.</summary>
    public double DurationSeconds => Samples.Length / (double)SampleRate;

    /// <summary>Throws when the audio is not in the format the model expects.</summary>
    public void EnsureWhisperFormat()
    {
        if (SampleRate != WhisperSampleRate)
        {
            throw new InvalidOperationException(
                $"Whisper needs {WhisperSampleRate} Hz audio but got {SampleRate} Hz. Resample first.");
        }
    }
}
