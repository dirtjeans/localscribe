using LocalScribe.Core.Audio;
using NAudio.Wave;

namespace LocalScribe.App;

/// <summary>
/// Reads audio files and converts them into the format Whisper needs.
/// <para>
/// Everything the model sees is mono 16 kHz float. Files arrive as stereo 44.1 kHz MP3s and
/// worse, so the conversion happens once, here, rather than being guarded against everywhere
/// downstream.
/// </para>
/// </summary>
public static class AudioFileLoader
{
    /// <summary>
    /// Containers that hold video as well as audio. NAudio's AudioFileReader does not open
    /// these; Media Foundation does, and pulls the audio track out on the way through.
    /// </summary>
    private static readonly HashSet<string> VideoExtensions =
        new(StringComparer.OrdinalIgnoreCase) { ".mp4", ".mov", ".m4v", ".avi", ".wmv", ".mkv", ".webm" };

    /// <summary>Every extension the picker should offer.</summary>
    public static IReadOnlyList<string> SupportedExtensions { get; } =
    [
        ".wav", ".mp3", ".m4a", ".flac", ".wma", ".aac", ".aiff", ".ogg",
        ".mp4", ".mov", ".m4v", ".avi", ".wmv", ".mkv", ".webm",
    ];

    /// <summary>True when the file is a video container rather than plain audio.</summary>
    public static bool IsVideo(string path) =>
        VideoExtensions.Contains(Path.GetExtension(path));

    /// <summary>
    /// Loads and resamples anything Windows can decode: wav, mp3, m4a and the rest, and the
    /// audio track of a video file.
    /// </summary>
    public static PcmAudio Load(string path)
    {
        using var reader = OpenReader(path);
        using var resampler = new MediaFoundationResampler(
            reader,
            WaveFormat.CreateIeeeFloatWaveFormat(PcmAudio.WhisperSampleRate, channels: 1))
        {
            ResamplerQuality = 60,
        };

        var samples = new List<float>();
        var buffer = new byte[PcmAudio.WhisperSampleRate * sizeof(float)];
        int bytesRead;

        while ((bytesRead = resampler.Read(buffer, 0, buffer.Length)) > 0)
        {
            for (var offset = 0; offset + sizeof(float) <= bytesRead; offset += sizeof(float))
            {
                samples.Add(BitConverter.ToSingle(buffer, offset));
            }
        }

        return new PcmAudio([.. samples]);
    }

    /// <summary>
    /// Opens whichever reader can handle the file.
    /// <para>
    /// Video goes straight to Media Foundation. Audio prefers AudioFileReader, which handles the
    /// common formats without involving the platform decoders at all, and falls back to Media
    /// Foundation for anything it does not recognise rather than failing in front of the user.
    /// </para>
    /// </summary>
    private static WaveStream OpenReader(string path)
    {
        if (IsVideo(path))
        {
            return new MediaFoundationReader(path);
        }

        try
        {
            return new AudioFileReader(path);
        }
        catch (Exception exception) when (exception is not FileNotFoundException)
        {
            return new MediaFoundationReader(path);
        }
    }
}

/// <summary>
/// Captures the default microphone and hands out 16 kHz mono float frames.
/// <para>
/// Capture runs on its own thread inside NAudio and raises events as buffers fill. Those events
/// arrive off the UI thread, which is exactly what we want: the transcription work is already
/// arranged to stay off it.
/// </para>
/// </summary>
public sealed class MicrophoneCapture : IDisposable
{
    private readonly WaveInEvent _waveIn;

    /// <summary>Raised whenever a new buffer of converted samples is ready.</summary>
    public event Action<float[]>? SamplesAvailable;

    public MicrophoneCapture(int bufferMilliseconds = 200)
    {
        _waveIn = new WaveInEvent
        {
            // Capturing directly at the model's rate avoids a resampling stage in the hot path.
            WaveFormat = new WaveFormat(PcmAudio.WhisperSampleRate, bits: 16, channels: 1),
            BufferMilliseconds = bufferMilliseconds,
        };

        _waveIn.DataAvailable += OnDataAvailable;
    }

    public void Start() => _waveIn.StartRecording();

    public void Stop() => _waveIn.StopRecording();

    private void OnDataAvailable(object? sender, WaveInEventArgs e)
    {
        var samples = new float[e.BytesRecorded / 2];
        for (var i = 0; i < samples.Length; i++)
        {
            // 16-bit signed PCM to normalised float.
            samples[i] = BitConverter.ToInt16(e.Buffer, i * 2) / 32768f;
        }

        SamplesAvailable?.Invoke(samples);
    }

    public void Dispose()
    {
        _waveIn.DataAvailable -= OnDataAvailable;
        _waveIn.Dispose();
    }
}
