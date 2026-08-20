using LocalScribe.Core.Audio;
using NAudio.Wave;

namespace LocalScribe.App;

/// <summary>
/// Plays back the audio a transcript came from, from any point in it.
/// <para>
/// Plays the decoded samples the transcriber was given rather than re-opening the source file.
/// That is what makes a click on a word land on the right sound: the timings in the transcript
/// are offsets into exactly this array, and a second decode of an MP3 does not have to agree
/// with the first about where anything is. It also means a live recording, which has no file at
/// all, can be replayed on the same path as a transcribed one.
/// </para>
/// </summary>
public sealed class TranscriptPlayer : IDisposable
{
    private readonly object _lock = new();

    private WaveOutEvent? _output;
    private PositionedSampleProvider? _source;
    private float[] _samples = [];
    private int _sampleRate = PcmAudio.WhisperSampleRate;

    /// <summary>Raised on the playback thread as the position moves, roughly ten times a second.</summary>
    public event Action<double>? PositionChanged;

    /// <summary>Raised when playback reaches the end or is stopped.</summary>
    public event Action? Stopped;

    /// <summary>
    /// Raised when playback could not start, with something worth showing the user. Audio output
    /// fails for ordinary reasons — no device, a device in exclusive use — and a play button that
    /// does nothing at all is the worst way to report any of them.
    /// </summary>
    public event Action<string>? Failed;

    /// <summary>True while sound is coming out.</summary>
    public bool IsPlaying => _output?.PlaybackState == PlaybackState.Playing;

    /// <summary>Length of the loaded audio.</summary>
    public double DurationSeconds => _samples.Length / (double)_sampleRate;

    /// <summary>True when there is anything to play.</summary>
    public bool HasAudio => _samples.Length > 0;

    /// <summary>Hands the player the audio a transcript describes.</summary>
    public void Load(PcmAudio audio)
    {
        ArgumentNullException.ThrowIfNull(audio);

        lock (_lock)
        {
            Teardown();
            _samples = audio.Samples;
            _sampleRate = audio.SampleRate;
        }
    }

    /// <summary>Forgets the audio, releasing it and any device held.</summary>
    public void Clear()
    {
        lock (_lock)
        {
            Teardown();
            _samples = [];
        }

        Stopped?.Invoke();
    }

    /// <summary>Starts playing from a position, restarting if already playing.</summary>
    public void PlayFrom(double seconds)
    {
        lock (_lock)
        {
            if (_samples.Length == 0)
            {
                Failed?.Invoke("There is no audio loaded to play.");
                return;
            }

            try
            {
                Teardown();

                _source = new PositionedSampleProvider(_samples, _sampleRate);
                _source.Seek(seconds);
                _source.PositionChanged += OnPositionChanged;

                _output = new WaveOutEvent();
                _output.PlaybackStopped += (_, _) => Stopped?.Invoke();

                // Converted to 16-bit rather than handed over as float. The samples are float
                // and WaveOutEvent will accept an ISampleProvider directly, but that hands the
                // legacy waveOut API a 32-bit IEEE float format, which plenty of drivers simply
                // refuse. It fails at Play with nothing audible and nothing thrown that reaches
                // a click handler, which is exactly how this looked: a transcript that could be
                // clicked all day and never made a sound.
                _output.Init(_source.ToWaveProvider16());
                _output.Play();
            }
            catch (Exception exception)
            {
                Teardown();
                Failed?.Invoke($"Could not start playback: {exception.Message}");
            }
        }
    }

    /// <summary>Stops, keeping the audio loaded.</summary>
    public void Stop()
    {
        lock (_lock)
        {
            Teardown();
        }

        Stopped?.Invoke();
    }

    private void OnPositionChanged(double seconds) => PositionChanged?.Invoke(seconds);

    private void Teardown()
    {
        if (_source is not null)
        {
            _source.PositionChanged -= OnPositionChanged;
            _source = null;
        }

        _output?.Dispose();
        _output = null;
    }

    public void Dispose()
    {
        lock (_lock)
        {
            Teardown();
            _samples = [];
        }
    }

    /// <summary>
    /// Reads from a float array and reports where it has got to. NAudio can play a stream or a
    /// file; this plays an array we already hold, which is the only thing whose timings are
    /// guaranteed to match the transcript.
    /// </summary>
    private sealed class PositionedSampleProvider(float[] samples, int sampleRate) : ISampleProvider
    {
        private int _position;
        private int _sinceLastReport;

        public event Action<double>? PositionChanged;

        public WaveFormat WaveFormat { get; } =
            WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channels: 1);

        public void Seek(double seconds) =>
            _position = Math.Clamp((int)(seconds * sampleRate), 0, samples.Length);

        public int Read(float[] buffer, int offset, int count)
        {
            var available = Math.Min(count, samples.Length - _position);
            if (available <= 0)
            {
                return 0;
            }

            Array.Copy(samples, _position, buffer, offset, available);
            _position += available;
            _sinceLastReport += available;

            // Reporting every buffer would raise this hundreds of times a second for a UI that
            // cannot use it. A tenth of a second is finer than anyone can see a highlight move.
            if (_sinceLastReport >= sampleRate / 10)
            {
                _sinceLastReport = 0;
                PositionChanged?.Invoke(_position / (double)sampleRate);
            }

            return available;
        }
    }
}
