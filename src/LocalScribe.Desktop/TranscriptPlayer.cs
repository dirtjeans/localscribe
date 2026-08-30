using System.Runtime.InteropServices;
using LocalScribe.Core.Audio;
using LocalScribe.Desktop;

// The namespace is the WinUI app's, deliberately: MainViewModel is compiled into this project
// from its original file, and it resolves TranscriptPlayer, MicrophoneCapture and
// AudioFileLoader by these names. Matching the namespace is what lets the view model stay one
// file with one history instead of two slowly-diverging copies.
namespace LocalScribe.App;

/// <summary>
/// Plays back the audio a transcript came from, from any point in it — the macOS twin of the
/// NAudio player, holding to its two paid-for rules: play the decoded samples the transcriber
/// was given (never re-decode; a second decode does not have to agree with the first about
/// where anything is), and report the device's position rather than the read position. Here
/// the gap between the two is one miniaudio period (~10 ms) instead of WASAPI's few hundred,
/// and the clock diagnostic stays the judge of it.
/// </summary>
public sealed class TranscriptPlayer : IDisposable
{
    private readonly object _lock = new();

    private float[] _samples = [];
    private int _sampleRate = PcmAudio.WhisperSampleRate;
    private GCHandle _pin;
    private Timer? _ticker;
    private bool _playing;
    private double _startedAtSeconds;

    /// <summary>Raised off the UI thread as the position moves, twenty times a second.</summary>
    public event Action<double>? PositionChanged;

    /// <summary>Raised when playback reaches the end or is stopped.</summary>
    public event Action? Stopped;

    /// <summary>Raised when playback could not start, with something worth showing the user.</summary>
    public event Action<string>? Failed;

    /// <summary>True while sound is coming out.</summary>
    public bool IsPlaying
    {
        get
        {
            lock (_lock)
            {
                return _playing;
            }
        }
    }

    /// <summary>Length of the loaded audio.</summary>
    public double DurationSeconds => _samples.Length / (double)_sampleRate;

    /// <summary>True when there is anything to play.</summary>
    public bool HasAudio => _samples.Length > 0;

    /// <summary>The loaded audio reduced to peaks for drawing.</summary>
    public float[] Peaks(int buckets)
    {
        lock (_lock)
        {
            return WaveformPeaks.Compute(_samples, buckets);
        }
    }

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

    /// <summary>Forgets the audio, releasing it and the device.</summary>
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

            Teardown();

            // Pinned for the duration: the audio thread reads this array directly, which is
            // what makes playback the same samples the transcript's timings describe.
            _pin = GCHandle.Alloc(_samples, GCHandleType.Pinned);

            var fromFrame = (ulong)Math.Clamp((long)(seconds * _sampleRate), 0, _samples.Length);

            var result = NativeAudio.PlayStart(
                _pin.AddrOfPinnedObject(), (ulong)_samples.Length, (uint)_sampleRate, fromFrame);

            if (result != 0)
            {
                Teardown();
                Failed?.Invoke($"Could not start playback (miniaudio error {result}).");
                return;
            }

            _playing = true;
            _startedAtSeconds = seconds;
            _wall.Restart();
            _nextLogAtSeconds = 0;
            Log(FormattableString.Invariant($"-- playing from {seconds:F2}s"));

            _ticker = new Timer(OnTick, null, dueTime: 50, period: 50);
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

    private void OnTick(object? state)
    {
        bool finished;
        double position;

        lock (_lock)
        {
            if (!_playing)
            {
                return;
            }

            finished = NativeAudio.PlayFinished() != 0;
            position = NativeAudio.PlayPosition() / (double)_sampleRate;

            if (finished)
            {
                Teardown();
            }
        }

        if (finished)
        {
            Stopped?.Invoke();
            return;
        }

        PositionChanged?.Invoke(position);

        var wall = _wall.Elapsed.TotalSeconds;

        if (wall >= _nextLogAtSeconds)
        {
            _nextLogAtSeconds = wall + 5;
            var device = position - _startedAtSeconds;
            Log(FormattableString.Invariant(
                $"at {position,7:F2}  wall {wall,7:F2}  device {device,7:F2}  gap {wall - device,6:F2}"));
        }
    }

    /// <summary>
    /// Real time since Play, for checking the device's counter rather than trusting it — the
    /// same instrument as on Windows, writing the same file. The gap column should hold steady
    /// at about one device period; a gap that climbs is the counter under-counting.
    /// </summary>
    private readonly System.Diagnostics.Stopwatch _wall = new();
    private double _nextLogAtSeconds;

    private static void Log(string line)
    {
        try
        {
            File.AppendAllText(
                Path.Combine(Path.GetTempPath(), "localscribe-clock.txt"),
                line + Environment.NewLine);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            // A diagnostic that cannot be written is not worth interrupting playback over.
        }
    }

    private void Teardown()
    {
        _ticker?.Dispose();
        _ticker = null;

        if (_playing)
        {
            NativeAudio.PlayStop();
            _playing = false;
        }

        if (_pin.IsAllocated)
        {
            _pin.Free();
        }
    }

    public void Dispose()
    {
        lock (_lock)
        {
            Teardown();
            _samples = [];
        }
    }
}
