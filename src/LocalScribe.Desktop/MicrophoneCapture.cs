using LocalScribe.Core.Audio;
using LocalScribe.Desktop;

namespace LocalScribe.App;

/// <summary>
/// Captures the default microphone and hands out 16 kHz mono float frames — the macOS twin of
/// the WASAPI capture, keeping its decision to capture at the model's own rate so the hot path
/// has no resampling stage. Frames arrive on miniaudio's audio thread, which is what the view
/// model wants: transcription work is already arranged to stay off the UI thread.
/// <para>
/// The capture device is a process-wide singleton in the native layer, so this class allows
/// one live instance at a time — which is also the app's shape: one microphone, one recording.
/// </para>
/// </summary>
public sealed class MicrophoneCapture : IDisposable
{
    private static MicrophoneCapture? _current;

    private readonly uint _bufferMilliseconds;
    private bool _running;

    /// <summary>Raised whenever a new buffer of captured samples is ready.</summary>
    public event Action<float[]>? SamplesAvailable;

    public MicrophoneCapture(int bufferMilliseconds = 200)
    {
        _bufferMilliseconds = (uint)bufferMilliseconds;
    }

    public unsafe void Start()
    {
        if (Interlocked.CompareExchange(ref _current, this, null) is not null && _current != this)
        {
            throw new InvalidOperationException("Another capture is already running.");
        }

        var result = NativeAudio.CaptureStart(
            &OnCaptured, (uint)PcmAudio.WhisperSampleRate, _bufferMilliseconds);

        if (result != 0)
        {
            _current = null;
            throw new InvalidOperationException(
                $"The microphone could not be started (miniaudio error {result}). If this is "
                + "the first recording, check the microphone permission in System Settings — "
                + "without it capture fails with nothing captured.");
        }

        _running = true;
    }

    public void Stop()
    {
        if (!_running)
        {
            return;
        }

        // Blocks until the audio thread has left the callback, so no frame arrives after this
        // returns — the view model unsubscribes first and relies on the tail being short.
        NativeAudio.CaptureStop();

        _running = false;
        _current = null;
    }

    [System.Runtime.InteropServices.UnmanagedCallersOnly(
        CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    private static unsafe void OnCaptured(float* samples, uint count)
    {
        if (_current is not { } capture || count == 0)
        {
            return;
        }

        var frame = new float[count];
        new ReadOnlySpan<float>(samples, (int)count).CopyTo(frame);

        capture.SamplesAvailable?.Invoke(frame);
    }

    public void Dispose() => Stop();
}
