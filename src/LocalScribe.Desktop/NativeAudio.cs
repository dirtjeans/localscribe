using System.Runtime.InteropServices;

namespace LocalScribe.Desktop;

/// <summary>
/// The miniaudio wrapper (native/localscribe_audio.c), one function per thing the app can do
/// with sound. The library holds one playback device and one capture device, which is exactly
/// the app's shape — one player, one microphone — so the .NET side adds no session handles.
/// </summary>
internal static partial class NativeAudio
{
    private const string Library = "localscribe-audio";

    [LibraryImport(Library, EntryPoint = "ls_play_start")]
    internal static partial int PlayStart(IntPtr samples, ulong count, uint sampleRate, ulong fromFrame);

    [LibraryImport(Library, EntryPoint = "ls_play_stop")]
    internal static partial void PlayStop();

    [LibraryImport(Library, EntryPoint = "ls_play_position")]
    internal static partial ulong PlayPosition();

    [LibraryImport(Library, EntryPoint = "ls_play_finished")]
    internal static partial int PlayFinished();

    [LibraryImport(Library, EntryPoint = "ls_capture_start")]
    internal static unsafe partial int CaptureStart(
        delegate* unmanaged[Cdecl]<float*, uint, void> handler,
        uint sampleRate,
        uint bufferMilliseconds);

    [LibraryImport(Library, EntryPoint = "ls_capture_stop")]
    internal static partial void CaptureStop();
}
