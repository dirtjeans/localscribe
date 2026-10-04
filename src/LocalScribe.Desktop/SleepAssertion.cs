using System.Runtime.InteropServices;

namespace LocalScribe.Desktop;

/// <summary>
/// Tells macOS the app is doing user-initiated work while models run, and only then.
/// <para>
/// Two different things go wrong without it. An idle machine sleeps mid-transcription, which
/// reads as "diarization is stuck". And — the subtler one, found by a run that stalled at the
/// preview with the display asleep while the same file finished from the command line — App
/// Nap: once no window is visible, macOS throttles the app's run loop, and every stage hands
/// its results back through that run loop. A power assertion stops only the first. An
/// NSProcessInfo activity stops both, which is what this holds, for exactly the stretch the
/// working glow shows. A closed lid still sleeps the machine; that is the user's call.
/// </para>
/// </summary>
internal static partial class SleepAssertion
{
    private const string ObjC = "/usr/lib/libobjc.A.dylib";

    // NSActivityUserInitiated: IdleSystemSleepDisabled plus the flags that keep App Nap off.
    private const ulong UserInitiated = 0x00FFFFFFUL;

    private static IntPtr _activity;

    [LibraryImport(ObjC, StringMarshalling = StringMarshalling.Utf8)]
    private static partial IntPtr objc_getClass(string name);

    [LibraryImport(ObjC, StringMarshalling = StringMarshalling.Utf8)]
    private static partial IntPtr sel_registerName(string name);

    [LibraryImport(ObjC, EntryPoint = "objc_msgSend")]
    private static partial IntPtr Send(IntPtr receiver, IntPtr selector);

    [LibraryImport(ObjC, EntryPoint = "objc_msgSend", StringMarshalling = StringMarshalling.Utf8)]
    private static partial IntPtr SendString(IntPtr receiver, IntPtr selector, string value);

    [LibraryImport(ObjC, EntryPoint = "objc_msgSend")]
    private static partial IntPtr SendBegin(IntPtr receiver, IntPtr selector, ulong options, IntPtr reason);

    [LibraryImport(ObjC, EntryPoint = "objc_msgSend")]
    private static partial void SendEnd(IntPtr receiver, IntPtr selector, IntPtr activity);

    public static void While(bool working)
    {
        if (!OperatingSystem.IsMacOS() || working == (_activity != IntPtr.Zero))
        {
            return;
        }

        var processInfo = Send(objc_getClass("NSProcessInfo"), sel_registerName("processInfo"));

        if (working)
        {
            var reason = SendString(
                objc_getClass("NSString"), sel_registerName("stringWithUTF8String:"),
                "LocalScribe is transcribing");

            var activity = SendBegin(
                processInfo, sel_registerName("beginActivityWithOptions:reason:"), UserInitiated, reason);

            // Retained, because the token is autoreleased and the activity ends when it dies.
            _activity = Send(activity, sel_registerName("retain"));
        }
        else
        {
            SendEnd(processInfo, sel_registerName("endActivity:"), _activity);
            Send(_activity, sel_registerName("release"));
            _activity = IntPtr.Zero;
        }
    }
}
