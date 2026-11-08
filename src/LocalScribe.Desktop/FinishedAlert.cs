using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Avalonia.Threading;
using LocalScribe.Core.Hardware;

namespace LocalScribe.Desktop;

/// <summary>
/// Tells someone who walked away that the run has ended: a macOS notification when they have
/// allowed one, and always a bounce of the Dock icon, which needs no permission. Without
/// permission a sound takes the notification's place, so turning notifications off for the app
/// still leaves an alert that can be heard.
/// <para>
/// The system frameworks are reached through the Objective-C runtime, as
/// <see cref="SleepAssertion"/> does. UNUserNotificationCenter raises an uncatchable exception
/// in a process with no bundle identifier, which is any run outside LocalScribe.app, so every
/// route to it goes through <see cref="Bundled"/>.
/// </para>
/// </summary>
internal static unsafe partial class FinishedAlert
{
    private const string ObjC = "/usr/lib/libobjc.A.dylib";
    private const string Notifications = "/System/Library/Frameworks/UserNotifications.framework/UserNotifications";
    private const string AppKit = "/System/Library/Frameworks/AppKit.framework/AppKit";

    private const ulong AlertAndSound = 2 | 4;     // UNAuthorizationOptionSound | Alert
    private const ulong InformationalRequest = 10; // NSInformationalRequest: one bounce, not until clicked

    private const int BlockIsGlobal = 1 << 28;

    [LibraryImport(ObjC, StringMarshalling = StringMarshalling.Utf8)]
    private static partial IntPtr objc_getClass(string name);

    [LibraryImport(ObjC, StringMarshalling = StringMarshalling.Utf8)]
    private static partial IntPtr sel_registerName(string name);

    [LibraryImport(ObjC, EntryPoint = "objc_msgSend")]
    private static partial IntPtr Send(IntPtr receiver, IntPtr selector);

    [LibraryImport(ObjC, EntryPoint = "objc_msgSend")]
    private static partial IntPtr Send(IntPtr receiver, IntPtr selector, IntPtr argument);

    [LibraryImport(ObjC, EntryPoint = "objc_msgSend")]
    private static partial IntPtr Send(IntPtr receiver, IntPtr selector, IntPtr first, IntPtr second);

    [LibraryImport(ObjC, EntryPoint = "objc_msgSend")]
    private static partial IntPtr Send(IntPtr receiver, IntPtr selector, IntPtr first, IntPtr second, IntPtr third);

    [LibraryImport(ObjC, EntryPoint = "objc_msgSend")]
    private static partial IntPtr SendNumber(IntPtr receiver, IntPtr selector, ulong value);

    [LibraryImport(ObjC, EntryPoint = "objc_msgSend")]
    private static partial void SendOptions(IntPtr receiver, IntPtr selector, ulong options, Block* handler);

    [LibraryImport(ObjC, EntryPoint = "objc_msgSend", StringMarshalling = StringMarshalling.Utf8)]
    private static partial IntPtr SendString(IntPtr receiver, IntPtr selector, string value);

    [StructLayout(LayoutKind.Sequential)]
    private struct Block
    {
        public IntPtr Isa;
        public int Flags;
        public int Reserved;
        public delegate* unmanaged[Cdecl]<Block*, byte, IntPtr, void> Invoke;
        public BlockDescriptor* Descriptor;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BlockDescriptor
    {
        public nuint Reserved;
        public nuint Size;
    }

    private static volatile bool _allowed;
    private static Action<string?>? _answered;

    /// <summary>True when notifications can be asked for at all.</summary>
    private static bool Bundled
    {
        get
        {
            if (!OperatingSystem.IsMacOS())
            {
                return false;
            }

            var bundle = Send(objc_getClass("NSBundle"), sel_registerName("mainBundle"));
            return bundle != IntPtr.Zero && Send(bundle, sel_registerName("bundleIdentifier")) != IntPtr.Zero;
        }
    }

    /// <summary>
    /// Asks macOS for permission, which shows its own prompt the first time and answers at once
    /// after that. Asked only once someone has ticked the box, never on launch.
    /// </summary>
    /// <param name="answered">Null when allowed; otherwise why not, on the UI thread.</param>
    public static void AskPermission(Action<string?> answered)
    {
        if (!Bundled)
        {
            answered("Notifications need the installed app.");
            return;
        }

        try
        {
            NativeLibrary.Load(Notifications);
            _answered = answered;

            var size = (nuint)sizeof(Block);
            var descriptor = (BlockDescriptor*)NativeMemory.Alloc((nuint)sizeof(BlockDescriptor));
            *descriptor = new BlockDescriptor { Size = size };

            // Global, so it needs no captures and the system's copy of it is this same one. Held
            // for the life of the process: it is small, and the system may call it at any time.
            var block = (Block*)NativeMemory.Alloc(size);
            *block = new Block
            {
                Isa = NativeLibrary.GetExport(NativeLibrary.Load("/usr/lib/libSystem.B.dylib"), "_NSConcreteGlobalBlock"),
                Flags = BlockIsGlobal,
                Invoke = &OnAnswer,
                Descriptor = descriptor,
            };

            SendOptions(
                Center(),
                sel_registerName("requestAuthorizationWithOptions:completionHandler:"),
                AlertAndSound,
                block);
        }
        catch (Exception exception) when (exception is DllNotFoundException or EntryPointNotFoundException)
        {
            answered(exception.Message);
        }
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void OnAnswer(Block* self, byte granted, IntPtr error)
    {
        _allowed = granted != 0;

        string? why = null;

        if (!_allowed)
        {
            why = "macOS has notifications turned off for LocalScribe.";

            if (error != IntPtr.Zero)
            {
                var description = Send(error, sel_registerName("localizedDescription"));
                var text = Marshal.PtrToStringUTF8(Send(description, sel_registerName("UTF8String")));
                why = text ?? why;
            }
        }

        var answered = _answered;
        Dispatcher.UIThread.Post(() => answered?.Invoke(why));
    }

    /// <summary>Alerts now. On the UI thread: AppKit's calls are not safe anywhere else.</summary>
    public static void Show(string title, string body)
    {
        if (!OperatingSystem.IsMacOS())
        {
            return;
        }

        if (_allowed && Bundled)
        {
            PostNotification(title, body);
        }
        else
        {
            Sound();
        }

        Bounce();
    }

    private static IntPtr Center() =>
        Send(objc_getClass("UNUserNotificationCenter"), sel_registerName("currentNotificationCenter"));

    private static IntPtr String(string value) =>
        SendString(objc_getClass("NSString"), sel_registerName("stringWithUTF8String:"), value);

    private static void PostNotification(string title, string body)
    {
        var content = Send(
            Send(objc_getClass("UNMutableNotificationContent"), sel_registerName("alloc")),
            sel_registerName("init"));

        Send(content, sel_registerName("setTitle:"), String(title));
        Send(content, sel_registerName("setBody:"), String(body));
        Send(
            content,
            sel_registerName("setSound:"),
            Send(objc_getClass("UNNotificationSound"), sel_registerName("defaultSound")));

        var request = Send(
            objc_getClass("UNNotificationRequest"),
            sel_registerName("requestWithIdentifier:content:trigger:"),
            String(Guid.NewGuid().ToString()),
            content,
            IntPtr.Zero);

        Send(Center(), sel_registerName("addNotificationRequest:withCompletionHandler:"), request, IntPtr.Zero);
        Send(content, sel_registerName("release"));
    }

    private static void Sound()
    {
        NativeLibrary.Load(AppKit);

        var sound = Send(objc_getClass("NSSound"), sel_registerName("soundNamed:"), String("Glass"));

        if (sound != IntPtr.Zero)
        {
            Send(sound, sel_registerName("play"));
        }
    }

    private static void Bounce()
    {
        NativeLibrary.Load(AppKit);

        var application = Send(objc_getClass("NSApplication"), sel_registerName("sharedApplication"));

        if (application != IntPtr.Zero)
        {
            SendNumber(application, sel_registerName("requestUserAttention:"), InformationalRequest);
        }
    }

    private static string PreferencePath =>
        Path.Combine(Path.GetDirectoryName(WorkPaces.DefaultPath)!, "alert-when-done.txt");

    /// <summary>Whether the box was left ticked. Off until someone asks, since it comes with a system prompt.</summary>
    public static bool Wanted
    {
        get
        {
            try
            {
                return File.Exists(PreferencePath) && File.ReadAllText(PreferencePath).Trim() == "on";
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                return false;
            }
        }

        set
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(PreferencePath)!);
                File.WriteAllText(PreferencePath, value ? "on" : "off");
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                // The box still applies for this session.
            }
        }
    }
}
