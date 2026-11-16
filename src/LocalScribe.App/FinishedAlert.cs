using System.Runtime.InteropServices;
using LocalScribe.Core.Hardware;
using Microsoft.Windows.AppNotifications;
using Microsoft.Windows.AppNotifications.Builder;

namespace LocalScribe.App;

/// <summary>
/// Tells someone who walked away that the run has ended: a Windows notification when they
/// allow them, and always a flash of the taskbar button, which needs no permission. With
/// notifications turned off a sound takes their place, so an alert can still be heard. The
/// Mac's FinishedAlert, with Windows' own three equivalents.
/// </summary>
internal static class FinishedAlert
{
    private static bool _registered;

    /// <summary>
    /// Registers with the notification platform, which an unpackaged app does at run time.
    /// Only once someone has ticked the box: registering writes the app's identity into the
    /// user's registry, and nobody asked for that by launching it.
    /// </summary>
    /// <returns>Null when notifications can be shown; otherwise why not.</returns>
    public static string? Register()
    {
        try
        {
            if (!_registered)
            {
                AppNotificationManager.Default.Register();
                _registered = true;
            }

            return AppNotificationManager.Default.Setting == AppNotificationSetting.Enabled
                ? null
                : "Windows has notifications turned off for LocalScribe.";
        }
        catch (Exception exception) when (exception is COMException or InvalidOperationException or UnauthorizedAccessException)
        {
            return exception.Message;
        }
    }

    /// <summary>
    /// Unregisters on the way out, so a notification clicked after the window closed does not
    /// relaunch an empty one.
    /// </summary>
    public static void Unregister()
    {
        if (!_registered)
        {
            return;
        }

        try
        {
            AppNotificationManager.Default.Unregister();
        }
        catch (Exception exception) when (exception is COMException or InvalidOperationException)
        {
            // Nothing to undo that matters at exit.
        }

        _registered = false;
    }

    /// <summary>Alerts now, for the window whose taskbar button should flash.</summary>
    public static void Show(nint window, string title, string body)
    {
        if (Register() is null)
        {
            try
            {
                AppNotificationManager.Default.Show(
                    new AppNotificationBuilder().AddText(title).AddText(body).BuildNotification());
            }
            catch (Exception exception) when (exception is COMException or InvalidOperationException)
            {
                MessageBeep(MbIconInformation);
            }
        }
        else
        {
            MessageBeep(MbIconInformation);
        }

        // Until the window is brought forward, as a waiting application's button does, rather
        // than a few blinks that end before anyone looks back.
        var flash = new FlashInfo
        {
            Size = (uint)Marshal.SizeOf<FlashInfo>(),
            Window = window,
            Flags = FlashTray | FlashUntilForeground,
        };
        FlashWindowEx(ref flash);
    }

    private static string PreferencePath =>
        Path.Combine(Path.GetDirectoryName(WorkPaces.DefaultPath)!, "alert-when-done.txt");

    /// <summary>Whether the box was left ticked. Off until someone asks, as on the Mac.</summary>
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

    private const uint MbIconInformation = 0x40;
    private const uint FlashTray = 0x2;
    private const uint FlashUntilForeground = 0xC;

    [StructLayout(LayoutKind.Sequential)]
    private struct FlashInfo
    {
        public uint Size;
        public nint Window;
        public uint Flags;
        public uint Count;
        public uint Timeout;
    }

    // DllImport, as GetDpiForWindow is, so unsafe code stays off for the whole app.
    [DllImport("user32.dll")]
    private static extern bool FlashWindowEx(ref FlashInfo info);

    [DllImport("user32.dll")]
    private static extern bool MessageBeep(uint type);
}
