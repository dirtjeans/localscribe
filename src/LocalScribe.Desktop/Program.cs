using Avalonia;

namespace LocalScribe.Desktop;

internal static class Program
{
    /// <summary>
    /// A file path argument opens that recording on launch — the same contract as the WinUI
    /// app, and for the same reason: it is how debugging drives the window headlessly.
    /// </summary>
    [STAThread]
    public static void Main(string[] args) => BuildAvaloniaApp()
        .StartWithClassicDesktopLifetime(args);

    public static AppBuilder BuildAvaloniaApp() => AppBuilder
        .Configure<App>()
        .UsePlatformDetect()
        .LogToTrace();
}
