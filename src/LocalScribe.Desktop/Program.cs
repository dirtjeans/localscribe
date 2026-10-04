using Avalonia;

namespace LocalScribe.Desktop;

internal static class Program
{
    /// <summary>
    /// A file path argument opens that recording on launch — the same contract as the WinUI
    /// app. <c>--headless &lt;file&gt;</c> runs the same pipeline with no window at all, which is
    /// the only way to drive it while the display is asleep or the screen is locked.
    /// </summary>
    [STAThread]
    public static int Main(string[] args)
    {
        if (args is ["--headless", var path, ..])
        {
            return Headless.Run(path, args is [_, _, "--save", var saveTo] ? saveTo : null);
        }

        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        return 0;
    }

    public static AppBuilder BuildAvaloniaApp() => AppBuilder
        .Configure<App>()
        .UsePlatformDetect()
        .LogToTrace();
}
