using Avalonia;

namespace LocalScribe.Desktop;

internal static class Program
{
    /// <summary>
    /// A file path argument opens that recording on launch — the same contract as the WinUI
    /// app. <c>--headless &lt;file&gt;</c> runs the same pipeline with no window at all, which is
    /// the only way to drive it while the display is asleep or the screen is locked. Several
    /// files are joined, in the order the window would propose, as the window would join them.
    /// </summary>
    [STAThread]
    public static int Main(string[] args)
    {
        if (args is ["--headless", ..])
        {
            var save = Array.IndexOf(args, "--save");
            var files = args[1..(save < 0 ? args.Length : save)];
            return Headless.Run(files, save >= 0 && save + 1 < args.Length ? args[save + 1] : null);
        }

        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        return 0;
    }

    public static AppBuilder BuildAvaloniaApp() => AppBuilder
        .Configure<App>()
        .UsePlatformDetect()
        .LogToTrace();
}
