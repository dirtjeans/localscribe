using Microsoft.UI.Xaml;

namespace LocalScribe.App;

public partial class App : Application
{
    private Window? _window;

    public App()
    {
        InitializeComponent();
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        var window = new MainWindow();
        _window = window;

        window.Activate();

        // A file named on the command line is opened once the window is up. This is what makes
        // "Open with" and dragging a recording onto the icon work, and it is the only way to
        // reach the app with a file that does not involve driving the picker by hand.
        if (FileArgument() is { } path)
        {
            window.OpenWhenReady(path);
        }
    }

    /// <summary>
    /// The first argument that names a file that exists. Anything else is ignored rather than
    /// reported: a shell can pass flags, and a startup that fails over an unrecognised one is
    /// worse than one that quietly opens empty.
    /// </summary>
    private static string? FileArgument() =>
        Environment.GetCommandLineArgs()
            .Skip(1)
            .FirstOrDefault(argument => !argument.StartsWith('-') && File.Exists(argument));
}
