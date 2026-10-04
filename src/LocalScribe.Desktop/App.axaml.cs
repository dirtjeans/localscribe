using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;

namespace LocalScribe.Desktop;

public sealed class App : Application
{
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    private void OnAboutClicked(object? sender, EventArgs e)
    {
        if (ApplicationLifetime is not IClassicDesktopStyleApplicationLifetime { MainWindow: { } owner })
        {
            return;
        }

        var about = new Avalonia.Controls.Window
        {
            Title = "About LocalScribe",
            SizeToContent = Avalonia.Controls.SizeToContent.WidthAndHeight,
            CanResize = false,
            WindowStartupLocation = Avalonia.Controls.WindowStartupLocation.CenterOwner,
            Content = new Avalonia.Controls.TextBlock
            {
                Margin = new Avalonia.Thickness(28, 22),
                Text = "LocalScribe\n\nOffline transcription. Whisper on the Neural Engine,\n"
                    + "a local language model for cleanup, nothing leaving this Mac.",
                TextAlignment = Avalonia.Media.TextAlignment.Center,
            },
        };

        about.ShowDialog(owner);
    }

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            desktop.MainWindow = new MainWindow(
                desktop.Args is { Length: > 0 } ? desktop.Args[0] : null);
        }

        base.OnFrameworkInitializationCompleted();
    }
}
