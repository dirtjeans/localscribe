using LocalScribe.Core.Provisioning;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;

namespace LocalScribe.App;

/// <summary>
/// The first thing a new user sees when the machine is not ready yet.
/// <para>
/// Built in code rather than XAML because the content is a list whose length and wording depend
/// on what the probe found, and because it is rebuilt in place after a download rather than
/// reopened.
/// </para>
/// <para>
/// Every style and brush used here is one the main window already uses. That is deliberate: a
/// theme resource that does not resolve throws at runtime, and this dialog is the one screen
/// that has to work on a machine where nothing else does yet.
/// </para>
/// </summary>
internal sealed class SetupDialog
{
    // Segoe Fluent Icons: CheckMark, Warning, and Remove for an absent optional extra.
    private const string InstalledGlyph = "\uE73E";
    private const string BlockingGlyph = "\uE7BA";
    private const string OptionalGlyph = "\uE738";

    private readonly SetupViewModel _setup;
    private readonly StackPanel _body = new() { Spacing = 12 };
    private readonly ProgressBar _progress = new() { Maximum = 1, Visibility = Visibility.Collapsed };
    private readonly TextBlock _status = new() { TextWrapping = TextWrapping.Wrap };
    private readonly ContentDialog _dialog;

    private SetupDialog(SetupViewModel setup, XamlRoot xamlRoot)
    {
        _setup = setup;

        var scroller = new ScrollViewer
        {
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            MaxHeight = 420,
            Content = _body,
        };

        var root = new StackPanel { Spacing = 12, Width = 480 };
        root.Children.Add(scroller);
        root.Children.Add(_progress);
        root.Children.Add(_status);

        _dialog = new ContentDialog
        {
            XamlRoot = xamlRoot,
            Title = "Set up LocalScribe",
            Content = root,
            DefaultButton = ContentDialogButton.Primary,
        };

        _dialog.PrimaryButtonClick += OnInstallRequested;
    }

    /// <summary>Builds the dialog for the current setup state and shows it.</summary>
    public static async Task ShowAsync(SetupViewModel setup, XamlRoot xamlRoot)
    {
        var dialog = new SetupDialog(setup, xamlRoot);
        dialog.Rebuild();
        await dialog._dialog.ShowAsync();
    }

    /// <summary>
    /// Redraws the body from the current setup state. Called before showing and again after a
    /// download, so the ticks and the buttons reflect what is now on disk.
    /// </summary>
    private void Rebuild()
    {
        _body.Children.Clear();
        _body.Children.Add(Intro());

        foreach (var component in _setup.Components)
        {
            _body.Children.Add(Row(component));
        }

        foreach (var component in _setup.ManualActions.Where(c => c.ManualInstructions is not null))
        {
            _body.Children.Add(ManualCard(component));
        }

        _body.Children.Add(new TextBlock
        {
            Text = _setup.Verdict,
            TextWrapping = TextWrapping.Wrap,
            Style = (Style)Application.Current.Resources["CaptionTextBlockStyle"],
            Foreground = (Brush)Application.Current.Resources["TextFillColorSecondaryBrush"],
        });

        _status.Text = _setup.Status;

        // Offering a download that would do nothing is worse than offering none: it invites a
        // click that appears to fail. The button only exists while there is something to fetch.
        _dialog.PrimaryButtonText = _setup.HasWorkToDo ? "Download what's missing" : string.Empty;
        _dialog.IsPrimaryButtonEnabled = _setup.HasWorkToDo && !_setup.IsInstalling;
        _dialog.CloseButtonText = _setup.CanTranscribe ? "Start using LocalScribe" : "Continue anyway";
    }

    private TextBlock Intro() => new()
    {
        Text = _setup.CanTranscribe
            ? "LocalScribe is ready. Everything below runs on this machine; nothing is sent anywhere."
            : "LocalScribe needs a few things before it can transcribe. Downloads come from "
                + "Hugging Face and Microsoft — after that, no audio ever leaves this machine.",
        TextWrapping = TextWrapping.Wrap,
        Style = (Style)Application.Current.Resources["BodyTextBlockStyle"],
    };

    private static Grid Row(ComponentStatus component)
    {
        var grid = new Grid { ColumnSpacing = 12 };
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

        var glyph = component.Installed
            ? InstalledGlyph
            : component.IsBlocking ? BlockingGlyph : OptionalGlyph;

        var icon = new FontIcon { Glyph = glyph, FontSize = 16, VerticalAlignment = VerticalAlignment.Top };
        Grid.SetColumn(icon, 0);
        grid.Children.Add(icon);

        var text = new StackPanel { Spacing = 2 };
        text.Children.Add(new TextBlock
        {
            Text = component.Required ? component.Title : $"{component.Title} (optional)",
            TextWrapping = TextWrapping.Wrap,
            Style = (Style)Application.Current.Resources["BodyStrongTextBlockStyle"],
        });
        text.Children.Add(new TextBlock
        {
            Text = component.Detail,
            TextWrapping = TextWrapping.Wrap,
            Style = (Style)Application.Current.Resources["CaptionTextBlockStyle"],
            Foreground = (Brush)Application.Current.Resources["TextFillColorSecondaryBrush"],
        });

        Grid.SetColumn(text, 1);
        grid.Children.Add(text);

        return grid;
    }

    /// <summary>
    /// Steps needing a person get their own card rather than a line in the list, because the
    /// instructions are long and the app cannot do any of it. The Hexagon driver is the reason
    /// this exists: it is a signed kernel driver behind an account wall, and automating it would
    /// produce a tool that reports success and leaves the app silently on the CPU.
    /// </summary>
    private static Border ManualCard(ComponentStatus component)
    {
        var panel = new StackPanel { Spacing = 4 };
        panel.Children.Add(new TextBlock
        {
            Text = component.Title,
            TextWrapping = TextWrapping.Wrap,
            Style = (Style)Application.Current.Resources["BodyStrongTextBlockStyle"],
        });
        panel.Children.Add(new TextBlock
        {
            Text = component.ManualInstructions ?? string.Empty,
            TextWrapping = TextWrapping.Wrap,
            Style = (Style)Application.Current.Resources["CaptionTextBlockStyle"],
            Foreground = (Brush)Application.Current.Resources["TextFillColorSecondaryBrush"],
        });

        return new Border
        {
            Background = (Brush)Application.Current.Resources["CardBackgroundFillColorDefaultBrush"],
            BorderBrush = (Brush)Application.Current.Resources["CardStrokeColorDefaultBrush"],
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(8),
            Padding = new Thickness(12),
            Child = panel,
        };
    }

    /// <summary>
    /// Runs the download without letting the dialog close under it. A ContentDialog dismisses
    /// itself on a button click by default, which would hide a download the moment it started.
    /// </summary>
    private async void OnInstallRequested(ContentDialog sender, ContentDialogButtonClickEventArgs args)
    {
        var deferral = args.GetDeferral();
        args.Cancel = true;

        try
        {
            _dialog.IsPrimaryButtonEnabled = false;
            _progress.Visibility = Visibility.Visible;

            void OnSetupChanged(object? _, System.ComponentModel.PropertyChangedEventArgs e)
            {
                switch (e.PropertyName)
                {
                    case nameof(SetupViewModel.Status):
                        _status.Text = _setup.Status;
                        break;
                    case nameof(SetupViewModel.Progress):
                        _progress.Value = _setup.Progress;
                        _progress.IsIndeterminate = _setup.Progress <= 0;
                        break;
                }
            }

            _setup.PropertyChanged += OnSetupChanged;

            try
            {
                await _setup.InstallAsync();
            }
            finally
            {
                _setup.PropertyChanged -= OnSetupChanged;
            }
        }
        finally
        {
            _progress.Visibility = Visibility.Collapsed;
            Rebuild();
            deferral.Complete();
        }
    }
}
