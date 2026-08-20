using System.ComponentModel;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Windows.Storage.Pickers;
using WinRT.Interop;

namespace LocalScribe.App;

/// <summary>
/// The single window. Bindings are wired by hand rather than through XAML compiled bindings so
/// that the view model stays free of WinUI types and can be exercised without a UI host.
/// </summary>
public sealed partial class MainWindow : Window
{
    private readonly MainViewModel _viewModel = new();
    private readonly DispatcherQueue _dispatcher;

    public MainWindow()
    {
        InitializeComponent();
        _dispatcher = DispatcherQueue.GetForCurrentThread();

        _viewModel.PropertyChanged += OnViewModelPropertyChanged;
        Closed += (_, _) => _viewModel.Dispose();

        _ = _viewModel.InitialiseAsync();
    }

    /// <summary>
    /// Marshals view-model changes onto the UI thread. Live transcription raises these from an
    /// audio capture thread, so this hop is required rather than defensive.
    /// </summary>
    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        _dispatcher.TryEnqueue(() =>
        {
            switch (e.PropertyName)
            {
                case nameof(MainViewModel.Status):
                    StatusText.Text = _viewModel.Status;
                    break;
                case nameof(MainViewModel.HardwareSummary):
                    HardwareText.Text = _viewModel.HardwareSummary;
                    break;
                case nameof(MainViewModel.Transcript):
                    TranscriptText.Text = _viewModel.Transcript;
                    break;
                case nameof(MainViewModel.ProvisionalText):
                    ProvisionalTextBlock.Text = _viewModel.ProvisionalText;
                    break;
                case nameof(MainViewModel.Progress):
                    ProgressBarControl.Value = _viewModel.Progress;
                    break;
                case nameof(MainViewModel.IsBusy):
                    OpenFileButton.IsEnabled = !_viewModel.IsBusy;
                    CancelButton.IsEnabled = _viewModel.IsBusy;
                    break;
                case nameof(MainViewModel.IsRecording):
                    RecordLabel.Text = _viewModel.IsRecording ? "Stop listening" : "Start listening";
                    // Segoe Fluent Icons: a stop square while recording, a microphone otherwise.
                    RecordIcon.Glyph = _viewModel.IsRecording ? "\uE71A" : "\uE720";
                    OpenFileButton.IsEnabled = !_viewModel.IsRecording;
                    break;
            }
        });
    }

    /// <summary>
    /// Reflects the three states the record button actually has. Preparing is not recording:
    /// the microphone is not running yet, and anything said now is lost rather than delayed, so
    /// it must not look like it is listening.
    /// </summary>
    private void UpdateRecordButton()
    {
        if (_viewModel.IsPreparing)
        {
            RecordLabel.Text = "Getting ready…";
            RecordIcon.Glyph = "";                 // stopwatch
            RecordButton.IsEnabled = false;
            OpenFileButton.IsEnabled = false;
            ProgressBarControl.IsIndeterminate = true;

            ShowCue(
                WaitBackground,
                "Wait — not recording yet",
                "Loading the model. The microphone is off, so anything said now is lost rather "
                + "than queued.",
                pulse: false);
            return;
        }

        ProgressBarControl.IsIndeterminate = false;

        if (_viewModel.IsRecording)
        {
            ShowCue(
                ListenBackground,
                "Speak now — listening",
                "The microphone is live and everything is being transcribed on this machine.",
                pulse: true);
        }
        else
        {
            PulseStoryboard.Stop();
            CueBanner.Visibility = Visibility.Collapsed;
        }

        RecordButton.IsEnabled = true;
        RecordLabel.Text = _viewModel.IsRecording ? "Stop listening" : "Start listening";

        // Segoe Fluent Icons: a stop square while recording, a microphone otherwise.
        RecordIcon.Glyph = _viewModel.IsRecording ? "" : "";
        OpenFileButton.IsEnabled = !_viewModel.IsRecording;
    }

    /// <summary>
    /// Saturated fills with white text rather than theme brushes. This is the one signal in the
    /// window that has to survive being glanced at, and a subtle tint that adapts to light and
    /// dark adapts itself into invisibility.
    /// </summary>
    private static readonly SolidColorBrush WaitBackground =
        new(Windows.UI.Color.FromArgb(255, 0xB4, 0x53, 0x09));   // amber

    private static readonly SolidColorBrush ListenBackground =
        new(Windows.UI.Color.FromArgb(255, 0x15, 0x80, 0x3D));   // green

    private void ShowCue(SolidColorBrush background, string title, string detail, bool pulse)
    {
        CueBanner.Background = background;
        CueTitle.Text = title;
        CueDetail.Text = detail;
        CueBanner.Visibility = Visibility.Visible;

        // Motion registers before either colour or words do, so it is reserved for the state
        // where acting on it matters: the microphone being live.
        if (pulse)
        {
            PulseStoryboard.Begin();
        }
        else
        {
            PulseStoryboard.Stop();
            CueDot.Opacity = 1.0;
        }
    }

    private async void OnOpenFile(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker();

        // A picker created in an unpackaged app has no window of its own and must be told which
        // window owns it, or the call fails at runtime rather than at compile time.
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(this));

        foreach (var extension in new[] { ".wav", ".mp3", ".m4a", ".flac", ".wma", ".aac" })
        {
            picker.FileTypeFilter.Add(extension);
        }

        var file = await picker.PickSingleFileAsync();
        if (file is not null)
        {
            await _viewModel.TranscribeFileAsync(file.Path);
        }
    }

    private async void OnToggleRecording(object sender, RoutedEventArgs e)
    {
        if (_viewModel.IsRecording)
        {
            await _viewModel.StopRecordingAsync();
        }
        else
        {
            await _viewModel.StartRecordingAsync();
        }
    }

    private void OnCancel(object sender, RoutedEventArgs e) => _viewModel.Cancel();

    /// <summary>
    /// Collects the terms the cleanup model should spell correctly: names, products, jargon.
    /// This is the cheapest accuracy win available, and no larger Whisper model substitutes for it.
    /// </summary>
    private async void OnEditGlossary(object sender, RoutedEventArgs e)
    {
        var textBox = new TextBox
        {
            AcceptsReturn = true,
            Height = 240,
            Text = string.Join(Environment.NewLine, _viewModel.Glossary),
            PlaceholderText = "One term per line, e.g. names, products, acronyms",
        };

        var dialog = new ContentDialog
        {
            XamlRoot = Content.XamlRoot,
            Title = "Glossary",
            Content = textBox,
            PrimaryButtonText = "Save",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Primary,
        };

        if (await dialog.ShowAsync() == ContentDialogResult.Primary)
        {
            _viewModel.Glossary.Clear();
            _viewModel.Glossary.AddRange(
                textBox.Text.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries
                    | StringSplitOptions.TrimEntries));
        }
    }
}
