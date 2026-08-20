using System.ComponentModel;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
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

        // Probing waits for the content to load rather than starting in the constructor,
        // because showing setup needs a XamlRoot and there is not one yet here.
        if (Content is FrameworkElement root)
        {
            root.Loaded += OnRootLoaded;
        }
    }

    private async void OnRootLoaded(object sender, RoutedEventArgs e)
    {
        ((FrameworkElement)sender).Loaded -= OnRootLoaded;

        try
        {
            await _viewModel.InitialiseAsync();

            // A machine that cannot transcribe yet gets setup up front. Anything else means a
            // new user's first action fails for a reason the app knew about before they clicked.
            if (!_viewModel.Setup.CanTranscribe)
            {
                await SetupDialog.ShowAsync(_viewModel.Setup, Content.XamlRoot);
            }
        }
        catch (Exception exception)
        {
            // Nothing above is allowed to close the window. This is an async void handler, so
            // an exception escaping here terminates the process rather than showing an error.
            StatusText.Text = $"Startup check failed: {exception.Message}";
        }
    }

    /// <summary>
    /// Sends the user to setup instead of into a failure they cannot act on. Setup is the only
    /// place that can fix a missing model, so an action that needs one goes there first rather
    /// than surfacing a file path in the status bar.
    /// </summary>
    private async Task<bool> EnsureReadyAsync()
    {
        if (_viewModel.Setup.CanTranscribe)
        {
            return true;
        }

        await SetupDialog.ShowAsync(_viewModel.Setup, Content.XamlRoot);
        return _viewModel.Setup.CanTranscribe;
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
                    // Listening during a file run would open a second transcriber and put two
                    // workloads on one NPU, which is the contention the pipeline avoids by
                    // transcribing windows one at a time.
                    RecordButton.IsEnabled = !_viewModel.IsBusy;
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

    private async void OnOpenFile(object sender, RoutedEventArgs e)
    {
        if (!await EnsureReadyAsync())
        {
            return;
        }

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
        else if (await EnsureReadyAsync())
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
