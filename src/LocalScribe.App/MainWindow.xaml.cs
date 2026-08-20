using System.ComponentModel;
using LocalScribe.Core.Transcription;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
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
    private IReadOnlyList<ParagraphView> _paragraphs = [];

    /// <summary>
    /// True while the scrubber is being moved by playback rather than by the user. The same
    /// slider carries both, and treating a playback update as a seek would restart the audio
    /// ten times a second.
    /// </summary>
    private bool _suppressScrub;

    public MainWindow()
    {
        InitializeComponent();
        _dispatcher = DispatcherQueue.GetForCurrentThread();

        _viewModel.PropertyChanged += OnViewModelPropertyChanged;

        _viewModel.Player.PositionChanged += OnPlaybackPosition;
        _viewModel.Player.Stopped += OnPlaybackStopped;
        _viewModel.Player.Failed += OnPlaybackFailed;

        Closed += (_, _) =>
        {
            _viewModel.Player.PositionChanged -= OnPlaybackPosition;
            _viewModel.Player.Stopped -= OnPlaybackStopped;
            _viewModel.Player.Failed -= OnPlaybackFailed;
            _viewModel.Dispose();
        };

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
                case nameof(MainViewModel.Paragraphs):
                    ShowParagraphs();
                    break;
                case nameof(MainViewModel.Summary):
                    SummaryText.Text = _viewModel.Summary;
                    SummaryCard.Visibility = _viewModel.Summary.Length > 0
                        ? Visibility.Visible
                        : Visibility.Collapsed;
                    break;
                case nameof(MainViewModel.HasTranscript):
                    CopyButton.IsEnabled = _viewModel.HasTranscript;
                    SaveButton.IsEnabled = _viewModel.HasTranscript;
                    DiscardButton.IsEnabled = _viewModel.HasTranscript;
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
                case nameof(MainViewModel.IsPreparing):
                    UpdateRecordButton();
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
            StartButton.Visibility = Visibility.Visible;
            StopButton.Visibility = Visibility.Collapsed;

            StartLabel.Text = "Getting ready…";
            StartIcon.Glyph = "";                  // stopwatch
            StartButton.IsEnabled = false;
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

        // Red is the one colour that means recording, so it belongs on the control the user is
        // looking at when they wonder whether it is. The banner says the same thing in words;
        // this says it without being read.
        StartButton.Visibility = _viewModel.IsRecording ? Visibility.Collapsed : Visibility.Visible;
        StopButton.Visibility = _viewModel.IsRecording ? Visibility.Visible : Visibility.Collapsed;

        StartButton.IsEnabled = true;
        StartLabel.Text = "Start listening";
        StartIcon.Glyph = "";                      // microphone
        OpenFileButton.IsEnabled = !_viewModel.IsRecording;

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

    /// <summary>
    /// Rebuilds the list. Scrolls to the end only while text is still arriving, so a transcript
    /// being read is not dragged out from under the reader.
    /// </summary>
    private void ShowParagraphs()
    {
        var following = _viewModel.IsRecording || _viewModel.IsBusy;

        _paragraphs = _viewModel.Paragraphs.Select(p => new ParagraphView(p)).ToList();
        ApplySearch();

        if (following && _paragraphs.Count > 0)
        {
            TranscriptList.ScrollIntoView(_paragraphs[^1]);
        }

        ShowTransport();
    }

    /// <summary>
    /// Shows the transport once there is something to play, and sizes the scrubber to it.
    /// </summary>
    private void ShowTransport()
    {
        var player = _viewModel.Player;

        if (!player.HasAudio || player.DurationSeconds <= 0)
        {
            TransportBar.Visibility = Visibility.Collapsed;
            return;
        }

        TransportBar.Visibility = Visibility.Visible;
        Scrubber.Maximum = player.DurationSeconds;
        UpdateClock(Scrubber.Value);
    }

    private void OnParagraphClick(object sender, ItemClickEventArgs e)
    {
        if (e.ClickedItem is not ParagraphView paragraph)
        {
            return;
        }

        TranscriptList.SelectedItem = paragraph;

        // Said rather than done silently. Clicking a line and getting nothing is the failure
        // this whole feature is supposed to avoid.
        if (!_viewModel.Player.HasAudio)
        {
            StatusText.Text = "No audio is loaded for this transcript, so there is nothing to play.";
            return;
        }

        Seek(paragraph.StartSeconds, play: true);
    }

    /// <summary>
    /// Moves both views at once. The transcript and the recording are the same thing seen two
    /// ways, so a position set from either has to land in both.
    /// </summary>
    private void Seek(double seconds, bool play)
    {
        _suppressScrub = true;
        Scrubber.Value = Math.Clamp(seconds, 0, Scrubber.Maximum);
        _suppressScrub = false;

        UpdateClock(seconds);
        HighlightAt(seconds);

        if (play)
        {
            _viewModel.Player.PlayFrom(seconds);
            StopPlaybackButton.Visibility = Visibility.Visible;
            PlayIcon.Glyph = "";        // pause
        }
    }

    /// <summary>Selects the paragraph covering a position, without disturbing the reader otherwise.</summary>
    private void HighlightAt(double seconds)
    {
        var playing = _paragraphs.FirstOrDefault(p => p.Contains(seconds))
            ?? _paragraphs.LastOrDefault(p => p.StartSeconds <= seconds);

        if (playing is not null && !ReferenceEquals(TranscriptList.SelectedItem, playing))
        {
            TranscriptList.SelectedItem = playing;
            TranscriptList.ScrollIntoView(playing);
        }
    }

    private void UpdateClock(double seconds) =>
        TransportClock.Text =
            $"{TranscriptFormatter.Clock(seconds)} / {TranscriptFormatter.Clock(_viewModel.Player.DurationSeconds)}";

    /// <summary>
    /// Scrubbing. Only acts on changes the user made: the same slider is moved by playback, and
    /// treating that as a seek would restart the audio ten times a second.
    /// </summary>
    private void OnScrub(object sender, RangeBaseValueChangedEventArgs e)
    {
        if (_suppressScrub)
        {
            return;
        }

        UpdateClock(e.NewValue);
        HighlightAt(e.NewValue);

        if (_viewModel.Player.IsPlaying)
        {
            _viewModel.Player.PlayFrom(e.NewValue);
        }
    }

    private void OnPlayPause(object sender, RoutedEventArgs e)
    {
        if (!_viewModel.Player.HasAudio)
        {
            StatusText.Text = "No audio is loaded for this transcript, so there is nothing to play.";
            return;
        }

        if (_viewModel.Player.IsPlaying)
        {
            _viewModel.Player.Stop();
            return;
        }

        Seek(Scrubber.Value, play: true);
    }

    private void OnPlaybackFailed(string message) =>
        _dispatcher.TryEnqueue(() =>
        {
            StatusText.Text = message;
            PlayIcon.Glyph = "";
            StopPlaybackButton.Visibility = Visibility.Collapsed;
        });

    /// <summary>
    /// Narrows the list to paragraphs containing the search text. Filtering rather than
    /// highlighting in place, because the thing people do with a transcript is find the moment
    /// something was said and then listen to it, and every row keeps its timestamp.
    /// </summary>
    private void OnSearchChanged(object sender, TextChangedEventArgs e) => ApplySearch();

    private void ApplySearch()
    {
        var query = SearchBox.Text.Trim();

        if (query.Length == 0)
        {
            TranscriptList.ItemsSource = _paragraphs;
            SearchCount.Text = string.Empty;
            return;
        }

        var matches = _paragraphs
            .Where(p => p.Text.Contains(query, StringComparison.OrdinalIgnoreCase)
                || p.Speaker.Contains(query, StringComparison.OrdinalIgnoreCase))
            .ToList();

        TranscriptList.ItemsSource = matches;
        SearchCount.Text = matches.Count switch
        {
            0 => "no matches",
            1 => "1 match",
            _ => $"{matches.Count} matches",
        };
    }

    /// <summary>
    /// Follows playback by moving the selection, so the highlight is the same mechanism the user
    /// clicked with rather than a second one that has to be kept in step with it.
    /// </summary>
    private void OnPlaybackPosition(double seconds) =>
        _dispatcher.TryEnqueue(() =>
        {
            _suppressScrub = true;
            Scrubber.Value = Math.Clamp(seconds, 0, Scrubber.Maximum);
            _suppressScrub = false;

            UpdateClock(seconds);
            HighlightAt(seconds);
        });

    private void OnPlaybackStopped() =>
        _dispatcher.TryEnqueue(() =>
        {
            StopPlaybackButton.Visibility = Visibility.Collapsed;
            PlayIcon.Glyph = "";
        });

    private void OnStopPlayback(object sender, RoutedEventArgs e) => _viewModel.Player.Stop();

    private void OnCopy(object sender, RoutedEventArgs e)
    {
        var package = new Windows.ApplicationModel.DataTransfer.DataPackage();
        package.SetText(_viewModel.Export(TranscriptFormat.PlainText));

        Windows.ApplicationModel.DataTransfer.Clipboard.SetContent(package);

        StatusText.Text = "Transcript copied.";
    }

    private async void OnSave(object sender, RoutedEventArgs e)
    {
        var picker = new FileSavePicker { SuggestedFileName = _viewModel.SourceName };
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(this));

        picker.FileTypeChoices.Add("Text", [".txt"]);
        picker.FileTypeChoices.Add("Markdown", [".md"]);
        picker.FileTypeChoices.Add("Subtitles", [".srt"]);

        var file = await picker.PickSaveFileAsync();
        if (file is null)
        {
            return;
        }

        var format = Path.GetExtension(file.Name).ToLowerInvariant() switch
        {
            ".md" => TranscriptFormat.Markdown,
            ".srt" => TranscriptFormat.SubRip,
            _ => TranscriptFormat.PlainText,
        };

        await Windows.Storage.FileIO.WriteTextAsync(file, _viewModel.Export(format));

        StatusText.Text = $"Saved to {file.Path}";
    }

    private async void OnDiscard(object sender, RoutedEventArgs e)
    {
        // Confirmed, because there is nowhere to get it back from: the audio lives only in
        // memory and goes with it.
        var dialog = new ContentDialog
        {
            XamlRoot = Content.XamlRoot,
            Title = "Discard this transcript?",
            Content = "The transcript and the recording behind it will be gone. "
                + "Nothing has been written to disk unless you saved it.",
            PrimaryButtonText = "Discard",
            CloseButtonText = "Keep",
            DefaultButton = ContentDialogButton.Close,
        };

        if (await dialog.ShowAsync() == ContentDialogResult.Primary)
        {
            _viewModel.Discard();
        }
    }

    private async void OnOpenFile(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker();

        // A picker created in an unpackaged app has no window of its own and must be told which
        // window owns it, or the call fails at runtime rather than at compile time.
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(this));

        foreach (var extension in AudioFileLoader.SupportedExtensions)
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
