using System.ComponentModel;
using LocalScribe.Core.Archive;
using LocalScribe.Core.Transcription;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Documents;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Windows.Graphics;
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

    /// <summary>Peaks the waveform is drawn from. Finer than the eye at any sane width.</summary>
    private const int PeakCount = 480;

    private string? _pendingFile;
    private float[] _peaks = [];
    private double _peaksFor = -1;
    private double _position;
    private bool _scrubbing;

    public MainWindow()
    {
        InitializeComponent();
        _dispatcher = DispatcherQueue.GetForCurrentThread();

        // The exe embeds the icon for Explorer and the taskbar, but an unpackaged window does
        // not read it for its own frame and shows the stock icon until told otherwise.
        AppWindow.SetIcon(Path.Combine(AppContext.BaseDirectory, "Assets", "localscribe.ico"));

        _viewModel.PropertyChanged += OnViewModelPropertyChanged;

        // Words are drawn with a colour taken from the theme when the row is built, so a theme
        // changed afterwards leaves every paragraph in the old ink.
        TranscriptList.ActualThemeChanged += OnThemeChanged;

        _viewModel.Player.PositionChanged += OnPlaybackPosition;
        _viewModel.Player.Stopped += OnPlaybackStopped;
        _viewModel.Player.Failed += OnPlaybackFailed;

        AppWindow.Closing += OnWindowClosing;

        Closed += (_, _) =>
        {
            _viewModel.Player.PositionChanged -= OnPlaybackPosition;
            _viewModel.Player.Stopped -= OnPlaybackStopped;
            _viewModel.Player.Failed -= OnPlaybackFailed;
            _viewModel.Dispose();
        };

        SizeToContent();

        // Probe, then warm the model while the user is still reading the hardware line.
        _ = InitialiseAsync();
    }

    /// <summary>Initial window size, in logical pixels, before any display scaling.</summary>
    private const int InitialWidth = 1040;

    private const int InitialHeight = 720;

    /// <summary>
    /// Opens at a size the content actually needs, centred, rather than at the framework
    /// default.
    /// <para>
    /// Only the opening size — the window stays freely resizable, and a transcript long enough
    /// to want more room is exactly the case for dragging it larger. The width is set by the
    /// toolbar, which is the widest thing here and looks broken when its two groups collide;
    /// the height by wanting several paragraphs visible above the transport bar.
    /// </para>
    /// <para>
    /// Sized in logical pixels and scaled by the window's DPI, so that a 200% display gets a
    /// window of the same apparent size rather than one of half of it.
    /// </para>
    /// </summary>
    private void SizeToContent()
    {
        var handle = WindowNative.GetWindowHandle(this);
        var scale = GetDpiForWindow(handle) / 96.0;

        var width = (int)(InitialWidth * scale);
        var height = (int)(InitialHeight * scale);

        var area = DisplayArea.GetFromWindowId(AppWindow.Id, DisplayAreaFallback.Primary);

        if (area is not null)
        {
            // Never larger than the screen it opens on. The chosen size is comfortable on a
            // laptop and too tall for a small one, and a window taller than the desktop puts its
            // own transport bar out of reach.
            width = Math.Min(width, area.WorkArea.Width);
            height = Math.Min(height, area.WorkArea.Height);
        }

        AppWindow.Resize(new SizeInt32(width, height));

        if (area is not null)
        {
            AppWindow.Move(new PointInt32(
                area.WorkArea.X + ((area.WorkArea.Width - width) / 2),
                area.WorkArea.Y + ((area.WorkArea.Height - height) / 2)));
        }
    }

    // DllImport rather than the newer LibraryImport, which generates unsafe marshalling code and
    // would mean turning unsafe on for the whole app to ask one number of one function.
    [System.Runtime.InteropServices.DllImport("user32.dll")]
    private static extern uint GetDpiForWindow(nint hwnd);

    private async Task InitialiseAsync()
    {
        await _viewModel.InitialiseAsync();
        await _viewModel.PreloadAsync();

        // Anything handed to the window before the model was ready waited for this moment.
        if (_pendingFile is { } path)
        {
            _pendingFile = null;
            await _viewModel.TranscribeFileAsync(path);
        }
    }

    /// <summary>
    /// Transcribes a file as soon as the app is ready for it. Called for a file named on the
    /// command line or dropped on the window, both of which can arrive before the hardware has
    /// even been probed.
    /// </summary>
    public void OpenWhenReady(string path)
    {
        // Saved transcripts need no model, so they never wait for one.
        if (TranscriptArchive.IsArchive(path))
        {
            OpenArchive(path);
            return;
        }

        if (_viewModel.IsModelReady)
        {
            _ = _viewModel.TranscribeFileAsync(path);
            return;
        }

        _pendingFile = path;
    }

    /// <summary>
    /// Dropping a recording on the window is the shortest path from a file to a transcript, and
    /// the one that does not involve a picker.
    /// </summary>
    private void OnDragOver(object sender, DragEventArgs e)
    {
        e.AcceptedOperation = Windows.ApplicationModel.DataTransfer.DataPackageOperation.Copy;
        e.DragUIOverride.Caption = "Transcribe this";
        e.DragUIOverride.IsGlyphVisible = true;
    }

    private async void OnDrop(object sender, DragEventArgs e)
    {
        if (!e.DataView.Contains(Windows.ApplicationModel.DataTransfer.StandardDataFormats.StorageItems))
        {
            return;
        }

        var items = await e.DataView.GetStorageItemsAsync();
        var file = items.OfType<Windows.Storage.StorageFile>().FirstOrDefault();

        if (file is null)
        {
            return;
        }

        if (TranscriptArchive.IsArchive(file.Name))
        {
            OpenArchive(file.Path);
            return;
        }

        if (!AudioFileLoader.SupportedExtensions.Contains(Path.GetExtension(file.Name), StringComparer.OrdinalIgnoreCase))
        {
            StatusText.Text = $"{file.Name} is not an audio or video file LocalScribe can read.";
            return;
        }

        await _viewModel.TranscribeFileAsync(file.Path);
    }

    /// <summary>
    /// Marshals view-model changes onto the UI thread. Live transcription raises these from an
    /// audio capture thread, so this hop is required rather than defensive.
    /// </summary>
    /// <summary>
    /// Shows what cleanup left undone, and offers to run it again.
    /// <para>
    /// The button is disabled rather than hidden while a run is going, so it does not appear and
    /// disappear under the pointer of somebody reaching for it.
    /// </para>
    /// </summary>
    private void ShowCleanupNotice()
    {
        if (_viewModel.CleanupNotice is { Length: > 0 } notice)
        {
            CleanupNoticeText.Text = notice;
            CleanupNotice.Visibility = Visibility.Visible;
            RetryCleanupButton.IsEnabled = _viewModel.CanRetryCleanup;
            return;
        }

        CleanupNotice.Visibility = Visibility.Collapsed;
    }

    private async void OnRetryCleanup(object sender, RoutedEventArgs e) =>
        await _viewModel.RetryCleanupAsync();

    private async void OnToggleTranslate(object sender, RoutedEventArgs e) =>
        await _viewModel.TranslateAgainAsync(TranslateButton.IsChecked == true);

    /// <summary>Shows the offer only when it would change something.</summary>
    private void ShowTranslateOffer() =>
        TranslateButton.Visibility = _viewModel.CanOfferTranslation
            ? Visibility.Visible
            : Visibility.Collapsed;

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        _dispatcher.TryEnqueue(() =>
        {
            switch (e.PropertyName)
            {
                case nameof(MainViewModel.CanOfferTranslation):
                    ShowTranslateOffer();
                    break;
                case nameof(MainViewModel.CleanupNotice):
                case nameof(MainViewModel.CanRetryCleanup):
                    ShowCleanupNotice();
                    break;
                case nameof(MainViewModel.Status):
                    StatusText.Text = _viewModel.Status;
                    break;
                case nameof(MainViewModel.HardwareSummary):
                    HardwareText.Text = _viewModel.HardwareSummary;
                    break;
                case nameof(MainViewModel.Paragraphs):
                    ShowParagraphs();
                    break;
                case nameof(MainViewModel.HasTranscript):
                    CopyButton.IsEnabled = _viewModel.HasTranscript;
                    SaveButton.IsEnabled = _viewModel.HasTranscript;
                    DiscardButton.IsEnabled = _viewModel.HasTranscript;
                    SpeakersButton.IsEnabled = _viewModel.CanFindSpeakers;
                    break;
                case nameof(MainViewModel.SpeakerCount):
                    UpdateSpeakerCount();
                    break;
                case nameof(MainViewModel.ProvisionalText):
                    ProvisionalTextBlock.Text = _viewModel.ProvisionalText;
                    break;
                case nameof(MainViewModel.Progress):
                    ProgressBarControl.Value = _viewModel.Progress;

                    // Re-judged on progress too, because the state the glow watches has no
                    // notification of its own: the player receives its audio a moment after
                    // IsBusy turns on, and the next progress report is the first sign of it.
                    UpdateGlow();
                    break;
                case nameof(MainViewModel.IsBusy):
                    OpenFileButton.IsEnabled = !_viewModel.IsBusy;
                    OpenTranscriptButton.IsEnabled = !_viewModel.IsBusy;
                    CancelButton.Visibility = _viewModel.IsBusy ? Visibility.Visible : Visibility.Collapsed;
                    UpdateGlow();
                    break;
                case nameof(MainViewModel.IsRecording):
                case nameof(MainViewModel.IsPreparing):
                case nameof(MainViewModel.IsWarmingUp):
                    UpdateRecordButton();
                    UpdateGlow();
                    break;
            }
        });
    }

    /// <summary>
    /// Puts the number of speakers found on the button that edits them.
    /// <para>
    /// Diarization is the step most likely to be quietly wrong, and wrong in a way that is
    /// obvious to the person who was in the room and invisible in a status line they have
    /// already scrolled past. Two people talking and a badge reading 5 is a whole diagnosis at a
    /// glance, and the fix is behind the button it is sitting on.
    /// </para>
    /// </summary>
    private void UpdateSpeakerCount()
    {
        var count = _viewModel.SpeakerCount;

        SpeakerCountText.Text = count.ToString();
        SpeakerCountBadge.Visibility = count > 0 ? Visibility.Visible : Visibility.Collapsed;

        ToolTipService.SetToolTip(
            SpeakersButton,
            count == 0
                ? "Work out who spoke"
                : $"{count} {(count == 1 ? "speaker" : "speakers")} found — click to rename or recount");
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
            OpenTranscriptButton.IsEnabled = false;
            ProgressBarControl.IsIndeterminate = true;

            ShowCue(
                WaitBackground,
                "Wait — not recording yet",
                "Loading the model. The microphone is off, so anything said now is lost rather "
                + "than queued.",
                pulse: false);
            return;
        }

        // Warming up needs no banner. The moving bar and the status line already say it, and the
        // banner style is reserved for the two states that are instructions to act on — hold off,
        // speak now.
        if (_viewModel.IsWarmingUp)
        {
            StartButton.Visibility = Visibility.Visible;
            StopButton.Visibility = Visibility.Collapsed;

            StartLabel.Text = "Start listening";
            StartIcon.Glyph = "";                // microphone

            // Greyed out rather than live but useless. Recording cannot start before the model
            // is loaded, so an enabled button would take the click and appear to do nothing;
            // disabled, with the reason on hover, says so where the user is already looking.
            StartButton.IsEnabled = false;
            ToolTipService.SetToolTip(
                StartButton,
                "Warming up — the model takes a few seconds the first time");

            // The file picker stays open. Choosing a recording takes long enough that the load
            // finishes underneath it, and a run queued meanwhile simply waits.
            OpenFileButton.IsEnabled = true;
            OpenTranscriptButton.IsEnabled = true;

            ProgressBarControl.IsIndeterminate = true;
            PulseStoryboard.Stop();
            CueBanner.Visibility = Visibility.Collapsed;
            return;
        }

        ProgressBarControl.IsIndeterminate = false;

        // Red is the one colour that means recording, so it belongs on the control the user is
        // looking at when they wonder whether it is. The banner says the same thing in words;
        // this says it without being read.
        StartButton.Visibility = _viewModel.IsRecording ? Visibility.Collapsed : Visibility.Visible;
        StopButton.Visibility = _viewModel.IsRecording ? Visibility.Visible : Visibility.Collapsed;

        StartButton.IsEnabled = true;
        ToolTipService.SetToolTip(StartButton, null);
        StartLabel.Text = "Start listening";
        StartIcon.Glyph = "";                      // microphone
        OpenFileButton.IsEnabled = !_viewModel.IsRecording;
        OpenTranscriptButton.IsEnabled = !_viewModel.IsRecording;

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

    /// <summary>
    /// Slate, and deliberately quieter than the other two. Those carry instructions the user has
    /// to act on — hold off, speak now — and shout accordingly. This one is the app explaining
    /// itself, which is worth saying and not worth alarming anyone about.
    /// </summary>
    private static readonly SolidColorBrush WarmingBackground =
        new(Windows.UI.Color.FromArgb(255, 0x3F, 0x4A, 0x5A));

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
    /// Rebuilds the list. While text is still arriving, the tail is followed only for a reader
    /// who was already at the tail — anyone who has scrolled up is reading the head the
    /// progressive preview just made worth reading, and gets put back exactly where they were.
    /// </summary>
    private void ShowParagraphs()
    {
        var following = _viewModel.IsRecording || _viewModel.IsBusy;
        var scroller = TranscriptScroller();

        var atTail = scroller is null
            || scroller.ScrollableHeight < 1
            || scroller.VerticalOffset >= scroller.ScrollableHeight - 80;

        var offset = scroller?.VerticalOffset ?? 0;

        // Views are reused wherever the paragraph did not change, so a streaming update only
        // rebuilds the rows it actually altered — the growing tail, and a head row the moment
        // its timing arrives. Replacing the whole list rebuilt every visible container and
        // read as the transcript blinking once a window.
        var fresh = _viewModel.Paragraphs;
        var views = new List<ParagraphView>(fresh.Count);

        for (var i = 0; i < fresh.Count; i++)
        {
            views.Add(i < _paragraphs.Count && _paragraphs[i].Shows(fresh[i])
                ? _paragraphs[i]
                : new ParagraphView(fresh[i]));
        }

        var alive = new HashSet<ParagraphView>(views);

        foreach (var stale in _spokenWords.Keys.Where(view => !alive.Contains(view)).ToList())
        {
            _spokenWords.Remove(stale);
        }

        _realised.RemoveWhere(view => !alive.Contains(view));

        if (_lit is not null && !alive.Contains(_lit))
        {
            _lit = null;
            _markedWord = -1;
        }

        _paragraphs = views;
        ApplySearch();

        if (following && _paragraphs.Count > 0)
        {
            if (atTail)
            {
                TranscriptList.ScrollIntoView(_paragraphs[^1]);
            }
            else if (scroller is not null)
            {
                // After the rebuild has laid out, or the restore lands on the old layout.
                _dispatcher.TryEnqueue(
                    DispatcherQueuePriority.Low,
                    () => scroller.ChangeView(null, offset, null, disableAnimation: true));
            }
        }

        ShowTransport();
    }

    private ScrollViewer? _transcriptScroller;

    /// <summary>The list's scroller, found once — rebuilds replace items, not the chrome.</summary>
    private ScrollViewer? TranscriptScroller()
    {
        if (_transcriptScroller is not null)
        {
            return _transcriptScroller;
        }

        var queue = new Queue<DependencyObject>();
        queue.Enqueue(TranscriptList);

        while (queue.Count > 0)
        {
            var next = queue.Dequeue();

            if (next is ScrollViewer found)
            {
                return _transcriptScroller = found;
            }

            for (var i = 0; i < Microsoft.UI.Xaml.Media.VisualTreeHelper.GetChildrenCount(next); i++)
            {
                queue.Enqueue(Microsoft.UI.Xaml.Media.VisualTreeHelper.GetChild(next, i));
            }
        }

        return null;
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
            _peaks = [];
            _peaksFor = -1;
            return;
        }

        TransportBar.Visibility = Visibility.Visible;

        // Recomputed only when the audio changes. It is a pass over every sample in the
        // recording, which is not something to do on a redraw.
        if (Math.Abs(_peaksFor - player.DurationSeconds) > 1e-6)
        {
            _peaks = _viewModel.WaveformPeaks(PeakCount);
            _peaksFor = player.DurationSeconds;
        }

        DrawWaveform();
    }

    /// <summary>
    /// Builds the waveform as one filled outline rather than hundreds of bars: a polygon across
    /// the peaks and back along their mirror. One shape for the layout engine to handle, instead
    /// of several hundred on every resize.
    /// </summary>
    private void DrawWaveform()
    {
        var width = Waveform.ActualWidth;
        var height = Waveform.ActualHeight;

        if (_peaks.Length < 2 || width <= 1 || height <= 1)
        {
            return;
        }

        var middle = height / 2;
        var points = new PointCollection();

        for (var i = 0; i < _peaks.Length; i++)
        {
            points.Add(new Windows.Foundation.Point(X(i, width), middle - Amplitude(i, middle)));
        }

        for (var i = _peaks.Length - 1; i >= 0; i--)
        {
            points.Add(new Windows.Foundation.Point(X(i, width), middle + Amplitude(i, middle)));
        }

        WaveformShape.Points = points;

        var played = new PointCollection();
        foreach (var point in points)
        {
            played.Add(point);
        }

        WaveformPlayed.Points = played;

        MovePlayhead(_position);
    }

    private double X(int index, double width) => index * width / (_peaks.Length - 1.0);

    /// <summary>
    /// A floor under every peak, so silence draws as a line rather than a gap. A waveform that
    /// disappears reads as missing audio rather than as quiet.
    /// </summary>
    private double Amplitude(int index, double middle) =>
        Math.Max(_peaks[index] * middle * 0.92, 1.0);

    /// <summary>Shows how far playback has got, by clipping the accent copy to it.</summary>
    private void MovePlayhead(double seconds)
    {
        var duration = _viewModel.Player.DurationSeconds;
        var width = Waveform.ActualWidth;

        if (duration <= 0 || width <= 1)
        {
            return;
        }

        var x = Math.Clamp(seconds / duration, 0, 1) * width;

        WaveformPlayed.Clip = new RectangleGeometry
        {
            Rect = new Windows.Foundation.Rect(0, 0, x, Math.Max(1, Waveform.ActualHeight)),
        };

        Playhead.Margin = new Thickness(Math.Max(0, x - 1), 0, 0, 0);

        TransportClock.Text =
            $"{TranscriptFormatter.Clock(seconds)} / {TranscriptFormatter.Clock(duration)}";
    }

    private void OnWaveformSizeChanged(object sender, SizeChangedEventArgs e)
    {
        DrawWaveform();
        DrawSearchMarks();
    }

    /// <summary>
    /// Marks on the waveform where the search term is said.
    /// <para>
    /// This is what turns the waveform into a map of the search rather than only of the sound:
    /// four marks spread across an hour tells you at a glance whether a topic came up once or
    /// ran through the whole meeting, which no amount of scrolling the text does.
    /// </para>
    /// </summary>
    private void DrawSearchMarks()
    {
        WaveformMarks.Children.Clear();

        var query = SearchBox.Text.Trim();
        var duration = _viewModel.Player.DurationSeconds;
        var width = Waveform.ActualWidth;
        var height = Waveform.ActualHeight;

        if (query.Length == 0 || duration <= 0 || width <= 1 || height <= 1)
        {
            return;
        }

        foreach (var (start, end) in _paragraphs.SelectMany(p => p.MatchSpans(query)))
        {
            var left = Math.Clamp(start / duration, 0, 1) * width;
            var right = Math.Clamp(end / duration, 0, 1) * width;

            // A brief mention is still a mention: a span narrower than this would be invisible,
            // so it is widened rather than left off the map.
            var markWidth = Math.Max(right - left, 3);

            var mark = new Microsoft.UI.Xaml.Shapes.Rectangle
            {
                Width = markWidth,
                Height = height,
                Fill = MarkBrush,
                RadiusX = 1,
                RadiusY = 1,
            };

            Canvas.SetLeft(mark, Math.Min(left, Math.Max(0, width - markWidth)));
            Canvas.SetTop(mark, 0);

            WaveformMarks.Children.Add(mark);
        }
    }

    /// <summary>
    /// The same yellow as the marks in the text, translucent so the waveform reads through it.
    /// One colour for one meaning: this is where the search term is.
    /// </summary>
    private static readonly SolidColorBrush MarkBrush =
        new(Windows.UI.Color.FromArgb(0x88, 0xFF, 0xC1, 0x07));

    /// <summary>
    /// A plain click on the waveform. Kept alongside the pointer handlers because a tap is a
    /// gesture the framework recognises whatever the pointer events did, and seeking by click is
    /// the thing most people will actually do with a waveform.
    /// </summary>
    private void OnWaveformTapped(object sender, TappedRoutedEventArgs e)
    {
        ScrubTo(e.GetPosition(Waveform).X, play: true);
        e.Handled = true;
    }

    private void OnWaveformPressed(object sender, PointerRoutedEventArgs e)
    {
        _scrubbing = true;
        Waveform.CapturePointer(e.Pointer);
        ScrubTo(e.GetCurrentPoint(Waveform).Position.X, play: false);
    }

    private void OnWaveformMoved(object sender, PointerRoutedEventArgs e)
    {
        if (_scrubbing)
        {
            ScrubTo(e.GetCurrentPoint(Waveform).Position.X, play: false);
        }
    }

    /// <summary>
    /// Releasing plays from where the pointer ended up. Dragging alone makes no sound, so the
    /// waveform can be used to read the transcript as well as to listen to it.
    /// </summary>
    private void OnWaveformReleased(object sender, PointerRoutedEventArgs e)
    {
        if (!_scrubbing)
        {
            return;
        }

        _scrubbing = false;
        Waveform.ReleasePointerCapture(e.Pointer);

        ScrubTo(e.GetCurrentPoint(Waveform).Position.X, play: true);
    }

    private void ScrubTo(double x, bool play)
    {
        var width = Waveform.ActualWidth;
        var duration = _viewModel.Player.DurationSeconds;

        if (width <= 1 || duration <= 0)
        {
            return;
        }

        var seconds = Math.Clamp(x / width, 0, 1) * duration;

        _position = seconds;
        MovePlayhead(seconds);
        HighlightAt(seconds);

        if (play || _viewModel.Player.IsPlaying)
        {
            StartPlayback(seconds);
        }
    }

    private void StartPlayback(double seconds)
    {
        _viewModel.Player.PlayFrom(seconds);
        UpdatePlayIcon();
    }

    /// <summary>
    /// Shows what the button will do, read from the player rather than assumed.
    /// <para>
    /// Set from three code paths it drifted out of step with what was actually happening. Asked
    /// for in one place, from the thing that knows, it cannot.
    /// </para>
    /// </summary>
    private void UpdatePlayIcon()
    {
        var playing = _viewModel.Player.IsPlaying;

        // Segoe Fluent Icons: pause while it plays, play while it does not.
        PlayIcon.Glyph = playing ? "\uE769" : "\uE768";

        ToolTipService.SetToolTip(PlayButton, playing ? "Pause" : "Play");
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(PlayButton, playing ? "Pause" : "Play");
    }

    private void PlayParagraph(ParagraphView paragraph)
    {
        TranscriptList.SelectedItem = paragraph;

        // Said rather than done silently. Clicking a line and getting nothing is the failure
        // this whole feature exists to avoid.
        if (!_viewModel.Player.HasAudio)
        {
            StatusText.Text = "No audio is loaded for this transcript, so there is nothing to play.";
            return;
        }

        if (ClicksMustWait(paragraph))
        {
            return;
        }

        Seek(paragraph.StartSeconds, play: true);
    }

    /// <summary>
    /// Renames a speaker, asking how far the change should reach.
    /// <para>
    /// The scope is the question, not an afterthought. Diarization splits one person into two
    /// often enough that "this part is also Kim" has to be possible, and having named someone
    /// once, naming the rest of what they said has to be one action rather than thirty.
    /// </para>
    /// </summary>
    private async void OnRenameSpeaker(object sender, RoutedEventArgs e)
    {
        if (sender is not FrameworkElement { DataContext: ParagraphView paragraph })
        {
            return;
        }

        var input = new TextBox
        {
            Text = paragraph.Speaker,
            PlaceholderText = "Name",
            SelectionStart = 0,
            SelectionLength = paragraph.Speaker.Length,
        };

        // Three scopes, so radio buttons rather than three verbs across the bottom of the
        // dialog. The buttons said "Rename everywhere" and "This part only", which read as
        // opposites and hid the fact that the interesting case is neither.
        var scope = new RadioButtons
        {
            Header = "Apply to",
            SelectedIndex = 0,
            ItemsSource = new[]
            {
                "This part only",
                $"Every part labelled {paragraph.Speaker}",
                "This part and others that sound like them",
            },
        };

        var dialog = new ContentDialog
        {
            XamlRoot = Content.XamlRoot,
            Title = $"Rename {paragraph.Speaker}",
            Content = new StackPanel
            {
                Spacing = 12,
                Children =
                {
                    input,
                    scope,
                    new TextBlock
                    {
                        Text = "Use \u201cevery part\u201d when two labels turn out to be one person, "
                            + "and \u201cothers that sound like them\u201d when one label turns out to "
                            + "be two \u2014 that compares the voices and moves only the matching parts.",
                        TextWrapping = TextWrapping.Wrap,
                        Opacity = 0.8,
                        Style = (Style)Application.Current.Resources["CaptionTextBlockStyle"],
                    },
                },
            },
            PrimaryButtonText = "Rename",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Primary,
        };

        var choice = await dialog.ShowAsync();
        if (choice != ContentDialogResult.Primary || input.Text.Trim().Length == 0)
        {
            return;
        }

        switch (scope.SelectedIndex)
        {
            case 1:
                _viewModel.RenameSpeaker(
                    paragraph.StartSeconds,
                    paragraph.EndSeconds,
                    paragraph.Speaker,
                    input.Text,
                    everywhere: true);
                break;

            case 2:
                await _viewModel.RenameSpeakerByVoiceAsync(
                    paragraph.StartSeconds,
                    paragraph.EndSeconds,
                    paragraph.Speaker,
                    input.Text);
                break;

            default:
                _viewModel.RenameSpeaker(
                    paragraph.StartSeconds,
                    paragraph.EndSeconds,
                    paragraph.Speaker,
                    input.Text,
                    everywhere: false);
                break;
        }
    }

    /// <summary>Opens a saved transcript, reporting a bad file rather than throwing at the user.</summary>
    private void OpenArchive(string path)
    {
        try
        {
            _viewModel.OpenArchive(path);
        }
        catch (Exception exception)
        {
            StatusText.Text = $"Could not open that transcript: {exception.Message}";
        }
    }

    /// <summary>
    /// Whether clicking this paragraph should wait, said out loud when it should.
    /// <para>
    /// Before a stretch is timed, a click there follows the transcriber's drifting stamps —
    /// up to a minute away on a long recording, or past the end of the audio entirely, which
    /// reads as the click doing nothing. The grey ink already says which stretches wait; this
    /// says it to whoever clicks anyway.
    /// </para>
    /// </summary>
    private bool ClicksMustWait(ParagraphView paragraph)
    {
        if (_viewModel.IsBusy
            && paragraph.Segments.Count > 0
            && !_viewModel.IsTimed(paragraph.Segments[0]))
        {
            StatusText.Text = "That part isn't timed yet — the grey lines gain their "
                + "clicks as timing reaches them.";
            return true;
        }

        return false;
    }

    private void OnParagraphClick(object sender, ItemClickEventArgs e)
    {
        if (e.ClickedItem is ParagraphView paragraph)
        {
            PlayParagraph(paragraph);
        }
    }

    /// <summary>
    /// Moves both views at once. The transcript and the recording are the same thing seen two
    /// ways, so a position set from either has to land in both.
    /// </summary>
    private void Seek(double seconds, bool play)
    {
        _position = seconds;
        MovePlayhead(seconds);
        HighlightAt(seconds);

        if (play)
        {
            StartPlayback(seconds);
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
            UpdatePlayIcon();
            return;
        }

        Seek(_position, play: true);
    }

    private void OnPlaybackFailed(string message) =>
        _dispatcher.TryEnqueue(() =>
        {
            StatusText.Text = message;
            UpdatePlayIcon();
        });

    /// <summary>
    /// Narrows the list to paragraphs containing the search text. Filtering rather than
    /// highlighting in place, because the thing people do with a transcript is find the moment
    /// something was said and then listen to it, and every row keeps its timestamp.
    /// </summary>
    private void OnSearchChanged(object sender, TextChangedEventArgs e) => ApplySearch();

    /// <summary>
    /// Marks the search term wherever it appears, as each row is realised.
    /// <para>
    /// Done here rather than by rebuilding the text because a list only realises what is on
    /// screen, so a transcript of any length costs the same. Highlighting is worth having on top
    /// of the filter: the filter says which paragraphs matched, and this says where in them,
    /// which is the part you actually want to read.
    /// </para>
    /// </summary>
    private void OnContainerContentChanging(ListViewBase sender, ContainerContentChangingEventArgs args)
    {
        if (args.InRecycleQueue)
        {
            if (args.Item is ParagraphView recycled)
            {
                _realised.Remove(recycled);
            }

            return;
        }

        if (args.ItemContainer?.ContentTemplateRoot is not FrameworkElement root)
        {
            return;
        }

        if (FindBodyText(root) is not { } body)
        {
            return;
        }

        if (args.Item is ParagraphView paragraph)
        {
            _realised.Add(paragraph);
            FillWithWords(body, paragraph);
            HighlightMatches(body, paragraph);
            return;
        }

        HighlightMatches(body);
    }

    /// <summary>
    /// Lays the paragraph out one word at a time, each one clickable.
    /// <para>
    /// The reason a paragraph is not simply bound to its text. Clicking a line already seeks to
    /// where the line began, which on a paragraph of forty words means seeking as much as twenty
    /// seconds before the part being pointed at. Every word carries its own time instead.
    /// </para>
    /// <para>
    /// Built here rather than in the template because it is per-item work on a virtualised list:
    /// only the paragraphs actually on screen are ever laid out this way.
    /// </para>
    /// </summary>
    private void FillWithWords(TextBlock body, ParagraphView paragraph)
    {
        body.Inlines.Clear();

        var words = _viewModel.WordsIn(paragraph.Segments);

        if (words.Count == 0)
        {
            // No recording to seek in — a transcript can still be read.
            body.Text = paragraph.Text;
            return;
        }

        var spans = new List<SpokenWord>(words.Count);
        var offset = 0;
        var wordAt = 0;

        // Words are built a segment at a time so the clickable boundary is visible: during a
        // run, a segment whose words have been measured draws in ink and takes clicks, and one
        // still waiting for the scan draws grey and takes none. The same rule the status line
        // states — "clickable through, grey lines follow" — drawn where the reader is looking.
        foreach (var segment in paragraph.Segments)
        {
            var own = _viewModel.WordsIn([segment]);
            var timed = !_viewModel.IsBusy || _viewModel.IsTimed(segment);

            for (var i = 0; i < own.Count; i++)
            {
                var at = own[i].StartSeconds;

                if (timed)
                {
                    var link = new Hyperlink
                    {
                        UnderlineStyle = UnderlineStyle.None,
                        Foreground = TextBrush(body),
                    };

                    link.Inlines.Add(new Run { Text = own[i].Text });

                    // A little before the word, not exactly on it. Seeking to the instant a
                    // word begins starts playback inside its first consonant, which sounds
                    // like a miss however accurate the timing was — the ear needs a moment of
                    // run-up to hear a word whole.
                    link.Click += (_, _) => Seek(Math.Max(0, at - RunUpSeconds), play: true);

                    body.Inlines.Add(link);
                }
                else
                {
                    body.Inlines.Add(new Run
                    {
                        Text = own[i].Text,
                        Foreground = (Brush)Application.Current.Resources["TextFillColorSecondaryBrush"],
                    });
                }

                spans.Add(new SpokenWord(
                    own[i].StartSeconds, own[i].EndSeconds, offset, own[i].Text.Length, own[i].Text));
                offset += own[i].Text.Length;
                wordAt++;

                if (wordAt < words.Count)
                {
                    body.Inlines.Add(new Run { Text = " " });
                    offset++;
                }
            }
        }

        // Checked against the text the block actually ended up with. The offsets above are
        // counted while building, which assumes the laid-out text is exactly the words joined by
        // single spaces; if that assumption is ever wrong, every marker lands on the wrong
        // characters and nothing says so. Where the text can be read back, the words are found
        // in it instead.
        var laid = body.Text;

        if (!string.IsNullOrEmpty(laid))
        {
            var cursor = 0;

            for (var i = 0; i < spans.Count; i++)
            {
                var found = laid.IndexOf(spans[i].Text, cursor, StringComparison.Ordinal);
                if (found < 0)
                {
                    break;
                }

                spans[i] = spans[i] with { Offset = found };
                cursor = found + spans[i].Text.Length;
            }
        }

        _spokenWords[paragraph] = spans;
    }

    /// <summary>Where one word sits, both in the recording and in the laid-out text.</summary>
    private sealed record SpokenWord(double From, double To, int Offset, int Length, string Text);

    private readonly Dictionary<ParagraphView, IReadOnlyList<SpokenWord>> _spokenWords = [];

    /// <summary>
    /// How far before a word to start playing it. Enough to hear the word begin rather than
    /// arriving in the middle of it.
    /// </summary>
    private const double RunUpSeconds = 0.12;

    /// <summary>Redraws the paragraphs on screen in the colours of the theme now in force.</summary>
    private void OnThemeChanged(FrameworkElement sender, object args)
    {
        foreach (var paragraph in _realised.ToList())
        {
            if (TranscriptList.ContainerFromItem(paragraph) is ContentControl container
                && container.ContentTemplateRoot is FrameworkElement root
                && FindBodyText(root) is { } body)
            {
                FillWithWords(body, paragraph);
                HighlightMatches(body, paragraph);
            }
        }
    }

    /// <summary>
    /// The colour a word should be, asked of the theme rather than copied off the block.
    /// <para>
    /// A hyperlink does not inherit the surrounding text colour — left alone it is accent blue —
    /// so one has to be given. Taking the block's own brush looks like the careful answer and is
    /// not: it is read while the row is being built, which during a transcription happens over
    /// and over as segments arrive, and whatever it returns is then frozen onto every word in
    /// that row. Read at the wrong moment it hands near-black ink to a near-black page, and the
    /// paragraph simply is not there. The timestamps beside it stay visible because nothing was
    /// ever copied onto them.
    /// </para>
    /// <para>
    /// Asking the theme for the same resource the block's own style uses gives an answer that is
    /// right for the theme in force, whenever it is asked.
    /// </para>
    /// </summary>
    private static Brush? TextBrush(TextBlock body) =>
        Application.Current.Resources.TryGetValue("TextFillColorPrimaryBrush", out var brush)
            ? brush as Brush
            : body.Foreground;

    /// <summary>The paragraph's own text, found by its tag rather than by position in the tree.</summary>
    private static TextBlock? FindBodyText(DependencyObject element)
    {
        if (element is TextBlock { Tag: "body" } tagged)
        {
            return tagged;
        }

        var children = VisualTreeHelper.GetChildrenCount(element);

        for (var i = 0; i < children; i++)
        {
            if (FindBodyText(VisualTreeHelper.GetChild(element, i)) is { } found)
            {
                return found;
            }
        }

        return null;
    }

    private void HighlightMatches(TextBlock body) => HighlightMatches(body, null);

    /// <summary>
    /// Marks what the reader is looking for and what they are hearing, in one pass.
    /// <para>
    /// Both at once because the highlighters are a single collection on the text block: applying
    /// one by clearing the collection would take the other away with it.
    /// </para>
    /// </summary>
    private void HighlightMatches(TextBlock body, ParagraphView? paragraph)
    {
        body.TextHighlighters.Clear();

        var query = SearchBox.Text.Trim();

        if (query.Length > 0)
        {
            var found = new TextHighlighter
            {
                Background = HighlightBackground,
                Foreground = HighlightForeground,
            };

            var text = body.Text;
            var at = text.IndexOf(query, StringComparison.OrdinalIgnoreCase);

            while (at >= 0)
            {
                found.Ranges.Add(new TextRange { StartIndex = at, Length = query.Length });
                at = text.IndexOf(query, at + query.Length, StringComparison.OrdinalIgnoreCase);
            }

            if (found.Ranges.Count > 0)
            {
                body.TextHighlighters.Add(found);
            }
        }

        // The word being said right now. This is what word timings are actually for: a number
        // that is right to a twentieth of a second is invisible until something moves with the
        // audio, and then it is the difference between a transcript and a place in a recording.
        // Nothing is lit in a paragraph the playhead has left. Without this the paragraph being
        // cleared lights up instead: every word in it has started by then, and "the last word to
        // have started" happily returns the final one, so the pass meant to put the marker out
        // was the thing keeping it on.
        //
        // Within the paragraph, the word covering this instant when there is one, and otherwise
        // the last to have started. Measured words do not touch — the quiet between them belongs
        // to neither — so containment alone leaves the marker holding nothing several times a
        // second, while starting alone skips ahead wherever two segments of a paragraph overlap
        // in time, which they do.
        if (paragraph is not null
            && paragraph.Contains(_position)
            && _viewModel.Player.IsPlaying
            && _spokenWords.TryGetValue(paragraph, out var spans)
            && (spans.FirstOrDefault(w => _position >= w.From && _position < w.To)
                ?? spans.LastOrDefault(w => _position >= w.From)) is { } speaking)
        {
            body.TextHighlighters.Add(new TextHighlighter
            {
                Background = SpokenBackground,
                Foreground = SpokenForeground,
                Ranges = { new TextRange { StartIndex = speaking.Offset, Length = speaking.Length } },
            });
        }
    }

    /// <summary>
    /// Which paragraph is being spoken now.
    /// <para>
    /// Not simply the first one covering the instant. Segments are placed by measuring where
    /// their words are, so two can cover the same moment and one can sit wholly inside another —
    /// on a podcast, twelve did. Taking the first leaves the nested one unable to ever light,
    /// and lights its neighbour instead.
    /// </para>
    /// <para>
    /// A paragraph with a word actually covering the instant wins, because that is the question
    /// being asked. Failing that, the last one to have begun: between two that overlap, the one
    /// that started more recently is the one being spoken.
    /// </para>
    /// </summary>
    private ParagraphView? Playing(double seconds)
    {
        ParagraphView? covering = null;
        ParagraphView? latest = null;

        foreach (var paragraph in _paragraphs)
        {
            // A word covering this instant settles it, whether or not the paragraph's own bounds
            // reach that far. They often do not: a segment's end is pulled back to where the next
            // one begins so that no two claim the same moment, while its last words were found by
            // searching a little past that and are still being spoken there. Asking the bounds
            // first hands that second to the following paragraph and leaves the words that are
            // actually sounding unlit.
            if (covering is null
                && _spokenWords.TryGetValue(paragraph, out var spans)
                && spans.Any(w => seconds >= w.From && seconds < w.To))
            {
                covering = paragraph;
            }

            if (paragraph.Contains(seconds) && (latest is null || paragraph.StartSeconds >= latest.StartSeconds))
            {
                latest = paragraph;
            }
        }

        return covering ?? latest;
    }

    /// <summary>
    /// Which word of a paragraph is being said, or the last one to have started.
    /// <para>
    /// The same rule <see cref="HighlightMatches(TextBlock, ParagraphView?)"/> paints by, kept
    /// here so that the decision to repaint and the decision of what to paint cannot drift apart.
    /// Answering with an index rather than the word itself is what lets a repaint be skipped when
    /// nothing has moved.
    /// </para>
    /// </summary>
    private static int WordAt(IReadOnlyList<SpokenWord> spans, double position)
    {
        var covering = -1;
        var started = -1;

        for (var i = 0; i < spans.Count; i++)
        {
            // Whichever began most recently, not whichever comes last in the paragraph. The two
            // are the same only while the words are in the order they were said, and a paragraph
            // holding an interjection is not: "…how they secure AI agents, Of course, that's
            // hugely topical. Or they're talking about…" reads in one order and was spoken in
            // another. Taking the last match in reading order then jumps the marker to the end of
            // the paragraph and leaves it there.
            if (position >= spans[i].From && position < spans[i].To)
            {
                if (covering < 0 || spans[i].From > spans[covering].From)
                {
                    covering = i;
                }
            }
            else if (position >= spans[i].From && (started < 0 || spans[i].From > spans[started].From))
            {
                started = i;
            }
        }

        return covering >= 0 ? covering : started;
    }

    /// <summary>
    /// Repaints so the marker follows the sound.
    /// <para>
    /// Only when the answer has changed, which is a few times a second rather than twenty.
    /// </para>
    /// </summary>
    private void FollowSpokenWord()
    {
        var paragraph = Playing(_position);

        var marked = paragraph is not null && _spokenWords.TryGetValue(paragraph, out var spans)
            ? WordAt(spans, _position)
            : -1;

        if (ReferenceEquals(paragraph, _lit) && marked == _markedWord)
        {
            return;
        }

        _lit = paragraph;
        _markedWord = marked;

        // Every paragraph on screen, not just the one being played and the one just left.
        // Tracking which paragraph needs clearing has now been wrong twice — first by never
        // repainting the one left behind, then by repainting it with a rule that lit it again —
        // and there are only ever a dozen or so realised at once. Repainting all of them cannot
        // leave a marker behind anywhere.
        foreach (var visible in _realised.ToList())
        {
            if (TranscriptList.ContainerFromItem(visible) is ContentControl container
                && container.ContentTemplateRoot is FrameworkElement root
                && FindBodyText(root) is { } body)
            {
                HighlightMatches(body, visible);
            }
            else
            {
                _realised.Remove(visible);
            }
        }
    }

    /// <summary>
    /// Starts or stops the working glow, from the same rule the Mac window uses: models busy
    /// over loaded audio, or the microphone being prepared. Recording itself gets no glow —
    /// the red banner owns that state, and two ornaments saying different things is worse
    /// than one.
    /// </summary>
    private void UpdateGlow()
    {
        var working = (_viewModel.IsBusy && _viewModel.Player.HasAudio) || _viewModel.IsPreparing;

        if (working && _glowTimer is null && AiGlowRing.Visibility != Visibility.Visible)
        {
            AiGlowRing.Visibility = Visibility.Visible;
            AiGlowHalo.Visibility = Visibility.Visible;

            // A person who has turned animations off system-wide gets a still ring: the
            // information (something is working) without the motion they opted out of.
            if (!new Windows.UI.ViewManagement.UISettings().AnimationsEnabled)
            {
                AiGlowHalo.Opacity = 0.4;
                return;
            }

            // Fifteen frames a second, the cadence the Mac settled on: at thirty the ornament
            // was measurably competing with the work it announced.
            _glowTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(66) };
            _glowTimer.Tick += (_, _) => TurnGlow();
            _glowTimer.Start();
        }
        else if (!working && AiGlowRing.Visibility == Visibility.Visible)
        {
            _glowTimer?.Stop();
            _glowTimer = null;

            AiGlowRing.Visibility = Visibility.Collapsed;
            AiGlowHalo.Visibility = Visibility.Collapsed;
            AiGlowHalo.Opacity = 0;
        }
    }

    /// <summary>One step of the sweep: the gradient turns, the halo breathes.</summary>
    private void TurnGlow()
    {
        _glowTicks++;

        // No conic gradient in WinUI, so the sweep is a linear gradient whose axis rotates.
        // The axis runs a little past the box so the clamped ends — solid runs of the two
        // end colours — stay in the corners instead of swallowing whole edges.
        var radians = _glowTicks * 5 * Math.PI / 180;
        var dx = Math.Cos(radians) * 0.62;
        var dy = Math.Sin(radians) * 0.62;

        var start = new Windows.Foundation.Point(0.5 - dx, 0.5 - dy);
        var end = new Windows.Foundation.Point(0.5 + dx, 0.5 + dy);

        GlowRingBrush.StartPoint = start;
        GlowRingBrush.EndPoint = end;
        GlowHaloBrush.StartPoint = start;
        GlowHaloBrush.EndPoint = end;

        // A four-second breath, out of phase with the sweep so the two read as independent
        // life rather than one mechanism.
        AiGlowHalo.Opacity = 0.32 + (0.18 * Math.Sin(_glowTicks * 0.066 * Math.PI / 2));
    }

    private DispatcherTimer? _glowTimer;
    private long _glowTicks;

    /// <summary>The paragraph currently carrying the spoken-word marker, and which word.</summary>
    private ParagraphView? _lit;

    private int _markedWord = -1;

    /// <summary>The paragraphs the list has actually built containers for.</summary>
    private readonly HashSet<ParagraphView> _realised = [];

    /// <summary>
    /// A literal highlighter pen rather than the accent colour. The accent already means "this
    /// is the paragraph playing", and two different meanings in one colour is worse than a
    /// second colour.
    /// </summary>
    private static readonly SolidColorBrush HighlightBackground =
        new(Windows.UI.Color.FromArgb(255, 0xFF, 0xE1, 0x6A));

    private static readonly SolidColorBrush HighlightForeground =
        new(Windows.UI.Color.FromArgb(255, 0x1A, 0x1A, 0x1A));

    /// <summary>
    /// Behind the word being said. Deliberately faint — this moves several times a second while
    /// the recording plays, and a strong colour flickering through a paragraph is harder to read
    /// than no marker at all. The search colour stays the loud one, because a search result is
    /// something to find rather than something to follow.
    /// </summary>
    private static readonly SolidColorBrush SpokenBackground =
        new(Windows.UI.Color.FromArgb(255, 0xD6, 0xE4, 0xFF));

    /// <summary>
    /// And the text on it. A highlighter sets the paper and not the ink, which works while the
    /// ink is dark: in a dark theme the text is nearly white, and near-white on pale blue is
    /// unreadable exactly where the reader is being asked to look. Both marks now name both
    /// colours, so neither depends on what the rest of the window happens to be doing.
    /// </summary>
    private static readonly SolidColorBrush SpokenForeground =
        new(Windows.UI.Color.FromArgb(255, 0x1A, 0x1A, 0x1A));

    /// <summary>
    /// The empty state is only for an empty transcript, not for a search that found nothing —
    /// "start listening or open a file" is unhelpful advice when the answer is to try another
    /// word. The match count already says that.
    /// </summary>
    /// <summary>
    /// Re-marks rows already on screen, once the list has actually built them.
    /// <para>
    /// Queued rather than run directly. Setting ItemsSource does not realise containers
    /// synchronously, so asking for them on the same tick finds nothing and the marks never
    /// appear — which is exactly how this failed the first time.
    /// </para>
    /// </summary>
    private void QueueHighlightRefresh() =>
        _dispatcher.TryEnqueue(DispatcherQueuePriority.Low, RefreshHighlights);

    private void RefreshHighlights()
    {
        DrawSearchMarks();

        for (var i = 0; i < TranscriptList.Items.Count; i++)
        {
            if (TranscriptList.ContainerFromIndex(i) is FrameworkElement container
                && FindBodyText(container) is { } body)
            {
                HighlightMatches(body);
            }
        }
    }

    /// <summary>Non-overlapping, case-insensitive occurrences of a term in a paragraph.</summary>
    private static int Occurrences(string text, string query)
    {
        if (query.Length == 0)
        {
            return 0;
        }

        var count = 0;
        var at = text.IndexOf(query, StringComparison.OrdinalIgnoreCase);

        while (at >= 0)
        {
            count++;
            at = text.IndexOf(query, at + query.Length, StringComparison.OrdinalIgnoreCase);
        }

        return count;
    }

    private void UpdateEmptyState(int showing)
    {
        var nothingAtAll = _paragraphs.Count == 0 && !_viewModel.IsRecording && !_viewModel.IsBusy;

        EmptyState.Visibility = nothingAtAll ? Visibility.Visible : Visibility.Collapsed;
        SearchBox.IsEnabled = _paragraphs.Count > 0;

        _ = showing;
    }

    private void OnFindAccelerator(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args)
    {
        SearchBox.Focus(FocusState.Programmatic);
        SearchBox.SelectAll();
        args.Handled = true;
    }

    private void OnEscapeAccelerator(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args)
    {
        if (SearchBox.Text.Length > 0)
        {
            SearchBox.Text = string.Empty;
            args.Handled = true;
        }
    }

    /// <summary>
    /// Control and space rather than space alone. The transcript is a list, and a list already
    /// uses space for the thing under the cursor.
    /// </summary>
    private void OnPlayAccelerator(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args)
    {
        OnPlayPause(sender, new RoutedEventArgs());
        args.Handled = true;
    }

    /// <summary>Everything the list shows outside a search, updated in place, never swapped.</summary>
    private readonly System.Collections.ObjectModel.ObservableCollection<ParagraphView> _shown = [];

    private void ApplySearch()
    {
        var query = SearchBox.Text.Trim();

        if (query.Length == 0)
        {
            // The collection is edited to match rather than reassigned: reassigning the
            // ItemsSource rebuilds every container on screen, which reads as a blink.
            for (var i = 0; i < _paragraphs.Count; i++)
            {
                if (i >= _shown.Count)
                {
                    _shown.Add(_paragraphs[i]);
                }
                else if (!ReferenceEquals(_shown[i], _paragraphs[i]))
                {
                    _shown[i] = _paragraphs[i];
                }
            }

            while (_shown.Count > _paragraphs.Count)
            {
                _shown.RemoveAt(_shown.Count - 1);
            }

            if (!ReferenceEquals(TranscriptList.ItemsSource, _shown))
            {
                TranscriptList.ItemsSource = _shown;
            }

            SearchCount.Text = string.Empty;
            UpdateEmptyState(_paragraphs.Count);
            QueueHighlightRefresh();
            return;
        }

        var matches = _paragraphs
            .Where(p => p.Text.Contains(query, StringComparison.OrdinalIgnoreCase)
                || p.Speaker.Contains(query, StringComparison.OrdinalIgnoreCase))
            .ToList();

        TranscriptList.ItemsSource = matches;
        UpdateEmptyState(matches.Count);
        QueueHighlightRefresh();

        // Occurrences, not paragraphs. Every one of them is marked in the text and on the
        // waveform, so counting the paragraphs they happen to sit in reports a different number
        // from the one on screen.
        var hits = matches.Sum(p => Occurrences(p.Text, query));

        SearchCount.Text = hits switch
        {
            0 => "no matches",
            1 => "1 match",
            _ => $"{hits} matches",
        };
    }

    /// <summary>
    /// Follows playback by moving the selection, so the highlight is the same mechanism the user
    /// clicked with rather than a second one that has to be kept in step with it.
    /// </summary>
    private void OnPlaybackPosition(double seconds)
    {
        Interlocked.Exchange(ref _latestPositionBits, BitConverter.DoubleToInt64Bits(seconds));

        // One update in flight at a time, always carrying the newest position. Twenty arrive a
        // second whatever the UI is doing, and queueing each one means a repaint that costs more
        // than its fifty milliseconds puts the queue permanently behind: the marker then replays
        // the past, a few words back where the paragraphs are short and sentences back where
        // they are long, and it cannot catch up while sound keeps coming. A dropped stale
        // position is invisible; an accumulated one was not.
        if (Interlocked.Exchange(ref _positionUpdateQueued, 1) == 1)
        {
            return;
        }

        _dispatcher.TryEnqueue(() =>
        {
            // Cleared before reading, so a position arriving after the read queues a fresh pass.
            Interlocked.Exchange(ref _positionUpdateQueued, 0);
            var latest = BitConverter.Int64BitsToDouble(Interlocked.Read(ref _latestPositionBits));

            // Ignored while dragging: the pointer is the authority then, and letting playback
            // fight it makes the waveform jitter under the finger.
            if (_scrubbing)
            {
                return;
            }

            _position = latest;
            MovePlayhead(latest);
            HighlightAt(latest);
            FollowSpokenWord();
        });
    }

    /// <summary>The newest reported playback position, as the bits of a double.</summary>
    private long _latestPositionBits;

    /// <summary>Whether a position update is already waiting for the UI thread.</summary>
    private int _positionUpdateQueued;

    private void OnPlaybackStopped() =>
        _dispatcher.TryEnqueue(() =>
        {
            UpdatePlayIcon();

            // Nothing is being said any more, so nothing should look as though it is.
            _lit = null;
            _markedWord = -1;

            foreach (var visible in _realised.ToList())
            {
                if (TranscriptList.ContainerFromItem(visible) is ContentControl container
                    && container.ContentTemplateRoot is FrameworkElement root
                    && FindBodyText(root) is { } body)
                {
                    HighlightMatches(body, visible);
                }
            }
        });


    private void OnCopy(object sender, RoutedEventArgs e)
    {
        var package = new Windows.ApplicationModel.DataTransfer.DataPackage();
        package.SetText(_viewModel.Export(TranscriptFormat.PlainText));

        Windows.ApplicationModel.DataTransfer.Clipboard.SetContent(package);

        StatusText.Text = "Transcript copied.";
    }

    private async void OnSave(object sender, RoutedEventArgs e) => await SaveViaPickerAsync();

    /// <summary>
    /// Offers the save dialog and reports whether anything was actually written — which is what
    /// closing needs to know, since a cancelled picker means the user changed their mind.
    /// </summary>
    private async Task<bool> SaveViaPickerAsync()
    {
        var picker = new FileSavePicker { SuggestedFileName = _viewModel.SourceName };
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(this));

        // First in the list, so it is what the dialog offers by default. The others keep the
        // words; only this one keeps the recording with them.
        picker.FileTypeChoices.Add("Transcript and audio", [TranscriptArchive.Extension]);
        picker.FileTypeChoices.Add("Text", [".txt"]);
        picker.FileTypeChoices.Add("Markdown", [".md"]);
        picker.FileTypeChoices.Add("Subtitles", [".srt"]);

        var file = await picker.PickSaveFileAsync();
        if (file is null)
        {
            return false;
        }

        var extension = Path.GetExtension(file.Name).ToLowerInvariant();

        if (TranscriptArchive.IsArchive(file.Name))
        {
            try
            {
                await Task.Run(() => _viewModel.SaveArchive(file.Path));
                StatusText.Text = $"Saved to {file.Path}, with the recording.";
                return true;
            }
            catch (Exception exception)
            {
                StatusText.Text = $"Could not save: {exception.Message}";
                return false;
            }
        }

        var format = extension switch
        {
            ".md" => TranscriptFormat.Markdown,
            ".srt" => TranscriptFormat.SubRip,
            _ => TranscriptFormat.PlainText,
        };

        await Windows.Storage.FileIO.WriteTextAsync(file, _viewModel.Export(format));

        StatusText.Text = $"Saved to {file.Path}";
        return true;
    }

    /// <summary>
    /// Stops the window closing over unsaved work without asking.
    /// <para>
    /// The recording behind a transcript often exists nowhere but this window's memory — a live
    /// recording never had a file, and a transcription is minutes of model time — so an
    /// accidental close is the one click in the app that destroys something irreplaceable.
    /// A transcript opened from a file and left alone closes silently; it is already safe.
    /// </para>
    /// </summary>
    private void OnWindowClosing(AppWindow sender, AppWindowClosingEventArgs e)
    {
        if (_closeConfirmed || !_viewModel.HasUnsavedWork)
        {
            return;
        }

        // The event cannot wait on a dialog, so the close is refused and asked about; closing
        // again is done here once the answer is known.
        e.Cancel = true;

        if (!_confirmingClose)
        {
            _confirmingClose = true;
            _ = ConfirmCloseAsync();
        }
    }

    private async Task ConfirmCloseAsync()
    {
        try
        {
            var dialog = new ContentDialog
            {
                XamlRoot = Content.XamlRoot,
                Title = "Save this transcript?",
                Content = "It exists only in this window until it is saved. Saving keeps the "
                    + "words and the recording together in one file; discarding lets both go.",
                PrimaryButtonText = "Save",
                SecondaryButtonText = "Discard",
                CloseButtonText = "Cancel",
                DefaultButton = ContentDialogButton.Primary,
            };

            var choice = await dialog.ShowAsync();

            if (choice == ContentDialogResult.None)
            {
                return;
            }

            if (choice == ContentDialogResult.Primary && !await SaveViaPickerAsync())
            {
                // A cancelled picker or a failed save is not consent to lose the work.
                return;
            }

            _closeConfirmed = true;
            Close();
        }
        finally
        {
            _confirmingClose = false;
        }
    }

    /// <summary>Whether losing the transcript on close has been explicitly agreed to.</summary>
    private bool _closeConfirmed;

    private bool _confirmingClose;

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

    private async void OnOpenRecording(object sender, RoutedEventArgs e)
    {
        var picker = PickerFor(AudioFileLoader.SupportedExtensions);

        var file = await picker.PickSingleFileAsync();
        if (file is null)
        {
            return;
        }

        // A saved transcript can arrive here too — dropped filters do not stop typing a name —
        // and it is opened, not transcribed again: it already has its words, its timings and
        // its speakers, and running the model over it would only spend minutes arriving
        // somewhere worse.
        if (TranscriptArchive.IsArchive(file.Name))
        {
            OpenArchive(file.Path);
            return;
        }

        await _viewModel.TranscribeFileAsync(file.Path);
    }

    private async void OnOpenTranscript(object sender, RoutedEventArgs e)
    {
        var picker = PickerFor(TranscriptArchive.Extensions);

        var file = await picker.PickSingleFileAsync();
        if (file is null)
        {
            return;
        }

        OpenArchive(file.Path);
    }

    private FileOpenPicker PickerFor(IEnumerable<string> extensions)
    {
        var picker = new FileOpenPicker();

        // A picker created in an unpackaged app has no window of its own and must be told which
        // window owns it, or the call fails at runtime rather than at compile time.
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(this));

        foreach (var extension in extensions)
        {
            picker.FileTypeFilter.Add(extension);
        }

        return picker;
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
    /// <summary>
    /// Asks how many people are talking, then works the speakers out again.
    /// <para>
    /// The count is the one thing the listener knows and the algorithm cannot. Inferring it from
    /// a distance threshold fails in both directions at once on a real recording — two people
    /// who sound alike merge while one who leaned towards the microphone splits in two — and no
    /// single threshold fixes both. Being told there are three ends the argument.
    /// </para>
    /// </summary>
    private async void OnEditSpeakers(object sender, RoutedEventArgs e)
    {
        var choices = new ComboBox { SelectedIndex = 0, MinWidth = 220 };
        choices.Items.Add("Work it out automatically");

        for (var n = 1; n <= 10; n++)
        {
            choices.Items.Add(n == 1 ? "1 speaker" : $"{n} speakers");
        }

        var dialog = new ContentDialog
        {
            XamlRoot = Content.XamlRoot,
            Title = "How many speakers?",
            Content = new StackPanel
            {
                Spacing = 12,
                Children =
                {
                    choices,
                    new TextBlock
                    {
                        Text = "Telling LocalScribe the number is far more reliable than letting it "
                            + "guess. Guessing tends to go wrong in both directions on the same "
                            + "recording, merging two voices that sound alike while splitting one "
                            + "that changes distance from the microphone.",
                        TextWrapping = TextWrapping.Wrap,
                        Opacity = 0.8,
                    },
                },
            },
            PrimaryButtonText = "Find speakers",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Primary,
        };

        if (await dialog.ShowAsync() != ContentDialogResult.Primary)
        {
            return;
        }

        var speakers = choices.SelectedIndex == 0 ? (int?)null : choices.SelectedIndex;

        await _viewModel.FindSpeakersAsync(speakers);
    }

    /// <summary>
    /// Collects the terms the cleanup model should spell correctly: names, products, jargon.
    /// <para>
    /// This is the cheapest accuracy win available and no larger Whisper model substitutes for
    /// it — the model has never heard of your colleagues or your product, and will keep guessing
    /// at them however big it gets. It is also the least obvious feature in the app, so the
    /// dialog explains itself rather than presenting an empty box.
    /// </para>
    /// </summary>
    private async void OnEditGlossary(object sender, RoutedEventArgs e)
    {
        var textBox = new TextBox
        {
            AcceptsReturn = true,
            Height = 200,
            Text = string.Join(Environment.NewLine, _viewModel.Glossary),
            PlaceholderText = "One per line, e.g.\r\nSiobhan\r\nKubernetes\r\nLocalScribe",
        };

        var explanation = new TextBlock
        {
            Text = "Whisper has never heard of your colleagues, your products or your acronyms, "
                + "and guesses at them — usually the same way every time. Terms listed here are "
                + "given to the cleanup model, which corrects them to your spelling wherever the "
                + "transcript clearly meant them.",
            TextWrapping = TextWrapping.Wrap,
            Opacity = 0.85,
        };

        var content = new StackPanel { Spacing = 12, Children = { explanation, textBox } };

        // The part worth saying loudest: without a cleanup model this list does nothing at all,
        // and neither do the summary or the punctuation repair.
        if (_viewModel.CleanupModel is { } model)
        {
            content.Children.Add(new TextBlock
            {
                Text = $"Cleanup is running on {model}.",
                TextWrapping = TextWrapping.Wrap,
                Opacity = 0.7,
                Style = (Style)Application.Current.Resources["CaptionTextBlockStyle"],
            });
        }
        var dialog = new ContentDialog
        {
            XamlRoot = Content.XamlRoot,
            Title = "Glossary",
            Content = content,
            PrimaryButtonText = "Save",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Primary,
        };

        if (_viewModel.CleanupModel is null)
        {
            // The one-click way out, not just the diagnosis. Everything the button does is
            // named in the message, runs on this machine, and reports into the status line —
            // the model is a download, not an account or a service.
            var fetch = new Button { Content = "Download and start it" };

            fetch.Click += async (_, _) =>
            {
                dialog.Hide();
                await _viewModel.ProvisionCleanupAsync();
            };

            content.Children.Add(new InfoBar
            {
                IsOpen = true,
                IsClosable = false,
                Severity = InfoBarSeverity.Warning,
                Title = "No cleanup model is running",
                Message = "The glossary has no effect until one is. Transcription itself is "
                    + "unaffected — only the glossary corrections and punctuation repair "
                    + "need it. Downloading fetches Foundry Local's small instruct model "
                    + "(about a gigabyte, one time) and runs it here; progress lands in the "
                    + "status line. If you use GenieX instead, just start it and reopen the app.",
                ActionButton = fetch,
            });
        }

        if (await dialog.ShowAsync() == ContentDialogResult.Primary)
        {
            _viewModel.Glossary.Clear();
            _viewModel.Glossary.AddRange(
                textBox.Text.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries
                    | StringSplitOptions.TrimEntries));
        }
    }
}
