using System.ComponentModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using LocalScribe.App;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Models;
using LocalScribe.Core.Pipeline;
using LocalScribe.Core.Provisioning;
using LocalScribe.Core.Transcription;
using LocalScribe.Onnx;

namespace LocalScribe.Desktop;

/// <summary>
/// The macOS window over the shared <see cref="MainViewModel"/>. The window owns only
/// presentation; every behaviour with a history — coalesced playback updates, the marker
/// repaint rule, the close gate — is ported here deliberately from the WinUI window, each with
/// its reason attached.
/// </summary>
public sealed partial class MainWindow : Window
{
    private readonly MainViewModel _viewModel;
    private readonly string? _openOnLaunch;

    private readonly List<Border> _paragraphBorders = [];
    private IReadOnlyList<TranscriptParagraph> _shownParagraphs = [];
    private bool _closeApproved;

    public MainWindow()
        : this(null)
    {
    }

    private readonly WaveformView _waveform = new();

    public MainWindow(string? openOnLaunch)
    {
        _openOnLaunch = openOnLaunch;

        _viewModel = new MainViewModel(FindModelRoot(), OpenEngine);
        _viewModel.OffersWordCorrection = true;
        _viewModel.PropertyChanged += OnViewModelChanged;
        _viewModel.RunEnded += OnRunEnded;
        _viewModel.Player.PositionChanged += OnPlaybackPosition;
        _viewModel.Player.Stopped += OnPlaybackStopped;
        _viewModel.Player.Failed += message => Post(() => StatusText.Text = message);

        InitializeComponent();
        BuildMenus();
        SetUpAlert();

        _waveform.PeakSource = buckets => _viewModel.WaveformPeaks(buckets);

        // Scrubbing moves the marker and the words under it live, with the sound held; on
        // release the sound resumes exactly where the drag ended, but only if it was playing
        // — a scrub while stopped is navigation, not a request for audio. A plain tap keeps
        // its old meaning: play from here.
        _waveform.TapSeek += StartPlayback;
        _waveform.ScrubStarted += () =>
        {
            _scrubWasPlaying = _viewModel.Player.IsPlaying;

            if (_scrubWasPlaying)
            {
                _viewModel.Player.Stop();
            }

            _followPlayback = true;
        };
        _waveform.Scrubbed += seconds => PaintMarker(seconds);
        _waveform.ScrubEnded += seconds =>
        {
            if (_scrubWasPlaying)
            {
                StartPlayback(seconds);
            }
            else
            {
                PaintMarker(seconds);
            }
        };
        WaveformHost.Content = _waveform;
        WatchForReaderScrolling();

        RefreshControls();

        Opened += OnOpened;
        Closing += OnClosing;

        AddHandler(DragDrop.DragOverEvent, OnDragOver);
        AddHandler(DragDrop.DropEvent, OnDrop);

        // Speaker colours come in a light and a dark set, chosen when a row is built, so a
        // change of appearance rebuilds the rows rather than leaving one theme's colours on
        // the other's background.
        ActualThemeVariantChanged += (_, _) =>
        {
            if (_viewModel.IsBusy)
            {
                ScheduleStreamingRebuild();
            }
            else
            {
                RebuildParagraphs();
            }

            _waveform.InvalidateVisual();
        };

        SizeChanged += (_, e) => FitTitleBar(e.NewSize.Width);
    }

    /// <summary>
    /// A narrow window keeps every control and drops words first: Open and Record become
    /// glyphs like the rest, then the search field gives up some width. The document name
    /// between them trims itself.
    /// </summary>
    private void FitTitleBar(double width)
    {
        OpenLabel.IsVisible = width >= 760;
        RecordLabel.IsVisible = width >= 760;
        SearchBox.Width = width >= 700 ? 170 : 120;
    }

    private string? DroppedFile(DragEventArgs e)
    {
        if (e.DataTransfer.TryGetFiles()?.FirstOrDefault()?.TryGetLocalPath() is not { } path)
        {
            return null;
        }

        var extension = Path.GetExtension(path);
        var known = AudioFileLoader.SupportedExtensions.Contains(extension, StringComparer.OrdinalIgnoreCase)
            || extension.Equals(".scrb", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".lscribe", StringComparison.OrdinalIgnoreCase);

        return known ? path : null;
    }

    private void OnDragOver(object? sender, DragEventArgs e) =>
        e.DragEffects = OpenButton.IsEnabled && DroppedFile(e) is not null
            ? DragDropEffects.Copy
            : DragDropEffects.None;

    /// <summary>A dropped file opens exactly as Open would open it, and only when Open could.</summary>
    private async void OnDrop(object? sender, DragEventArgs e)
    {
        if (OpenButton.IsEnabled && DroppedFile(e) is { } path)
        {
            await OpenPathAsync(path);
        }
    }

    private async void OnOpened(object? sender, EventArgs e)
    {
        Opened -= OnOpened;

        await ProvisionModelsAsync();

        await _viewModel.InitialiseAsync();

        SyncPaceControls();

        // Nemotron tells eight people apart and the pipeline ten; offering the stepper more
        // than this machine's engine can deliver would be a promise the models cannot keep.
        SpeakerCountBox.Maximum = SpeakerEngines.MostSpeakers(SpeakerEngines.Current);

        RefreshControls();

        await _viewModel.PreloadAsync();

        if (_openOnLaunch is not null)
        {
            await OpenPathAsync(_openOnLaunch);
        }
    }

    /// <summary>
    /// The app downloads its own models, saying what it costs before spending it — a few
    /// gigabytes on a first launch, one file when an update moves a model — and every stage
    /// degrades the way the pipeline already does, so
    /// a failed download costs precision or labels, never the app. Model weights only:
    /// nothing else is ever installed.
    /// </summary>
    private async Task ProvisionModelsAsync()
    {
        var modelRoot = FindModelRoot();

        if (!ModelProvisioner.NeedsAnything(modelRoot, whisperCpp: true))
        {
            return;
        }

        var bytes = ModelProvisioner.ApproximateBytesToFetch(modelRoot, whisperCpp: true);
        var cost = bytes >= 1_000_000_000 ? $"{bytes / 1e9:F1} GB" : $"{bytes / 1_000_000} MB";

        StatusText.Text = bytes >= 1_000_000_000
            ? $"First run: downloading the models this app runs on (about {cost}). "
                + "The app is usable the moment they land — nothing else to set up."
            : $"Updating a model this app runs on (about {cost}).";
        WorkProgress.IsVisible = true;

        var provisioner = new ModelProvisioner(diarizationExtractor: new LocalScribe.Doctor.TarBz2Extractor());

        var progress = new Progress<InstallProgress>(update => Post(() =>
        {
            StatusText.Text = update.Message;

            if (update.Fraction is { } fraction)
            {
                WorkProgress.Value = fraction;
            }
        }));

        try
        {
            var ready = await provisioner.EnsureAsync(
                modelRoot, whisperCpp: true, coreMl: true, progress);

            StatusText.Text = ready
                ? "Models ready."
                : "The speech model could not be downloaded — transcription needs it. "
                  + "It will be tried again next launch, or run 'localscribe-doctor --fetch-models'.";
        }
        finally
        {
            WorkProgress.IsVisible = false;
            WorkProgress.Value = 0;
        }
    }

    /// <summary>
    /// The engine the advisor recommends is the engine the window runs — wrapped so that an
    /// engine failure mid-recording restarts it and retries the window it failed on, rather
    /// than costing the transcript. One hiccup is invisible but counted; a window that fails
    /// a fresh engine too still fails honestly.
    /// </summary>
    internal static ITranscriber OpenEngine(ExecutionPlan plan, string? onnxDirectory) =>
        new ResilientTranscriber(() => OpenRawEngine(plan, onnxDirectory));

    /// <summary>whisper.cpp whenever its model is on disk, the ONNX layout otherwise.</summary>
    private static ITranscriber OpenRawEngine(ExecutionPlan plan, string? onnxDirectory)
    {
        var directory = Path.Combine(FindModelRoot(), WhisperCppModelSource.DirectoryName);

        var ggml = Directory.Exists(directory)
            ? Directory.EnumerateFiles(directory, "ggml-*.bin").OrderBy(f => f).FirstOrDefault()
            : null;

        if (ggml is not null)
        {
            return WhisperCpp.WhisperCppTranscriber.Load(ggml, plan);
        }

        if (onnxDirectory is null)
        {
            throw new InvalidOperationException(
                "No transcription model on disk. Run 'localscribe-doctor --fetch-models' first.");
        }

        return WhisperOnnxTranscriber.Load(onnxDirectory, plan);
    }

    /* ---- opening, recording, saving --------------------------------------------------- */

    private async void OnOpenClicked(object? sender, RoutedEventArgs e)
    {
        var patterns = AudioFileLoader.SupportedExtensions.Select(x => "*" + x).ToList();

        var picked = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Open a recording or transcript",
            AllowMultiple = false,
            FileTypeFilter =
            [
                new FilePickerFileType("Recordings and transcripts")
                {
                    Patterns = [.. patterns, "*.scrb", "*.lscribe"],
                },
            ],
        });

        if (picked is [{ } file] && file.TryGetLocalPath() is { } path)
        {
            await OpenPathAsync(path);
        }
    }

    private async Task OpenPathAsync(string path)
    {
        var extension = Path.GetExtension(path);

        if (extension.Equals(".scrb", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".lscribe", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                _viewModel.OpenArchive(path);
            }
            catch (Exception exception)
            {
                StatusText.Text = $"Could not open the archive: {exception.Message}";
            }

            return;
        }

        await _viewModel.TranscribeFileAsync(path);
    }

    private async void OnRecordClicked(object? sender, RoutedEventArgs e)
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

    /// <summary>
    /// Where a pause left off, so play resumes there. Null once playback starts anywhere
    /// else or reaches the end — a finished recording plays again from the top.
    /// </summary>
    private double? _pausedAtSeconds;

    private void OnPlayPauseClicked(object? sender, RoutedEventArgs e)
    {
        if (_viewModel.Player.IsPlaying)
        {
            // Pause is a stop that remembers: the player itself only stops, and the newest
            // marker position is the honest place to resume from.
            _pausedAtSeconds = Volatile.Read(ref _latestPositionSeconds);
            _viewModel.Player.Stop();
        }
        else if (_viewModel.Player.HasAudio)
        {
            var resumeAt = _pausedAtSeconds ?? 0;
            _pausedAtSeconds = null;
            StartPlayback(resumeAt);
        }
    }

    /// <summary>
    /// Whether the transcript follows the marker. Re-engaged by asking to hear something;
    /// surrendered the moment the reader scrolls. Told by their input, not by where the view
    /// ended up: timing windows failed whenever the marker itself was scrolling, and "the
    /// highlight left the viewport" never fired in a paragraph taller than the window — some of
    /// it was always in view, so every scroll was pulled straight back.
    /// </summary>
    private bool _followPlayback = true;

    /// <summary>Hands the view to the reader the moment they scroll it themselves, by any means.</summary>
    private void WatchForReaderScrolling()
    {
        // Tunnel and handled-too: the scroll viewer and the selectable text both claim these.
        TranscriptScroll.AddHandler(
            PointerWheelChangedEvent,
            (_, _) => _followPlayback = false,
            RoutingStrategies.Tunnel,
            handledEventsToo: true);

        TranscriptScroll.AddHandler(
            PointerPressedEvent,
            (_, e) =>
            {
                if (e.Source is Avalonia.Visual source
                    && Avalonia.VisualTree.VisualExtensions.FindAncestorOfType<Avalonia.Controls.Primitives.ScrollBar>(source, includeSelf: true) is not null)
                {
                    _followPlayback = false;
                }
            },
            RoutingStrategies.Tunnel,
            handledEventsToo: true);

        TranscriptScroll.AddHandler(
            KeyDownEvent,
            (_, e) =>
            {
                if (e.Key is Key.PageUp or Key.PageDown or Key.Up or Key.Down or Key.Home or Key.End)
                {
                    _followPlayback = false;
                }
            },
            RoutingStrategies.Tunnel,
            handledEventsToo: true);
    }

    /// <summary>
    /// Keeps the word being heard on screen with a few lines around it, scrolling only when it
    /// is about to leave. The word rather than its paragraph: a paragraph taller than the window
    /// brought into view shows its top, and the voice can be a screen further down.
    /// </summary>
    private void KeepInView(int paragraph, WordRun? word)
    {
        const double context = 75;
        var row = _paragraphBorders[paragraph];

        if (word is not null
            && Avalonia.VisualTree.VisualExtensions.GetVisualDescendants(row).OfType<SelectableTextBlock>().FirstOrDefault() is { } block
            && block.TextLayout.HitTestTextRange(word.CharStart, Math.Max(1, word.CharEnd - word.CharStart)).FirstOrDefault() is var line
            && line.Height > 0
            && block.TranslatePoint(line.TopLeft, TranscriptScroll) is { } top)
        {
            if (top.Y >= context && top.Y + line.Height <= TranscriptScroll.Bounds.Height - context)
            {
                return;
            }

            block.BringIntoView(line.Inflate(new Thickness(0, context)));
            return;
        }

        if (!IsVisibleInTranscript(row))
        {
            row.BringIntoView();
        }
    }
    private bool _scrubWasPlaying;

    private void StartPlayback(double seconds)
    {
        _followPlayback = true;
        _pausedAtSeconds = null;
        _viewModel.Player.PlayFrom(seconds);

        // The button reflects the player, not the marker: gating the pause glyph on a
        // highlighted paragraph left it reading "play" whenever sound came from the untimed
        // tail, where no paragraph is highlighted at all.
        if (_viewModel.Player.IsPlaying)
        {
            PlayPauseIcon.Data = (Geometry)this.FindResource("IconPause")!;
        }
    }

    private async void OnSaveClicked(object? sender, RoutedEventArgs e) => await SaveAsync();

    private async Task<bool> SaveAsync()
    {
        if (!_viewModel.CanSaveArchive)
        {
            return false;
        }

        var picked = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save transcript and audio",
            SuggestedFileName = _viewModel.SourceName + ".scrb",
            FileTypeChoices =
            [
                new FilePickerFileType("LocalScribe transcript") { Patterns = ["*.scrb"] },
            ],
        });

        if (picked?.TryGetLocalPath() is not { } path)
        {
            return false;
        }

        try
        {
            _viewModel.SaveArchive(path);
            StatusText.Text = $"Saved to {Path.GetFileName(path)}.";
            return true;
        }
        catch (Exception exception)
        {
            StatusText.Text = $"Could not save: {exception.Message}";
            return false;
        }
    }

    private void OnDiscardClicked(object? sender, RoutedEventArgs e) => _viewModel.Discard();

    /// <summary>
    /// Exports the words alone, in a format chosen by the extension the user picks. The
    /// archive stays the first-class save — text loses the half that makes the app useful —
    /// which is why this lives behind its own button rather than inside Save.
    /// </summary>
    private async void OnExportClicked(object? sender, RoutedEventArgs e)
    {
        if (!_viewModel.HasTranscript)
        {
            return;
        }

        var picked = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Export transcript text",
            SuggestedFileName = _viewModel.SourceName + ".txt",
            FileTypeChoices =
            [
                new FilePickerFileType("Plain text") { Patterns = ["*.txt"] },
                new FilePickerFileType("Markdown") { Patterns = ["*.md"] },
                new FilePickerFileType("SubRip subtitles") { Patterns = ["*.srt"] },
            ],
        });

        if (picked?.TryGetLocalPath() is not { } path)
        {
            return;
        }

        var format = Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".md" => TranscriptFormat.Markdown,
            ".srt" => TranscriptFormat.SubRip,
            _ => TranscriptFormat.PlainText,
        };

        try
        {
            await File.WriteAllTextAsync(path, _viewModel.Export(format));
            StatusText.Text = $"Exported to {Path.GetFileName(path)}.";
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            StatusText.Text = $"Could not export: {exception.Message}";
        }
    }

    /// <summary>
    /// Runs the speakers again with a count from the stepper. The count is the one thing a
    /// person always knows and the algorithm never does — which is why it gets a control
    /// built for small honest numbers rather than a text box.
    /// </summary>
    private async void OnSpeakersCountClicked(object? sender, RoutedEventArgs e)
    {
        SpeakersButton.Flyout?.Hide();
        await _viewModel.FindSpeakersAsync((int?)SpeakerCountBox.Value);
    }

    private async void OnSpeakersAutoClicked(object? sender, RoutedEventArgs e)
    {
        SpeakersButton.Flyout?.Hide();
        await _viewModel.FindSpeakersAsync(null);
    }

    /// <summary>
    /// The pace is the planner's budget with the user's hand on it. The default stop is the
    /// planner's own answer, unchanged — the invariant — and the detail line names exactly
    /// what each stop spends on this machine, so the choice is informed rather than vibes.
    /// </summary>
    private void OnPaceChosen(object? sender, RoutedEventArgs e)
    {
        if (sender is RadioButton { Tag: string name })
        {
            _viewModel.Pace = WorkPaces.Parse(name);
            PaceDetail.Text = _viewModel.PaceDescription;
        }
    }

    private void SyncPaceControls()
    {
        // Flyout content can be wired lazily; a launch must never die over a pace label.
        if (PaceLight is null || PaceDetail is null)
        {
            return;
        }

        var pace = _viewModel.Pace;
        PaceLight.IsChecked = pace == WorkPace.Light;
        PaceBalanced.IsChecked = pace == WorkPace.Balanced;
        PaceFast.IsChecked = pace == WorkPace.Fast;
        PaceFastest.IsChecked = pace == WorkPace.Full;
        PaceDetail.Text = _viewModel.PaceDescription;
    }

    /* ---- the close gate ---------------------------------------------------------------- */

    /// <summary>
    /// Refuses the close over unsaved work and asks. A cancelled save picker is not consent to
    /// lose anything: it returns to the window with the work intact, exactly as on Windows.
    /// </summary>
    private async void OnClosing(object? sender, WindowClosingEventArgs e)
    {
        if (_closeApproved || !_viewModel.HasUnsavedWork)
        {
            _viewModel.Dispose();
            return;
        }

        e.Cancel = true;

        var choice = await CloseGateDialog.AskAsync(this);

        switch (choice)
        {
            case CloseGateDialog.Choice.Save:
                if (await SaveAsync())
                {
                    _closeApproved = true;
                    Close();
                }

                break;

            case CloseGateDialog.Choice.Discard:
                _closeApproved = true;
                Close();
                break;

            case CloseGateDialog.Choice.Cancel:
                break;
        }
    }

    /* ---- view-model plumbing ----------------------------------------------------------- */

    /// <summary>
    /// Marshals a view-model notification onto the UI thread. Some arrive already there (file
    /// transcription runs on the UI context); the microphone's arrive from the audio thread.
    /// </summary>
    private static void Post(Action action)
    {
        if (Dispatcher.UIThread.CheckAccess())
        {
            action();
        }
        else
        {
            Dispatcher.UIThread.Post(action);
        }
    }

    private void OnViewModelChanged(object? sender, PropertyChangedEventArgs e) => Post(() =>
    {
        switch (e.PropertyName)
        {
            case nameof(MainViewModel.Status):
                StatusText.Text = _viewModel.Status;

                // The ellipsis keeps layouts honest; the tooltip keeps the words available.
                ToolTip.SetTip(StatusText, _viewModel.Status);
                break;

            case nameof(MainViewModel.HardwareSummary):
                ShowModels(_viewModel.HardwareSummary);
                break;

            case nameof(MainViewModel.Progress):
                // A run shows its steps instead; this bar is for the single jobs, such as
                // comparing voices, that have only one number to give.
                WorkProgress.Value = _viewModel.Progress;
                WorkProgress.IsVisible = _viewModel.Progress is > 0 and < 1 && _viewModel.ProgressSteps.Count == 0;
                break;

            case nameof(MainViewModel.Paragraphs):
                // While a transcription streams, each window replaces the whole transcript —
                // and the full rebuild prices every paragraph's words against the audio, which
                // grows quadratic over a long file and is what made the app feel like the hog
                // it was announcing. Streaming gets cheap plain-text cards; the full word-run
                // rebuild happens once, when the work settles.
                if (_viewModel.IsBusy)
                {
                    ScheduleStreamingRebuild();
                }
                else
                {
                    RebuildParagraphs();
                }

                break;

            case nameof(MainViewModel.IsBusy) when !_viewModel.IsBusy && _lightParagraphs:
                RebuildParagraphs();
                break;

            case nameof(MainViewModel.ProvisionalText):
                ProvisionalTextBlock.Text = _viewModel.ProvisionalText;
                break;

            case nameof(MainViewModel.SpeakerCount):
                // The stepper opens showing the current answer, so "one more than it found"
                // is a single click.
                if (_viewModel.SpeakerCount > 0)
                {
                    SpeakerCountBox.Value = _viewModel.SpeakerCount;
                }

                break;
        }

        RefreshControls();
    });

    private void RefreshControls()
    {
        // The glow means "models are thinking", not "a file is being read": decoding the
        // audio is disk work, and the player receiving the samples is the moment it ends —
        // model stages start immediately after. Preparing to record is model loading too.
        UpdateGlow((_viewModel.IsBusy && _viewModel.Player.HasAudio) || _viewModel.IsPreparing);
        OpenButton.IsEnabled = !_viewModel.IsBusy && !_viewModel.IsRecording;
        RecordButton.IsEnabled = !_viewModel.IsBusy && !_viewModel.IsPreparing;
        RecordLabel.Text = _viewModel.IsRecording ? "Stop" : "Record";
        RecordIcon.Data = (Geometry)this.FindResource(_viewModel.IsRecording ? "IconStop" : "IconMic")!;
        RecordingDot.IsVisible = _viewModel.IsRecording;
        PlayPauseButton.IsEnabled = _viewModel.Player.HasAudio && !_viewModel.IsRecording;
        SaveButton.IsEnabled = _viewModel.CanSaveArchive;
        ExportButton.IsEnabled = _viewModel.HasTranscript;
        SpeakersButton.IsEnabled = _viewModel.CanFindSpeakers && !_viewModel.IsBusy;
        DiscardButton.IsEnabled = _viewModel.HasTranscript;

        Reveal(TransportRow, _viewModel.Player.HasAudio);
        Reveal(EmptyState, !_viewModel.HasTranscript && !_viewModel.IsBusy && !_viewModel.IsRecording
            && !_viewModel.IsPreparing && string.IsNullOrEmpty(_viewModel.ProvisionalText));

        if (_viewModel.ActiveModels != _shownActive)
        {
            RenderModels();
        }

        // Only when a different recording arrives: this runs on every change the view model
        // reports, progress ticks included, and each call threw the waveform's peaks away.
        var duration = _viewModel.Player.HasAudio ? _viewModel.Player.DurationSeconds : 0;

        if (duration != _shownDuration)
        {
            _shownDuration = duration;
            _waveform.SetAudio(duration);
            TimeReadout.Text = $"0:00 / {Clock(duration)}";
        }

        UpdateTitle();
        SyncMenu();
        ShowSteps();
    }

    /// <summary>Draws the run's steps, or takes them away when no run is under way.</summary>
    private void ShowSteps()
    {
        var steps = _viewModel.ProgressSteps;
        Reveal(StepsPanel, steps.Count > 0);

        if (steps.Count == 0)
        {
            return;
        }

        (TextBlock Label, TextBlock State, ProgressBar Bar)[] rows =
        [
            (StepLabel0, StepState0, StepBar0),
            (StepLabel1, StepState1, StepBar1),
            (StepLabel2, StepState2, StepBar2),
        ];

        for (var i = 0; i < rows.Length; i++)
        {
            var (label, state, bar) = rows[i];
            var step = steps[i];

            label.Text = step.Label;
            ToolTip.SetTip(label, step.Label);
            label.Opacity = step.State == MainViewModel.ProgressStepState.Waiting ? 0.5 : 1;
            bar.Value = step.State == MainViewModel.ProgressStepState.Done ? 1 : step.Fraction;
            state.Text = step.State switch
            {
                MainViewModel.ProgressStepState.Done when step.TookSeconds is { } took => $"✓ {RunTimes.Clock(took)}",
                MainViewModel.ProgressStepState.Done => "✓",
                MainViewModel.ProgressStepState.Working => $"{(int)(step.Fraction * 100)}%",
                _ => string.Empty,
            };
        }

        ProgressHintText.Text = _viewModel.ProgressHint;
        ProgressHintText.IsVisible = _viewModel.ProgressHint is not null;
    }

    private void SetUpAlert()
    {
        AlertCheck.IsVisible = OperatingSystem.IsMacOS();
        AlertCheck.IsChecked = FinishedAlert.Wanted;

        if (FinishedAlert.Wanted)
        {
            // Already asked, since the box is ticked, so this only finds out the answer.
            FinishedAlert.AskPermission(_ => { });
        }
    }

    private void OnAlertToggled(object? sender, RoutedEventArgs e)
    {
        var wanted = AlertCheck.IsChecked == true;
        FinishedAlert.Wanted = wanted;

        if (wanted)
        {
            FinishedAlert.AskPermission(why =>
            {
                if (why is not null)
                {
                    StatusText.Text = $"You'll get a sound and a Dock bounce instead of a notification. {why} "
                        + "You can change this in System Settings, under Notifications.";
                }
            });
        }
    }

    /// <summary>
    /// Says the run has ended, if that was asked for and nobody is looking at the window: the
    /// words on screen already say so to anyone who is.
    /// </summary>
    private void OnRunEnded(MainViewModel.RunOutcome outcome) => Post(() =>
    {
        if (AlertCheck.IsChecked == true && !IsActive)
        {
            FinishedAlert.Show(
                outcome.Succeeded ? $"{outcome.SourceName} is ready" : $"{outcome.SourceName} stopped",
                outcome.Message);
        }
    });

    private double _shownDuration = -1;

    /// <summary>
    /// The document's name in the title bar, with what a reader wants to know before opening
    /// it: how long it is and how many people are in it. The window title carries the name
    /// too, for the Window menu and Mission Control.
    /// </summary>
    private void UpdateTitle()
    {
        var named = _viewModel.HasTranscript || _viewModel.Player.HasAudio || _viewModel.IsRecording;
        var name = named ? _viewModel.SourceName : "LocalScribe";

        TitleText.Text = name;
        Title = named ? $"{name} — LocalScribe" : "LocalScribe";

        var facts = new List<string>();

        if (_viewModel.Player.HasAudio)
        {
            facts.Add(Clock(_viewModel.Player.DurationSeconds));
        }

        if (_viewModel.SpeakerCount > 0)
        {
            facts.Add(_viewModel.SpeakerCount == 1 ? "1 speaker" : $"{_viewModel.SpeakerCount} speakers");
        }

        SubtitleText.Text = string.Join(" · ", facts);
        SubtitleText.IsVisible = named && facts.Count > 0;
    }

    /* ---- the models chip ------------------------------------------------------------- */

    /// <summary>
    /// Turns the view model's one-line roster into a chip and a panel. The roster is one string
    /// in the shared view model — the WinUI window shows it whole — so it is taken apart here
    /// by its separator rather than changing what Windows reads.
    /// </summary>
    private string[] _rosterParts = [];
    private MainViewModel.ModelRoles _shownActive = MainViewModel.ModelRoles.None;

    private static readonly IBrush WorkingBrush = new SolidColorBrush(Color.FromRgb(168, 85, 247));

    private void ShowModels(string roster)
    {
        var parts = roster.Split(" · ", StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();

        // The aligner is not in the shared roster — the WinUI window prints that line whole,
        // and Windows has not been asked to make room for it — but it is the stage that runs
        // longest, so a chip that lights up what is working cannot leave it out.
        if (parts.Count > 0 && AlignerPart() is { } aligner)
        {
            parts.Insert(1, aligner);
        }

        _rosterParts = [.. parts];
        RenderModels();
    }

    private static string? AlignerPart()
    {
        if (ForcedAligner.Find(FindModelRoot()) is not { } directory)
        {
            return null;
        }

        if (GpuAligner.Applies && GpuAligner.IsReady(directory))
        {
            return "MMS word aligner, fp16 on the GPU";
        }

        var quantised = File.Exists(Path.Combine(directory, AlignmentModelSource.QuantisedModelFileName))
            && AlignmentModelSource.PreferQuantised;
        return quantised ? "MMS word aligner, 4-bit" : "MMS word aligner, fp16";
    }

    /// <summary>
    /// Draws the chip and the panel, lighting whichever models are running inference now.
    /// Rebuilt only at stage boundaries — a handful of times a run — so building it whole is
    /// simpler than keeping controls to toggle.
    /// </summary>
    private void RenderModels()
    {
        var active = _viewModel.ActiveModels;
        _shownActive = active;

        if (_rosterParts.Length == 0)
        {
            ModelsChip.IsVisible = false;
            return;
        }

        ModelsChip.IsVisible = true;

        // The chip sits faded at rest so it never competes with the status line; while a
        // model works it comes up to full strength, which is the point of looking at it.
        ModelsChip.Opacity = active == MainViewModel.ModelRoles.None ? 0.7 : 1;
        ModelsChipRow.Children.Clear();

        for (var i = 0; i < _rosterParts.Length; i++)
        {
            var part = _rosterParts[i];
            var working = (RoleOf(part) & active) != 0;

            if (i > 0)
            {
                ModelsChipRow.Children.Add(new TextBlock { Text = " · ", Opacity = 0.6 });
            }

            if (working)
            {
                ModelsChipRow.Children.Add(new Avalonia.Controls.Shapes.Ellipse
                {
                    Width = 6,
                    Height = 6,
                    Fill = WorkingBrush,
                    Margin = new Thickness(0, 0, 4, 0),
                    VerticalAlignment = VerticalAlignment.Center,
                    Classes = { "pulse" },
                });
            }

            ModelsChipRow.Children.Add(Tinted(new TextBlock
            {
                Text = Nickname(part),
                FontWeight = working ? FontWeight.SemiBold : FontWeight.Normal,
                Opacity = working || active == MainViewModel.ModelRoles.None ? 1 : 0.6,
            }, working));
        }

        ToolTip.SetTip(ModelsChip, active == MainViewModel.ModelRoles.None
            ? "The models doing the work, and where each runs"
            : "Lit: the models working right now. Click for detail.");

        ModelsPanel.Children.Clear();
        ModelsPanel.Children.Add(new TextBlock
        {
            Text = "Models in use",
            FontWeight = FontWeight.SemiBold,
            FontSize = 13,
        });

        foreach (var part in _rosterParts)
        {
            var (role, detail) = Describe(part);
            var working = (RoleOf(part) & active) != 0;

            ModelsPanel.Children.Add(new StackPanel
            {
                Spacing = 1,
                Children =
                {
                    Tinted(new TextBlock
                    {
                        Text = working ? $"{role} — working now" : role,
                        FontSize = 11.5,
                        Opacity = working ? 1 : 0.6,
                        FontWeight = working ? FontWeight.SemiBold : FontWeight.Normal,
                    }, working),
                    new TextBlock { Text = detail, FontSize = 12.5, TextWrapping = TextWrapping.Wrap },
                },
            });
        }
    }

    /// <summary>
    /// Lights a working model's name, and otherwise leaves the colour alone. Setting it to
    /// null is not "default": a local null outranks the inherited text colour, and every idle
    /// name rendered in no colour at all — the chip showed only its separators.
    /// </summary>
    private static TextBlock Tinted(TextBlock text, bool working)
    {
        if (working)
        {
            text.Foreground = WorkingBrush;
        }

        return text;
    }

    private static MainViewModel.ModelRoles RoleOf(string part)
    {
        var p = part.ToLowerInvariant();

        if (p.Contains("aligner"))
        {
            return MainViewModel.ModelRoles.Words;
        }

        if (p.Contains("whisper") || p.StartsWith("encoder", StringComparison.Ordinal))
        {
            return MainViewModel.ModelRoles.Transcription;
        }

        if (p.Contains("speakers"))
        {
            return MainViewModel.ModelRoles.Speakers;
        }

        return MainViewModel.ModelRoles.None;
    }

    /// <summary>A word or two per model, for the chip.</summary>
    private static string Nickname(string part)
    {
        var p = part.ToLowerInvariant();

        if (p.Contains("aligner"))
        {
            return "MMS aligner";
        }

        if (p.Contains("whisper"))
        {
            return p.Contains("turbo") ? "Whisper turbo" : "Whisper";
        }

        if (p.Contains("nemotron"))
        {
            return "Nemotron";
        }

        if (p.Contains("pyannote"))
        {
            return "pyannote";
        }

        // Unknown shapes (the ONNX encoder/decoder line, a new model) fall back to their
        // first few words rather than to nothing.
        return string.Join(' ', part.Split(' ').Take(3));
    }

    /// <summary>The role and the full description, for the panel.</summary>
    private static (string Role, string Detail) Describe(string part)
    {
        var p = part.ToLowerInvariant();

        if (p.Contains("aligner"))
        {
            return p.Contains("on the gpu")
                ? ("Word timing", part.Replace(" on the GPU", string.Empty, StringComparison.Ordinal)
                    + $" — on the GPU, through Core ML; places each word against the audio. "
                    + $"Recordings under {GpuAligner.ShortestSeconds:F0} seconds are timed on the CPU")
                : ("Word timing", part + " — on the CPU; places each word against the audio");
        }

        if (p.Contains("whisper") || p.StartsWith("encoder", StringComparison.Ordinal))
        {
            var where = p.Contains("coreml") ? " — encoder on the Neural Engine" : string.Empty;
            return ("Transcription", part + where);
        }

        if (p.Contains("speakers"))
        {
            return ("Who spoke", part.Replace(" speakers", string.Empty, StringComparison.Ordinal) + " — on the CPU");
        }

        return ("Model", part);
    }

    /* ---- the working glow -------------------------------------------------------------- */

    /// <summary>
    /// The sweep itself: one conic gradient, rotated a few degrees per frame while any model
    /// stage runs. Driven by a timer rather than a style animation because the angle lives on
    /// the brush, not the control — and stopped when idle, so a resting window costs nothing.
    /// </summary>
    private readonly ConicGradientBrush _glowBrush = new()
    {
        GradientStops =
        {
            new GradientStop(Color.FromArgb(160, 244, 63, 94), 0.00),
            new GradientStop(Color.FromArgb(160, 168, 85, 247), 0.25),
            new GradientStop(Color.FromArgb(160, 59, 130, 246), 0.50),
            new GradientStop(Color.FromArgb(160, 236, 72, 153), 0.75),
            new GradientStop(Color.FromArgb(160, 244, 63, 94), 1.00),
        },
    };

    private DispatcherTimer? _glowTimer;

    private void UpdateGlow(bool working)
    {
        // The glow and the sleep assertion are the same statement in two audiences: something
        // is working, so the person sees a sweep and the machine holds off idle sleep.
        SleepAssertion.While(working || _viewModel.IsRecording);

        if (working && _glowTimer is null)
        {
            // One brush feeds both rings, so the halo's bloom and the crisp edge sweep as a
            // single light source rather than two rotating separately.
            AiGlow.BorderBrush = _glowBrush;
            AiGlowHalo.BorderBrush = _glowBrush;
            AiGlowHalo.Classes.Add("breathing");
            AiGlowLayer.Opacity = 1;

            // Fifteen frames a second, because each step repaints a blurred ring: at thirty
            // the ornament was measurably competing with the work it announced.
            _glowTimer = new DispatcherTimer(
                TimeSpan.FromMilliseconds(66),
                DispatcherPriority.Render,
                (_, _) => _glowBrush.Angle = (_glowBrush.Angle + 5) % 360);
            _glowTimer.Start();
        }
        else if (!working && _glowTimer is not null)
        {
            _glowTimer.Stop();
            _glowTimer = null;
            AiGlowHalo.Classes.Remove("breathing");
            AiGlowLayer.Opacity = 0;
        }
    }

    /// <summary>
    /// Shows or hides a panel with a fade instead of a pop. IsVisible cannot animate, so the
    /// fade rides on Opacity: shown at zero, then eased to one a frame later — the "reveal"
    /// style carries the transition.
    /// </summary>
    private static void Reveal(Control control, bool visible)
    {
        if (control.IsVisible == visible)
        {
            return;
        }

        control.Classes.Add("reveal");
        control.IsVisible = visible;

        if (visible)
        {
            control.Opacity = 0;
            Dispatcher.UIThread.Post(() => control.Opacity = 1, DispatcherPriority.Background);
        }
    }

    /* ---- the transcript and the marker ------------------------------------------------- */

    private static readonly IBrush MarkerBrush = new SolidColorBrush(Color.FromArgb(48, 64, 128, 255));
    private static readonly IBrush WordBrush = new SolidColorBrush(Color.FromArgb(150, 64, 128, 255));

    /// <summary>Per paragraph: each word's run, its time, and where it starts in the display text.</summary>
    private sealed record WordRun(Run Run, WordTimings.Word Word, int CharStart, int CharEnd);

    private readonly List<IReadOnlyList<WordRun>> _paragraphWordRuns = [];

    /// <summary>True while the panel holds the streaming shape rather than the final build.</summary>
    private bool _lightParagraphs;

    private bool _streamRebuildQueued;

    /// <summary>Whether the last streaming build rendered any clickable rows yet.</summary>
    private bool _streamHasTimedRows;

    /// <summary>
    /// Coalesces streaming rebuilds, latest-wins: streamed windows and head-timing passes both
    /// replace the transcript every few seconds, and rebuilding per update was the hog. One
    /// rebuild per interval reads identically and the final build lands when the run ends.
    /// </summary>
    /// <summary>Whether this run already spent its one unthrottled rebuild.</summary>
    private bool _streamPrimeAttempted;

    private void ScheduleStreamingRebuild()
    {
        // The moment the first words graduate to measured, render now: the status bar
        // announces "clickable through…" the instant the view model times them, and a
        // coalescing delay right then is a status line telling a small lie. ONE attempt,
        // whatever it finds: the measured table is keyed by segment value, and when its
        // records do not match the rows on screen this condition stays true forever — as a
        // standing bypass it rebuilt a 950-row transcript on every streamed update, and the
        // window spent the whole transcription painting instead of showing words.
        if (!_streamPrimeAttempted && !_streamHasTimedRows && _viewModel.HasMeasuredWords)
        {
            _streamPrimeAttempted = true;
            RebuildParagraphsStreaming();
            return;
        }

        if (_streamRebuildQueued)
        {
            return;
        }

        _streamRebuildQueued = true;

        DispatcherTimer.RunOnce(() =>
        {
            _streamRebuildQueued = false;

            if (_speakerMenuOpen)
            {
                // The menu's Closed handler schedules the rebuild this skipped.
                return;
            }

            if (_viewModel.IsBusy)
            {
                RebuildParagraphsStreaming();
            }
        }, TimeSpan.FromMilliseconds(400));
    }

    /// <summary>
    /// The streaming shape of the transcript, matching the Windows window: the timed head is
    /// full clickable word runs — its words carry measured times already, so building them
    /// prices nothing against the audio — and the untimed tail is quiet plain text, offered
    /// no click on purpose: a click on unmeasured stamps lands unpredictably, which is worse
    /// than a click that plainly is not on offer yet.
    /// </summary>
    private void RebuildParagraphsStreaming()
    {
        var offset = TranscriptScroll.Offset;

        _lightParagraphs = true;
        _shownParagraphs = _viewModel.Paragraphs;
        _paragraphBorders.Clear();
        _paragraphWordRuns.Clear();
        _highlightedRuns.Clear();
        _searchRuns.Clear();
        _searchMatches.Clear();
        ParagraphsPanel.Children.Clear();
        _speakerOrder.Clear();

        _streamHasTimedRows = false;

        foreach (var paragraph in _shownParagraphs)
        {
            // Any timed segment makes the row worth the full build: its timed stretch is
            // clickable now, its grey rest fills in as timing reaches it. All-segments was
            // the first gate here, and one long opening paragraph held the whole head
            // hostage — "clickable through 0:58" on screen, nothing clickable under it.
            var timed = paragraph.Segments.Any(_viewModel.IsTimed);

            _streamHasTimedRows |= timed;

            if (timed)
            {
                // Renameable the moment it appears: the view model keeps the name against
                // the voice, so the windows still to come arrive already carrying it.
                var border = ParagraphRow(paragraph, BuildWordBlock(paragraph, streaming: true));
                _paragraphBorders.Add(border);
                ParagraphsPanel.Children.Add(border);
            }
            else
            {
                // Indexes stay aligned with the timed rows so the marker and search maths
                // hold; the empty run list makes this row invisible to both.
                _paragraphWordRuns.Add([]);
                var plain = new SelectableTextBlock
                {
                    Text = paragraph.Text,
                    TextWrapping = TextWrapping.Wrap,
                    FontSize = 15,
                    LineHeight = 25,
                    Opacity = 0.7,
                };

                OfferCorrections(plain, paragraph, []);

                var placeholder = ParagraphRow(paragraph, plain);
                placeholder.Classes.Remove("paragraph");

                _paragraphBorders.Add(placeholder);
                ParagraphsPanel.Children.Add(placeholder);
            }
        }

        // The reader's place survives the swap; following the tail is the marker's job, and
        // only for a reader who was already there.
        TranscriptScroll.Offset = offset;
    }

    private void RebuildParagraphs()
    {
        _lightParagraphs = false;
        _streamPrimeAttempted = false;
        _shownParagraphs = _viewModel.Paragraphs;
        _paragraphBorders.Clear();
        _paragraphWordRuns.Clear();
        _highlightedRuns.Clear();
        ParagraphsPanel.Children.Clear();
        _speakerOrder.Clear();

        foreach (var paragraph in _shownParagraphs)
        {
            var border = ParagraphRow(paragraph, BuildWordBlock(paragraph));
            _paragraphBorders.Add(border);
            ParagraphsPanel.Children.Add(border);
        }
    }

    /// <summary>
    /// One paragraph: who spoke and when on a quiet line above, the words below. The same shape
    /// streaming and finished, so a transcript does not change its look the moment it settles.
    /// </summary>
    private Border ParagraphRow(TranscriptParagraph paragraph, Control body)
    {
        var lines = new StackPanel { Spacing = 3 };
        var header = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };

        if (paragraph.Speaker is { Length: > 0 } speaker)
        {
            header.Children.Add(SpeakerLabel(paragraph, speaker));
        }

        header.Children.Add(Secondary(new TextBlock
        {
            Text = Clock(paragraph.StartSeconds),
            FontSize = 11.5,
            FontFamily = new FontFamily("Menlo, monospace"),
            VerticalAlignment = VerticalAlignment.Center,
        }));

        if (paragraph.Overlapped)
        {
            header.Children.Add(Secondary(new TextBlock
            {
                Text = "crosstalk",
                FontSize = 11.5,
                FontStyle = FontStyle.Italic,
                VerticalAlignment = VerticalAlignment.Center,
            }));
        }

        lines.Children.Add(header);
        lines.Children.Add(body);

        return new Border
        {
            Child = lines,
            Padding = new Avalonia.Thickness(14, 9),
            CornerRadius = new Avalonia.CornerRadius(10),
            HorizontalAlignment = HorizontalAlignment.Stretch,
            Classes = { "paragraph" },
        };
    }

    /// <summary>
    /// The speaker's name, in that speaker's colour, and the way to rename them.
    /// <para>
    /// Renaming hangs off the label because the label is the thing that is wrong. All three
    /// verbs are needed: "this part" for a paragraph misfiled, "everywhere" for a person split
    /// across labels, "by voice" for one label meaning two people. A plain click opens them as
    /// well as a right-click: a name that looks like a name gives no hint that it hides a menu.
    /// </para>
    /// </summary>
    private TextBlock SpeakerLabel(TranscriptParagraph paragraph, string speaker)
    {
        var label = new TextBlock
        {
            Text = speaker,
            FontWeight = FontWeight.SemiBold,
            FontSize = 12.5,
            Foreground = SpeakerBrush(speaker),
            Cursor = new Cursor(StandardCursorType.Hand),
            VerticalAlignment = VerticalAlignment.Center,
            ContextMenu = BuildSpeakerMenu(paragraph),
        };

        ToolTip.SetTip(label, "Rename this speaker");
        label.PointerEntered += (_, _) => label.TextDecorations = TextDecorations.Underline;
        label.PointerExited += (_, _) => label.TextDecorations = null;
        label.PointerPressed += (_, e) =>
        {
            if (e.GetCurrentPoint(label).Properties.IsLeftButtonPressed)
            {
                label.ContextMenu?.Open(label);
                e.Handled = true;
            }
        };

        return label;
    }

    /// <summary>
    /// Quiet text, bound to the palette rather than read from it once, so it follows a change
    /// of appearance and is never left holding a colour that was not found.
    /// </summary>
    private static TextBlock Secondary(TextBlock text)
    {
        text[!TextBlock.ForegroundProperty] = new Avalonia.Markup.Xaml.MarkupExtensions.DynamicResourceExtension("SecondaryText");
        return text;
    }

    /// <summary>Speakers in the order they first speak, for their colours.</summary>
    private readonly Dictionary<string, int> _speakerOrder = new(StringComparer.Ordinal);

    // Eight hues, one per voice Nemotron can tell apart, chosen to stay distinct from each
    // other and readable on both backgrounds; dark mode takes the lighter set.
    private static readonly Color[] LightSpeakerColors =
    [
        Color.Parse("#2563EB"), Color.Parse("#C2410C"), Color.Parse("#047857"), Color.Parse("#7C3AED"),
        Color.Parse("#BE185D"), Color.Parse("#A16207"), Color.Parse("#0E7490"), Color.Parse("#4D7C0F"),
    ];

    private static readonly Color[] DarkSpeakerColors =
    [
        Color.Parse("#7AA7FF"), Color.Parse("#FF9466"), Color.Parse("#4FD1A5"), Color.Parse("#B99AFB"),
        Color.Parse("#F57CB8"), Color.Parse("#F2BE4A"), Color.Parse("#5FCDE4"), Color.Parse("#A9CF4B"),
    ];

    private IBrush SpeakerBrush(string speaker)
    {
        if (!_speakerOrder.TryGetValue(speaker, out var index))
        {
            index = _speakerOrder.Count;
            _speakerOrder[speaker] = index;
        }

        var palette = ActualThemeVariant == Avalonia.Styling.ThemeVariant.Dark ? DarkSpeakerColors : LightSpeakerColors;
        return new SolidColorBrush(palette[index % palette.Length]);
    }

    private ContextMenu BuildSpeakerMenu(TranscriptParagraph paragraph)
    {
        var thisPart = new MenuItem { Header = "Rename this part…" };
        var everywhere = new MenuItem { Header = $"Rename {paragraph.Speaker} everywhere…" };
        var byVoice = new MenuItem { Header = "Rename here and wherever this voice speaks…" };

        thisPart.Click += async (_, _) =>
        {
            if (await AskForName(paragraph.Speaker) is { } name)
            {
                _viewModel.RenameSpeaker(
                    paragraph.StartSeconds, paragraph.EndSeconds, paragraph.Speaker, name,
                    everywhere: false);
            }
        };

        everywhere.Click += async (_, _) =>
        {
            if (await AskForName(paragraph.Speaker) is { } name)
            {
                _viewModel.RenameSpeaker(
                    paragraph.StartSeconds, paragraph.EndSeconds, paragraph.Speaker, name,
                    everywhere: true);
            }
        };

        byVoice.Click += async (_, _) =>
        {
            if (await AskForName(paragraph.Speaker) is { } name)
            {
                await _viewModel.RenameSpeakerByVoiceAsync(
                    paragraph.StartSeconds, paragraph.EndSeconds, paragraph.Speaker, name);
            }
        };

        // By voice compares this paragraph against the finished transcript's others, which
        // mid-run do not exist yet; the two renames that are kept against voice and time work
        // from the first label on.
        var menu = _viewModel.IsBusy
            ? new ContextMenu { Items = { thisPart, everywhere } }
            : new ContextMenu { Items = { thisPart, everywhere, byVoice } };

        HoldRebuildsWhileOpen(menu);
        return menu;
    }

    /// <summary>
    /// A streamed update rebuilds every row, and any menu with it — closing it under the
    /// pointer. Rebuilds wait while one is open; see ScheduleStreamingRebuild.
    /// </summary>
    private void HoldRebuildsWhileOpen(ContextMenu menu)
    {
        menu.Opened += (_, _) => _speakerMenuOpen = true;
        menu.Closed += (_, _) =>
        {
            _speakerMenuOpen = false;

            if (_viewModel.IsBusy)
            {
                ScheduleStreamingRebuild();
            }
        };
    }

    private bool _speakerMenuOpen;

    private async Task<string?> AskForName(string? current)
    {
        var name = await InputDialog.AskAsync(
            this, "Rename speaker", "Who is this?", current ?? string.Empty);

        return string.IsNullOrWhiteSpace(name) ? null : name.Trim();
    }

    private static readonly IBrush UntimedBrush = new SolidColorBrush(Color.FromArgb(130, 128, 128, 128));

    /// <summary>
    /// A paragraph as one run per word, so a word can be lit and a click can be mapped back
    /// to a time — at segment grain: while a run streams, a paragraph can be half measured,
    /// and its timed segments are clickable now while the untimed rest waits in grey. That is
    /// the Windows behaviour ("the grey lines follow as timing reaches them"), and it is what
    /// keeps a long opening paragraph from holding the whole head hostage. The display text
    /// is built from the words themselves; it is the words that carry the times.
    /// </summary>
    private Control BuildWordBlock(TranscriptParagraph paragraph, bool streaming = false)
    {
        var block = new SelectableTextBlock
        {
            TextWrapping = TextWrapping.Wrap,
            FontSize = 15,
            LineHeight = 25,
        };

        TrackSelection(block);

        var runs = new List<WordRun>();
        var at = 0;

        foreach (var segment in paragraph.Segments)
        {
            var timed = !streaming || _viewModel.IsTimed(segment);

            if (!timed)
            {
                // Untimed while streaming: shown, grey, inert — and never priced against the
                // audio, which is what made streaming rebuilds quadratic once before.
                if (at > 0)
                {
                    block.Inlines!.Add(new Run(" "));
                    at++;
                }

                var grey = new Run(segment.Text.Trim()) { Foreground = UntimedBrush };
                block.Inlines!.Add(grey);
                at += grey.Text!.Length;
                continue;
            }

            foreach (var word in _viewModel.WordsIn([segment]))
            {
                if (at > 0)
                {
                    block.Inlines!.Add(new Run(" "));
                    at++;
                }

                var run = new Run(word.Text);
                block.Inlines!.Add(run);
                runs.Add(new WordRun(run, word, at, at + word.Text.Length));
                at += word.Text.Length;
            }
        }

        if (runs.Count == 0 && !streaming)
        {
            // Nothing timed at all — a transcript with no audio behind it. Plain text, and a
            // click starts at the paragraph, which is the best time anybody has.
            block.Inlines!.Clear();
            block.Text = paragraph.Text;
            var start = paragraph.StartSeconds;
            OnClick(block, _ => StartPlayback(start));
            OfferCorrections(block, paragraph, []);
            _paragraphWordRuns.Add([]);
            return block;
        }

        _paragraphWordRuns.Add(runs);

        // Click a word to hear it from that word. The hit test maps the pointer to a
        // character, the character to a word, and the word carries its own measured time —
        // the same table the marker follows, so what is clicked is what is heard.
        var mine = runs;
        OnClick(block, point =>
        {
            var hit = block.TextLayout.HitTestPoint(point);
            var index = hit.TextPosition;

            // A direct hit plays; a click in the space between two timed words plays from the
            // word on its left. A click beyond the timed words — in a grey untimed stretch —
            // does nothing, because the honest time for it does not exist yet.
            var word = mine.FirstOrDefault(r => index >= r.CharStart && index < r.CharEnd);

            if (word is null
                && mine.LastOrDefault(r => r.CharEnd <= index) is { } left
                && mine.Any(r => r.CharStart > index))
            {
                word = left;
            }

            if (word is not null)
            {
                StartPlayback(word.Word.StartSeconds);
            }
        });

        ShowClickTarget(block, mine);
        OfferCorrections(block, paragraph, mine);
        return block;
    }

    /// <summary>
    /// Right-click a word, or a selection, to correct it — here or everywhere, the same two
    /// reaches a speaker's name has. A left click is already taken: it plays the word. The
    /// selectable block's own Copy moves into this menu, since one control has one menu.
    /// </summary>
    private void OfferCorrections(SelectableTextBlock block, TranscriptParagraph paragraph, List<WordRun> runs)
    {
        Point? pressedAt = null;
        var menu = new ContextMenu();

        block.ContextFlyout = null;
        block.ContextMenu = menu;
        HoldRebuildsWhileOpen(menu);

        block.AddHandler(
            PointerPressedEvent,
            (_, e) =>
            {
                var button = e.GetCurrentPoint(block).Properties;

                // Control-click is a right-click on a Mac, and arrives as a left press.
                var controlClick = OperatingSystem.IsMacOS()
                    && button.IsLeftButtonPressed
                    && e.KeyModifiers.HasFlag(KeyModifiers.Control);

                if (button.IsRightButtonPressed || controlClick)
                {
                    pressedAt = e.GetPosition(block);
                }

                if (controlClick)
                {
                    menu.Open(block);
                }
            },
            RoutingStrategies.Bubble,
            handledEventsToo: true);

        menu.Opening += (_, _) =>
        {
            menu.Items.Clear();

            var text = block.Inlines is { Count: > 0 } inlines ? inlines.Text ?? string.Empty : block.Text ?? string.Empty;
            var low = Math.Min(block.SelectionStart, block.SelectionEnd);
            var high = Math.Max(block.SelectionStart, block.SelectionEnd);
            var at = pressedAt is { } point ? block.TextLayout.HitTestPoint(point).TextPosition : low;

            // On the selection means the selection; anywhere else means the word clicked.
            var found = high > low && at >= low && at <= high
                ? WordCorrections.WordsAround(text, low, high)
                : WordCorrections.WordsAround(text, at, at);

            if (found is var (start, words))
            {
                var seconds = runs.FirstOrDefault(run => start >= run.CharStart && start < run.CharEnd)?.Word.StartSeconds
                    ?? paragraph.StartSeconds
                        + ((paragraph.EndSeconds - paragraph.StartSeconds) * start / Math.Max(1, text.Length));

                var shown = words.Length > 40 ? $"{words[..39]}…" : words;
                var places = _viewModel.PlacesOf(words);

                var here = new MenuItem { Header = $"Correct “{shown}” here…" };
                var everywhere = new MenuItem
                {
                    Header = places == 1
                        ? $"Correct “{shown}” everywhere…"
                        : $"Correct “{shown}” everywhere ({places} places)…",
                };

                here.Click += async (_, _) => await AskForCorrection(words, everywhere: false, seconds);
                everywhere.Click += async (_, _) => await AskForCorrection(words, everywhere: true, seconds);

                menu.Items.Add(here);
                menu.Items.Add(everywhere);
                menu.Items.Add(new Separator());
            }

            var copy = new MenuItem { Header = "Copy", IsEnabled = high > low };
            copy.Click += (_, _) => block.Copy();
            menu.Items.Add(copy);
        };
    }

    private async Task AskForCorrection(string words, bool everywhere, double seconds)
    {
        var right = await InputDialog.AskAsync(this, "Correct", $"What should “{words}” say?", words);

        if (!string.IsNullOrWhiteSpace(right))
        {
            _viewModel.CorrectWords(words, right.Trim(), everywhere, seconds);
        }
    }

    /// <summary>
    /// Underlines the word a click would play from, as the pointer crosses it — the cue the
    /// hand cursor used to give, which selectable text cannot wear. Not while a button is
    /// down: that is a selection being dragged, and an underline chasing it reads as noise.
    /// </summary>
    private static void ShowClickTarget(SelectableTextBlock block, List<WordRun> words)
    {
        WordRun? lit = null;

        void Light(WordRun? word)
        {
            if (word == lit)
            {
                return;
            }

            if (lit is not null)
            {
                lit.Run.TextDecorations = null;
            }

            if (word is not null)
            {
                word.Run.TextDecorations = TextDecorations.Underline;
            }

            lit = word;
        }

        block.AddHandler(
            PointerMovedEvent,
            (_, e) =>
            {
                if (e.GetCurrentPoint(block).Properties.IsLeftButtonPressed)
                {
                    Light(null);
                    return;
                }

                var hit = block.TextLayout.HitTestPoint(e.GetPosition(block));
                var index = hit.TextPosition;

                Light(hit.IsInside
                    ? words.FirstOrDefault(r => index >= r.CharStart && index < r.CharEnd)
                    : null);
            },
            RoutingStrategies.Bubble,
            handledEventsToo: true);

        block.PointerExited += (_, _) => Light(null);
    }

    /// <summary>
    /// A click, as opposed to a drag: the text is selectable now, so pressing on a word may be
    /// the start of a selection. The seek waits for the release and happens only if the
    /// pointer barely moved and nothing ended up selected — a click hears the word, a drag
    /// copies it, and neither does the other's job.
    /// </summary>
    private static void OnClick(SelectableTextBlock block, Action<Point> clicked)
    {
        Point? pressedAt = null;

        // Handled-too, because the selectable block claims its own pointer presses to start
        // a selection, and an ordinary handler would never hear them.
        block.AddHandler(
            PointerPressedEvent,
            (_, e) =>
            {
                pressedAt = e.GetCurrentPoint(block).Properties.IsLeftButtonPressed
                        && !e.KeyModifiers.HasFlag(KeyModifiers.Control)
                    ? e.GetPosition(block)
                    : null;
            },
            RoutingStrategies.Bubble,
            handledEventsToo: true);

        block.AddHandler(
            PointerReleasedEvent,
            (_, e) =>
            {
                if (pressedAt is not { } from)
                {
                    return;
                }

                pressedAt = null;
                var to = e.GetPosition(block);
                var moved = Math.Abs(to.X - from.X) + Math.Abs(to.Y - from.Y);

                if (moved <= 4 && string.IsNullOrEmpty(block.SelectedText))
                {
                    clicked(to);
                }
            },
            RoutingStrategies.Bubble,
            handledEventsToo: true);
    }

    /// <summary>The paragraph holding the latest selection, for Edit > Copy.</summary>
    private SelectableTextBlock? _selecting;

    private void TrackSelection(SelectableTextBlock block)
    {
        block.AddHandler(
            PointerReleasedEvent,
            (_, _) =>
            {
                if (!string.IsNullOrEmpty(block.SelectedText))
                {
                    // One selection at a time, as in any reader: the paragraph left behind
                    // lets go of its highlight rather than looking copied along with this one.
                    if (_selecting is { } previous && previous != block)
                    {
                        previous.SelectionStart = previous.SelectionEnd = 0;
                    }

                    _selecting = block;
                }
            },
            RoutingStrategies.Bubble,
            handledEventsToo: true);
    }

    /// <summary>
    /// Edit > Copy and Copy Transcript. The menu owns ⌘C once it lists Copy, so Copy serves
    /// whatever has the selection: the search box when it is focused, otherwise the paragraph
    /// last selected in. Copy Transcript is the way across paragraphs — each is its own block,
    /// and a drag selects within one.
    /// </summary>
    private void BuildMenus()
    {
        var command = OperatingSystem.IsMacOS() ? KeyModifiers.Meta : KeyModifiers.Control;

        var copy = new NativeMenuItem("Copy") { Gesture = new KeyGesture(Key.C, command) };
        copy.Click += async (_, _) =>
        {
            if (FocusManager?.GetFocusedElement() is TextBox box)
            {
                box.Copy();
                return;
            }

            if (_selecting?.SelectedText is { Length: > 0 } text && Clipboard is { } clipboard)
            {
                await clipboard.SetTextAsync(text);
            }
        };

        var copyAll = new NativeMenuItem("Copy Transcript")
        {
            Gesture = new KeyGesture(Key.C, command | KeyModifiers.Shift),
        };
        copyAll.Click += async (_, _) =>
        {
            if (_viewModel.Transcript is { Length: > 0 } text && Clipboard is { } clipboard)
            {
                await clipboard.SetTextAsync(text);
                StatusText.Text = "Transcript copied, with speaker names.";
            }
        };

        var edit = new NativeMenuItem("Edit") { Menu = new NativeMenu { Items = { copy, copyAll } } };

        // The toolbar names most of these with a glyph alone; the menu is where each one is
        // spelled out, with its shortcut, as a Mac user expects to find it.
        _openItem = new NativeMenuItem("Open…") { Gesture = new KeyGesture(Key.O, command) };
        _openItem.Click += (_, _) => OnOpenClicked(this, new RoutedEventArgs());
        _recordItem = new NativeMenuItem("Record") { Gesture = new KeyGesture(Key.R, command) };
        _recordItem.Click += (_, _) => OnRecordClicked(this, new RoutedEventArgs());
        _saveItem = new NativeMenuItem("Save…") { Gesture = new KeyGesture(Key.S, command) };
        _saveItem.Click += (_, _) => OnSaveClicked(this, new RoutedEventArgs());
        _exportItem = new NativeMenuItem("Export…") { Gesture = new KeyGesture(Key.E, command) };
        _exportItem.Click += (_, _) => OnExportClicked(this, new RoutedEventArgs());
        _discardItem = new NativeMenuItem("Discard Transcript");
        _discardItem.Click += (_, _) => OnDiscardClicked(this, new RoutedEventArgs());

        var file = new NativeMenuItem("File")
        {
            Menu = new NativeMenu
            {
                Items =
                {
                    _openItem, _recordItem, new NativeMenuItemSeparator(),
                    _saveItem, _exportItem, new NativeMenuItemSeparator(), _discardItem,
                },
            },
        };

        NativeMenu.SetMenu(this, new NativeMenu { Items = { file, edit } });
    }

    private NativeMenuItem? _openItem;
    private NativeMenuItem? _recordItem;
    private NativeMenuItem? _saveItem;
    private NativeMenuItem? _exportItem;
    private NativeMenuItem? _discardItem;

    /// <summary>The File menu follows the toolbar: a shortcut must never do what its button cannot.</summary>
    private void SyncMenu()
    {
        if (_openItem is null)
        {
            return;
        }

        _openItem.IsEnabled = OpenButton.IsEnabled;
        _recordItem!.IsEnabled = RecordButton.IsEnabled;
        _recordItem.Header = _viewModel.IsRecording ? "Stop Recording" : "Record";
        _saveItem!.IsEnabled = SaveButton.IsEnabled;
        _exportItem!.IsEnabled = ExportButton.IsEnabled;
        _discardItem!.IsEnabled = DiscardButton.IsEnabled;
    }

    /// <summary>
    /// Playback position updates are coalesced, latest-wins: at most one UI update in flight,
    /// reading the newest position when it runs. Twenty queued updates a second against slow
    /// repaints made the marker replay the past on Windows, and nothing about that lesson was
    /// platform-specific.
    /// </summary>
    private double _latestPositionSeconds;
    private int _markerUpdateQueued;

    private void OnPlaybackPosition(double seconds)
    {
        Interlocked.Exchange(ref _latestPositionSeconds, seconds);

        if (Interlocked.Exchange(ref _markerUpdateQueued, 1) == 0)
        {
            Dispatcher.UIThread.Post(() =>
            {
                Interlocked.Exchange(ref _markerUpdateQueued, 0);
                PaintMarker(Volatile.Read(ref _latestPositionSeconds));
            });
        }
    }

    private void OnPlaybackStopped() => Post(() =>
    {
        PaintMarker(double.NegativeInfinity);
        RefreshControls();
        PlayPauseIcon.Data = (Geometry)this.FindResource("IconPlay")!;

        TimeReadout.Text = _pausedAtSeconds is { } paused
            ? $"{Clock(paused)} / {Clock(_viewModel.Player.DurationSeconds)}"
            : $"0:00 / {Clock(_viewModel.Player.DurationSeconds)}";
    });

    private static string Clock(double seconds) =>
        TimeSpan.FromSeconds(Math.Max(0, seconds)).ToString(seconds >= 3600 ? @"h\:mm\:ss" : @"m\:ss");

    /// <summary>Every run this window has ever lit, so a clear can never be missed.</summary>
    private readonly HashSet<Run> _highlightedRuns = [];

    /// <summary>
    /// The marker rules, at word grain, each clause a paid-for lesson: a word covering the
    /// instant beats paragraph bounds; otherwise the most recently begun word. Clearing walks
    /// everything ever lit rather than remembering the one to clear — the Windows window
    /// tracked "the previous word" and a missed clear left two highlights on screen.
    /// </summary>
    private void PaintMarker(double seconds)
    {
        var currentParagraph = -1;
        WordRun? covering = null;
        WordRun? mostRecent = null;
        var mostRecentParagraph = -1;

        for (var p = 0; p < _paragraphWordRuns.Count; p++)
        {
            foreach (var run in _paragraphWordRuns[p])
            {
                if (run.Word.StartSeconds <= seconds && seconds <= run.Word.EndSeconds)
                {
                    covering = run;
                    currentParagraph = p;
                    break;
                }

                if (run.Word.StartSeconds <= seconds
                    && (mostRecent is null || run.Word.StartSeconds >= mostRecent.Word.StartSeconds))
                {
                    mostRecent = run;
                    mostRecentParagraph = p;
                }
            }

            if (covering is not null)
            {
                break;
            }
        }

        var lit = covering ?? mostRecent;

        if (covering is null)
        {
            currentParagraph = mostRecentParagraph;
        }

        foreach (var stale in _highlightedRuns)
        {
            // A run can carry a search highlight under the marker; clearing the marker gives
            // the search colour back rather than wiping both.
            stale.Background = _searchRuns.Contains(stale) ? SearchBrush : null;
        }

        _highlightedRuns.Clear();

        for (var i = 0; i < _paragraphBorders.Count; i++)
        {
            if (i == currentParagraph)
            {
                _paragraphBorders[i].Background = MarkerBrush;
            }
            else
            {
                // Cleared rather than set to null: a local null would beat the hover style,
                // and a card the pointer is resting on would stop saying it is clickable.
                _paragraphBorders[i].ClearValue(Border.BackgroundProperty);
            }
        }

        if (lit is not null)
        {
            lit.Run.Background = WordBrush;
            _highlightedRuns.Add(lit.Run);
        }

        if (currentParagraph >= 0 && _followPlayback)
        {
            KeepInView(currentParagraph, lit);
        }

        if (_viewModel.Player.IsPlaying || seconds >= 0)
        {
            TimeReadout.Text = $"{Clock(seconds)} / {Clock(_viewModel.Player.DurationSeconds)}";
        }

        _waveform.SetPosition(seconds);
    }

    /// <summary>Whether any part of a row is inside the transcript's viewport.</summary>
    private bool IsVisibleInTranscript(Control row)
    {
        if (row.TranslatePoint(new Avalonia.Point(0, 0), TranscriptScroll) is not { } top)
        {
            return false;
        }

        return top.Y + row.Bounds.Height > 0 && top.Y < TranscriptScroll.Bounds.Height;
    }

    /* ---- search ------------------------------------------------------------------------ */

    private static readonly IBrush SearchBrush = new SolidColorBrush(Color.FromArgb(70, 235, 180, 60));
    private static readonly IBrush CurrentSearchBrush = new SolidColorBrush(Color.FromArgb(150, 235, 180, 60));

    private readonly HashSet<Run> _searchRuns = [];
    private readonly List<(int Paragraph, IReadOnlyList<WordRun> Runs)> _searchMatches = [];
    private int _searchIndex = -1;

    protected override void OnKeyDown(Avalonia.Input.KeyEventArgs e)
    {
        if (e.Key == Avalonia.Input.Key.F
            && e.KeyModifiers.HasFlag(Avalonia.Input.KeyModifiers.Meta))
        {
            SearchBox.Focus();
            SearchBox.SelectAll();
            e.Handled = true;
            return;
        }

        base.OnKeyDown(e);
    }

    private void OnSearchChanged(object? sender, TextChangedEventArgs e) => RunSearch();

    private void OnSearchKeyDown(object? sender, Avalonia.Input.KeyEventArgs e)
    {
        switch (e.Key)
        {
            case Avalonia.Input.Key.Enter:
                StepSearch(e.KeyModifiers.HasFlag(Avalonia.Input.KeyModifiers.Shift) ? -1 : +1);
                e.Handled = true;
                break;

            case Avalonia.Input.Key.Escape:
                SearchBox.Text = string.Empty;
                TranscriptScroll.Focus();
                e.Handled = true;
                break;
        }
    }

    /// <summary>
    /// Finds every occurrence and lights the words carrying it. Matching is done against the
    /// same display text the runs were built from, so a hit maps straight onto the words —
    /// and a phrase spanning several words lights all of them.
    /// </summary>
    private void RunSearch()
    {
        foreach (var run in _searchRuns)
        {
            run.Background = null;
        }

        _searchRuns.Clear();
        _searchMatches.Clear();
        _searchIndex = -1;

        var needle = SearchBox.Text?.Trim();

        if (string.IsNullOrEmpty(needle))
        {
            StatusText.Text = _viewModel.Status;
            return;
        }

        for (var p = 0; p < _paragraphWordRuns.Count; p++)
        {
            var runs = _paragraphWordRuns[p];

            if (runs.Count == 0)
            {
                continue;
            }

            var text = ParagraphDisplayText(runs);

            for (var at = text.IndexOf(needle, StringComparison.OrdinalIgnoreCase);
                 at >= 0;
                 at = text.IndexOf(needle, at + 1, StringComparison.OrdinalIgnoreCase))
            {
                var end = at + needle.Length;
                var hit = runs.Where(r => r.CharEnd > at && r.CharStart < end).ToList();

                if (hit.Count > 0)
                {
                    _searchMatches.Add((p, hit));
                }
            }
        }

        foreach (var (_, runs) in _searchMatches)
        {
            foreach (var run in runs)
            {
                run.Run.Background = SearchBrush;
                _searchRuns.Add(run.Run);
            }
        }

        StatusText.Text = _searchMatches.Count switch
        {
            0 => $"No match for “{needle}”.",
            1 => "1 match — Enter to jump to it.",
            var n => $"{n} matches — Enter to step through them.",
        };

        if (_searchMatches.Count > 0)
        {
            StepSearch(+1);
        }
    }

    private static string ParagraphDisplayText(IReadOnlyList<WordRun> runs)
    {
        var text = new System.Text.StringBuilder(runs[^1].CharEnd);

        foreach (var run in runs)
        {
            if (text.Length > 0)
            {
                text.Append(' ');
            }

            text.Append(run.Word.Text);
        }

        return text.ToString();
    }

    private void StepSearch(int direction)
    {
        if (_searchMatches.Count == 0)
        {
            return;
        }

        if (_searchIndex >= 0)
        {
            foreach (var run in _searchMatches[_searchIndex].Runs)
            {
                run.Run.Background = SearchBrush;
            }
        }

        _searchIndex = ((_searchIndex + direction) % _searchMatches.Count + _searchMatches.Count)
            % _searchMatches.Count;

        var (paragraph, runs) = _searchMatches[_searchIndex];

        foreach (var run in runs)
        {
            run.Run.Background = CurrentSearchBrush;
        }

        // Going to a match is the reader taking the view, as scrolling is.
        _followPlayback = false;
        _paragraphBorders[paragraph].BringIntoView();
        StatusText.Text = $"Match {_searchIndex + 1} of {_searchMatches.Count}.";
    }

    /// <summary>
    /// Finds the model root: beside the executable in a published app, or up the tree in a
    /// development run, where the binary sits four directories below the repository.
    /// </summary>
    internal static string FindModelRoot()
    {
        // Trimmed first: BaseDirectory carries a trailing separator, and GetDirectoryName
        // spends its first call removing it rather than hopping — which once left this walk
        // one level short of the repository and sent the provisioner off to download three
        // gigabytes of models the machine already had.
        var directory = Path.TrimEndingDirectorySeparator(AppContext.BaseDirectory);

        for (var hops = 0; hops < 7 && directory is not null; hops++)
        {
            var candidate = Path.Combine(directory, "models");

            if (Directory.Exists(candidate))
            {
                return candidate;
            }

            directory = Path.GetDirectoryName(directory);
        }

        // Inside a bundle, models live in Application Support rather than beside the binary:
        // the provisioner writes gigabytes on first run, and writing into a signed .app
        // invalidates its signature.
        // Spelled out rather than via SpecialFolder, which .NET maps to ~/.config on macOS —
        // a Linux convention no Mac user would think to look in.
        if (AppContext.BaseDirectory.Contains(".app/Contents/MacOS", StringComparison.Ordinal))
        {
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "Library", "Application Support", "LocalScribe", "models");
        }

        return Path.Combine(AppContext.BaseDirectory, "models");
    }
}
