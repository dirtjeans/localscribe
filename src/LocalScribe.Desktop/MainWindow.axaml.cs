using System.ComponentModel;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using LocalScribe.App;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Models;
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
        _viewModel.PropertyChanged += OnViewModelChanged;
        _viewModel.Player.PositionChanged += OnPlaybackPosition;
        _viewModel.Player.Stopped += OnPlaybackStopped;
        _viewModel.Player.Failed += message => Post(() => StatusText.Text = message);

        InitializeComponent();

        _waveform.PeakSource = buckets => _viewModel.WaveformPeaks(buckets);
        _waveform.SeekRequested += seconds => _viewModel.Player.PlayFrom(seconds);
        WaveformHost.Content = _waveform;

        RefreshControls();

        Opened += OnOpened;
        Closing += OnClosing;
    }

    private async void OnOpened(object? sender, EventArgs e)
    {
        Opened -= OnOpened;

        await ProvisionModelsAsync();

        await _viewModel.InitialiseAsync();

        // Only now is the offer honest: before initialisation, "no cleanup model" would just
        // mean "not looked yet".
        _initialised = true;
        RefreshControls();

        await _viewModel.PreloadAsync();

        if (_openOnLaunch is not null)
        {
            await OpenPathAsync(_openOnLaunch);
        }
    }

    private bool _initialised;
    private bool _provisioningFoundry;

    /// <summary>
    /// The user asked, so the app installs: Homebrew tap, service start, model download — the
    /// same flow the glossary notice drives on Windows, offered here as a banner because the
    /// macOS window has no glossary dialog yet. The click is the consent; nothing here runs
    /// uninvited.
    /// </summary>
    private async void OnFoundryClicked(object? sender, RoutedEventArgs e)
    {
        if (_provisioningFoundry)
        {
            return;
        }

        _provisioningFoundry = true;
        FoundryButton.IsEnabled = false;
        WorkProgress.IsVisible = true;

        try
        {
            await _viewModel.ProvisionCleanupAsync();
        }
        finally
        {
            _provisioningFoundry = false;
            FoundryButton.IsEnabled = true;
            WorkProgress.IsVisible = false;
            RefreshControls();
        }
    }

    /// <summary>
    /// A first launch downloads its own models, saying what it costs before spending it —
    /// about 2.8 GiB all told — and every stage degrades the way the pipeline already does, so
    /// a failed download costs precision or labels, never the app. Model weights only:
    /// installing services (Foundry Local) stays behind an explicit user action.
    /// </summary>
    private async Task ProvisionModelsAsync()
    {
        var modelRoot = FindModelRoot();

        if (!ModelProvisioner.NeedsAnything(modelRoot, whisperCpp: true))
        {
            return;
        }

        StatusText.Text = "First run: downloading the models this app runs on (about 2.8 GiB). "
            + "The app is usable the moment they land — nothing else to set up.";
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
    /// The engine the advisor recommends is the engine the window runs: whisper.cpp whenever
    /// its model is on disk, the ONNX layout otherwise.
    /// </summary>
    private static ITranscriber OpenEngine(ExecutionPlan plan, string? onnxDirectory)
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

    private void OnPlayClicked(object? sender, RoutedEventArgs e)
    {
        if (_viewModel.Player.IsPlaying)
        {
            _viewModel.Player.Stop();
        }
        else if (_viewModel.Player.HasAudio)
        {
            _viewModel.Player.PlayFrom(0);
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
    /// Runs the speakers again with a count. The count is the one thing a person always knows
    /// and the algorithm never does; empty means infer it.
    /// </summary>
    private async void OnSpeakersClicked(object? sender, RoutedEventArgs e)
    {
        var answer = await InputDialog.AskAsync(
            this,
            "Find speakers",
            "How many people are talking? Leave empty to work it out.");

        if (answer is null)
        {
            return;
        }

        await _viewModel.FindSpeakersAsync(
            int.TryParse(answer, out var count) && count > 0 ? count : null);
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
                break;

            case nameof(MainViewModel.HardwareSummary):
                HardwareText.Text = _viewModel.HardwareSummary;
                break;

            case nameof(MainViewModel.Progress):
                WorkProgress.Value = _viewModel.Progress;
                WorkProgress.IsVisible = _viewModel.Progress is > 0 and < 1;
                break;

            case nameof(MainViewModel.Paragraphs):
                RebuildParagraphs();
                break;

            case nameof(MainViewModel.ProvisionalText):
                ProvisionalTextBlock.Text = _viewModel.ProvisionalText;
                break;

            case nameof(MainViewModel.SpeakerCount):
                SpeakerBadge.Text = _viewModel.SpeakerCount switch
                {
                    0 => string.Empty,
                    1 => "1 speaker",
                    var n => $"{n} speakers",
                };
                break;
        }

        RefreshControls();
    });

    private void RefreshControls()
    {
        CleanupOffer.IsVisible = _initialised && !_provisioningFoundry && _viewModel.CleanupModel is null;
        OpenButton.IsEnabled = !_viewModel.IsBusy && !_viewModel.IsRecording;
        RecordButton.IsEnabled = !_viewModel.IsBusy && !_viewModel.IsPreparing;
        RecordButton.Content = _viewModel.IsRecording ? "Stop recording" : "Record";
        PlayButton.IsEnabled = _viewModel.Player.HasAudio && !_viewModel.IsRecording;
        SaveButton.IsEnabled = _viewModel.CanSaveArchive;
        ExportButton.IsEnabled = _viewModel.HasTranscript;
        SpeakersButton.IsEnabled = _viewModel.CanFindSpeakers && !_viewModel.IsBusy;
        DiscardButton.IsEnabled = _viewModel.HasTranscript;

        WaveformHost.IsVisible = _viewModel.Player.HasAudio;

        if (_viewModel.Player.HasAudio)
        {
            _waveform.SetAudio(_viewModel.Player.DurationSeconds);
        }
    }

    /* ---- the transcript and the marker ------------------------------------------------- */

    private static readonly IBrush MarkerBrush = new SolidColorBrush(Color.FromArgb(48, 64, 128, 255));
    private static readonly IBrush WordBrush = new SolidColorBrush(Color.FromArgb(150, 64, 128, 255));

    /// <summary>Per paragraph: each word's run, its time, and where it starts in the display text.</summary>
    private sealed record WordRun(Run Run, WordTimings.Word Word, int CharStart, int CharEnd);

    private readonly List<IReadOnlyList<WordRun>> _paragraphWordRuns = [];

    private void RebuildParagraphs()
    {
        _shownParagraphs = _viewModel.Paragraphs;
        _paragraphBorders.Clear();
        _paragraphWordRuns.Clear();
        _highlightedRuns.Clear();
        ParagraphsPanel.Children.Clear();

        foreach (var paragraph in _shownParagraphs)
        {
            var lines = new StackPanel { Spacing = 2 };

            if (paragraph.Speaker is { Length: > 0 } speaker)
            {
                var label = new TextBlock
                {
                    Text = paragraph.Overlapped ? $"{speaker} — crosstalk" : speaker,
                    FontWeight = FontWeight.Bold,
                    FontSize = 13,
                    Opacity = 0.8,
                };

                // Renaming hangs off the label because the label is the thing that is wrong.
                // All three verbs are needed: "this part" for a paragraph misfiled, "everywhere"
                // for a person split across labels, "by voice" for one label meaning two people.
                label.ContextMenu = BuildSpeakerMenu(paragraph);
                lines.Children.Add(label);
            }

            lines.Children.Add(BuildWordBlock(paragraph));

            var border = new Border
            {
                Child = lines,
                Padding = new Avalonia.Thickness(12, 6),
                CornerRadius = new Avalonia.CornerRadius(6),
                HorizontalAlignment = HorizontalAlignment.Stretch,
            };

            _paragraphBorders.Add(border);
            ParagraphsPanel.Children.Add(border);
        }
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

        return new ContextMenu { Items = { thisPart, everywhere, byVoice } };
    }

    private async Task<string?> AskForName(string? current)
    {
        var name = await InputDialog.AskAsync(
            this, "Rename speaker", "Who is this?", current ?? string.Empty);

        return string.IsNullOrWhiteSpace(name) ? null : name.Trim();
    }

    /// <summary>
    /// A paragraph as one run per word, so a word can be lit and a click can be mapped back to
    /// a time. The display text is built from the words themselves rather than from the
    /// paragraph's own string: the two agree except over stray whitespace, and it is the words
    /// that carry the times.
    /// </summary>
    private Control BuildWordBlock(TranscriptParagraph paragraph)
    {
        var words = _viewModel.WordsIn(paragraph.Segments);

        var block = new TextBlock
        {
            TextWrapping = TextWrapping.Wrap,
            FontSize = 15,
            LineHeight = 24,
        };

        if (words.Count == 0)
        {
            // Nothing timed — a transcript with no audio behind it. Plain text, and a click
            // starts at the paragraph, which is the best time anybody has.
            block.Text = paragraph.Text;
            var start = paragraph.StartSeconds;
            block.PointerPressed += (_, _) => _viewModel.Player.PlayFrom(start);
            _paragraphWordRuns.Add([]);
            return block;
        }

        var runs = new List<WordRun>(words.Count);
        var at = 0;

        foreach (var word in words)
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

        _paragraphWordRuns.Add(runs);

        // Click a word to hear it from that word. The hit test maps the pointer to a
        // character, the character to a word, and the word carries its own measured time —
        // the same table the marker follows, so what is clicked is what is heard.
        var mine = runs;
        block.PointerPressed += (_, e) =>
        {
            var point = e.GetPosition(block);
            var hit = block.TextLayout.HitTestPoint(point);
            var index = hit.TextPosition;

            var word = mine.FirstOrDefault(r => index >= r.CharStart && index < r.CharEnd)
                ?? mine.LastOrDefault(r => r.CharStart <= index);

            if (word is not null)
            {
                _viewModel.Player.PlayFrom(word.Word.StartSeconds);
            }
        };

        return block;
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
        PlayButton.Content = "Play";
    });

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
            stale.Background = null;
        }

        _highlightedRuns.Clear();

        for (var i = 0; i < _paragraphBorders.Count; i++)
        {
            _paragraphBorders[i].Background = i == currentParagraph ? MarkerBrush : null;
        }

        if (lit is not null)
        {
            lit.Run.Background = WordBrush;
            _highlightedRuns.Add(lit.Run);
        }

        if (currentParagraph >= 0)
        {
            PlayButton.Content = "Stop playback";
            _paragraphBorders[currentParagraph].BringIntoView();
        }

        _waveform.SetPosition(seconds);
    }

    /// <summary>
    /// Finds the model root: beside the executable in a published app, or up the tree in a
    /// development run, where the binary sits four directories below the repository.
    /// </summary>
    private static string FindModelRoot()
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
