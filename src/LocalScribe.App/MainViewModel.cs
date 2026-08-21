using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text;
using LocalScribe.Core.Audio;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Models;
using LocalScribe.Core.Pipeline;
using LocalScribe.Core.Refinement;
using LocalScribe.Core.Transcription;
using LocalScribe.Onnx;

namespace LocalScribe.App;

/// <summary>
/// Drives the window. Holds no UI types, so the interesting behaviour can be reasoned about
/// (and later tested) without standing up a WinUI host.
/// </summary>
public sealed class MainViewModel : INotifyPropertyChanged, IDisposable
{
    private readonly string _modelRoot;
    private ExecutionPlan? _plan;
    private DeviceCapabilities? _capabilities;
    private CancellationTokenSource? _cancellation;
    private MicrophoneCapture? _microphone;
    private LiveTranscriptionSession? _liveSession;

    /// <summary>
    /// The live transcriber is kept between recordings rather than rebuilt. Opening the QNN
    /// sessions takes about five seconds, and the session does not own it, so re-creating it per
    /// recording both leaked the old one and paid that cost again every time.
    /// </summary>
    private ITranscriber? _transcriber;
    private string? _transcriberKey;
    private readonly SemaphoreSlim _transcriberLock = new(1, 1);
    private bool _modelReady;
    private SessionDiagnostics? _diagnostics;

    /// <summary>
    /// The segments behind the displayed text. Kept because the transcript is no longer just a
    /// string: paragraphs, export formats and clicking a line to hear it all need the timings.
    /// </summary>
    private IReadOnlyList<TranscriptSegment> _segments = [];

    /// <summary>
    /// Everything the microphone delivered this session, so a finished recording can be played
    /// back and its transcript clicked through. A recording has no file to re-open, so if this
    /// is not kept the audio is gone the moment capture stops.
    /// </summary>
    private List<float>? _liveCapture;

    /// <summary>
    /// The audio behind the current transcript, and the segments as the transcriber returned
    /// them, before any speaker was attached. Kept so diarization can be run again with a
    /// different answer without transcribing anything twice — and run from the original
    /// segments, since attribution splits them and splitting an already-split transcript
    /// compounds whatever the first attempt got wrong.
    /// </summary>
    private PcmAudio? _audio;
    private IReadOnlyList<TranscriptSegment> _segmentsBeforeSpeakers = [];

    private string _status = "Starting up…";
    private string _hardwareSummary = string.Empty;
    private string _transcript = string.Empty;
    private string _provisionalText = string.Empty;
    private string _summary = string.Empty;
    private double _progress;
    private bool _busy;
    private bool _recording;
    private bool _preparing;
    private bool _warming;

    public MainViewModel(string? modelRoot = null)
    {
        _modelRoot = modelRoot ?? Path.Combine(AppContext.BaseDirectory, "models");
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>
    /// True while the model is loading at startup and nothing has asked for it yet.
    /// <para>
    /// Its own state rather than a flavour of <see cref="IsPreparing"/>. Preparing means the user
    /// pressed record and must not speak; warming up means the app is getting itself ready and
    /// the user can do as they please. Both wait for the same load, and telling someone to hold
    /// still when they have not asked for anything is a different and worse message.
    /// </para>
    /// </summary>
    public bool IsWarmingUp
    {
        get => _warming;
        private set => Set(ref _warming, value);
    }

    /// <summary>True once the model is loaded and a recording can start immediately.</summary>
    public bool IsModelReady
    {
        get => _modelReady;
        private set => Set(ref _modelReady, value);
    }

    /// <summary>Summary and action items from the cleanup model, empty when there was none.</summary>
    public string Summary
    {
        get => _summary;
        private set => Set(ref _summary, value);
    }

    /// <summary>Plays back whatever was last transcribed, so a line can be clicked and heard.</summary>
    public TranscriptPlayer Player { get; } = new();

    /// <summary>
    /// The loaded audio reduced to peaks for drawing. Computed here because the view model owns
    /// the samples; the window owns only how wide to draw them.
    /// </summary>
    public float[] WaveformPeaks(int buckets) => Player.Peaks(buckets);

    /// <summary>The transcript grouped into paragraphs, which is what the window shows.</summary>
    public IReadOnlyList<TranscriptParagraph> Paragraphs { get; private set; } = [];

    /// <summary>True when there is a transcript to copy, save or discard.</summary>
    public bool HasTranscript => _segments.Count > 0;

    /// <summary>Where the transcript came from, used to name an export.</summary>
    public string SourceName { get; private set; } = "Transcript";

    /// <summary>
    /// Replaces the transcript and everything derived from it. One place, so the paragraphs, the
    /// flat text and the export never disagree about what the transcript currently is.
    /// </summary>
    private void SetTranscript(IReadOnlyList<TranscriptSegment> segments)
    {
        _segments = segments;
        Paragraphs = TranscriptFormatter.Paragraphs(segments);
        Transcript = TranscriptFormatter.ToPlainText(Paragraphs);

        Raise(nameof(Paragraphs));
        Raise(nameof(HasTranscript));
    }

    /// <summary>The transcript in one of the formats the save dialog offers.</summary>
    public string Export(TranscriptFormat format) => format switch
    {
        TranscriptFormat.Markdown => TranscriptFormatter.ToMarkdown(Paragraphs, SourceName),
        TranscriptFormat.SubRip => TranscriptFormatter.ToSubRip(_segments),
        _ => TranscriptFormatter.ToPlainText(Paragraphs),
    };

    /// <summary>
    /// Works out who spoke when, if the speaker models are installed.
    /// <para>
    /// Optional on purpose. Whisper does not separate voices and the models that do are a
    /// separate download, so a machine without them transcribes exactly as before rather than
    /// failing. A failure here costs the labels, not the transcript.
    /// </para>
    /// </summary>
    private async Task<IReadOnlyList<TranscriptSegment>> AttributeSpeakersAsync(
        IReadOnlyList<TranscriptSegment> segments,
        PcmAudio audio,
        int? speakers = null)
    {
        var directory = Path.Combine(_modelRoot, "diarization");

        if (!File.Exists(Path.Combine(directory, "segmentation.onnx")))
        {
            return segments;
        }

        Status = "Working out who spoke…";

        try
        {
            return await Task.Run(() =>
            {
                using var diarizer = SpeakerDiarizer.Load(directory);

                var turns = diarizer.Diarize(
                    audio,
                    maxSpeakers: speakers,
                    exactSpeakers: speakers,
                    cancellationToken: _cancellation?.Token ?? default);

                return SpeakerDiarizer.Attribute(segments, turns);
            }).ConfigureAwait(true);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception exception)
        {
            Status = $"Transcribed, but speakers could not be identified: {exception.Message}";
            return segments;
        }
    }

    /// <summary>True when there is audio to work out speakers from.</summary>
    public bool CanFindSpeakers => _audio is not null && _segmentsBeforeSpeakers.Count > 0;

    /// <summary>
    /// Works out the speakers again, told how many there are.
    /// <para>
    /// The count is the one thing a person always knows and the algorithm never does. Without it
    /// the number of speakers has to be inferred from a distance threshold, and a threshold that
    /// is slightly wrong for a recording fails in both directions at once — merging two people
    /// who sound alike while splitting one who moved closer to the microphone. Being told there
    /// are three removes the guess entirely.
    /// </para>
    /// </summary>
    /// <param name="speakers">How many people are talking, or null to infer it.</param>
    public async Task FindSpeakersAsync(int? speakers)
    {
        if (_audio is not { } audio || _segmentsBeforeSpeakers.Count == 0)
        {
            return;
        }

        Status = speakers is { } n ? $"Finding {n} speakers…" : "Working out who spoke…";

        var attributed = await AttributeSpeakersAsync(_segmentsBeforeSpeakers, audio, speakers);

        SetTranscript(attributed);

        var found = attributed.Select(s => s.Speaker).Where(s => s is not null).Distinct().Count();
        Status = found == 0 ? "No speakers were found." : $"Found {found} speaker(s).";
    }

    /// <summary>
    /// Renames a speaker, either for one paragraph or throughout.
    /// <para>
    /// Both are needed and they are not the same operation. Diarization splits one person into
    /// two often enough that "this part is also Kim" has to be possible, and having named
    /// someone once, naming every other thing they said has to be one action rather than
    /// thirty.
    /// </para>
    /// </summary>
    /// <param name="startSeconds">Start of the paragraph being renamed.</param>
    /// <param name="endSeconds">End of the paragraph being renamed.</param>
    /// <param name="currentName">The label as it stands, used to find the rest of them.</param>
    /// <param name="newName">What to call them.</param>
    /// <param name="everywhere">True to rename every paragraph currently sharing the label.</param>
    public void RenameSpeaker(
        double startSeconds,
        double endSeconds,
        string? currentName,
        string newName,
        bool everywhere)
    {
        if (string.IsNullOrWhiteSpace(newName) || _segments.Count == 0)
        {
            return;
        }

        var name = newName.Trim();

        var updated = _segments.Select(segment =>
        {
            var inScope = everywhere
                ? string.Equals(segment.Speaker, currentName, StringComparison.Ordinal)

                // Otherwise only the segments inside the paragraph that was clicked. Compared on
                // midpoint so a segment straddling a boundary belongs to one paragraph rather
                // than to both or to neither.
                : Midpoint(segment) >= startSeconds && Midpoint(segment) <= endSeconds;

            return inScope ? segment with { Speaker = name } : segment;
        }).ToList();

        SetTranscript(updated);
        Status = everywhere ? $"Renamed to {name} throughout." : $"Renamed to {name} for that part.";
    }

    private static double Midpoint(TranscriptSegment segment) =>
        segment.StartSeconds + ((segment.EndSeconds - segment.StartSeconds) / 2);

    /// <summary>Throws the transcript away and releases the audio behind it.</summary>
    public void Discard()
    {
        Player.Clear();

        _audio = null;
        _segmentsBeforeSpeakers = [];

        SetTranscript([]);
        ProvisionalText = string.Empty;
        Summary = string.Empty;
        Progress = 0;
        SourceName = "Transcript";
        Status = "Discarded.";
    }

    /// <summary>What the app is doing right now, shown in the status bar.</summary>
    public string Status
    {
        get => _status;
        private set => Set(ref _status, value);
    }

    /// <summary>Where the models are running, so the user can see the NPU is actually in use.</summary>
    public string HardwareSummary
    {
        get => _hardwareSummary;
        private set => Set(ref _hardwareSummary, value);
    }

    /// <summary>Committed transcript text.</summary>
    public string Transcript
    {
        get => _transcript;
        private set => Set(ref _transcript, value);
    }

    /// <summary>
    /// Words the model may still revise. Shown in a lighter style so the user can tell settled
    /// text from a guess that is about to change.
    /// </summary>
    public string ProvisionalText
    {
        get => _provisionalText;
        private set => Set(ref _provisionalText, value);
    }

    public double Progress
    {
        get => _progress;
        private set => Set(ref _progress, value);
    }

    public bool IsBusy
    {
        get => _busy;
        private set => Set(ref _busy, value);
    }

    public bool IsRecording
    {
        get => _recording;
        private set => Set(ref _recording, value);
    }

    /// <summary>
    /// True while the model is loading and the microphone is not yet running.
    /// <para>
    /// This is its own state rather than a flavour of <see cref="IsBusy"/> because it is the one
    /// moment the user must not speak. Loading the QNN sessions takes seconds, and anything said
    /// before the microphone starts is gone — not delayed, gone.
    /// </para>
    /// </summary>
    public bool IsPreparing
    {
        get => _preparing;
        private set => Set(ref _preparing, value);
    }

    /// <summary>
    /// What is doing the cleanup, or null when nothing is. The glossary, the summary and the
    /// punctuation repair all depend on this, and all three silently do nothing without it —
    /// which is worth saying where the user is looking rather than leaving them to notice.
    /// </summary>
    public string? CleanupModel => _languageModel?.Description;

    /// <summary>Terms the cleanup model should spell correctly. Edited by the user in the UI.</summary>
    public List<string> Glossary { get; } = [];

    /// <summary>
    /// Probes the machine and works out the plan. Runs once at startup; the result is shown in
    /// the UI rather than hidden, because "is it using the NPU?" is the question everyone asks.
    /// </summary>
    public async Task InitialiseAsync()
    {
        Status = "Checking hardware…";

        var capabilities = await Task.Run(() => DeviceProbe.Probe(_modelRoot));

        _languageModel = await LocalLanguageModel.ResolveAsync();
        capabilities = capabilities with { LocalLanguageModelPresent = _languageModel is not null };

        _capabilities = capabilities;
        _plan = AcceleratorPlanner.Plan(capabilities);

        HardwareSummary = _plan.Summary;
        Status = _plan.Warnings.Count == 0
            ? "Ready."
            : $"Ready, with {_plan.Warnings.Count} warning(s). Run localscribe-doctor for detail.";
    }

    /// <summary>Transcribes a file from disk.</summary>
    public async Task TranscribeFileAsync(string path)
    {
        if (_plan is null)
        {
            Status = "Still checking hardware.";
            return;
        }

        IsBusy = true;
        Progress = 0;
        Summary = string.Empty;
        SetTranscript([]);
        _cancellation = new CancellationTokenSource();

        try
        {
            Status = AudioFileLoader.IsVideo(path) ? "Extracting audio…" : "Loading audio…";
            var audio = await Task.Run(() => AudioFileLoader.Load(path), _cancellation.Token);

            SourceName = Path.GetFileNameWithoutExtension(path);
            _audio = audio;

            // Held so a line in the transcript can be clicked and heard. The timings refer to
            // these samples, not to the file, which is why the decoded audio is what is kept.
            Player.Load(audio);

            Status = IsModelReady ? "Preparing…" : "Loading the model…";
            var transcriber = await Task.Run(() => TranscriberFor(_plan), _cancellation.Token);
            var pipeline = new TranscriptionPipeline(transcriber, BuildRefiner());

            Status = $"Transcribing with {transcriber.Description}…";

            // Each window's text as it lands, rather than a bare percentage. A long file is
            // otherwise several minutes of a progress bar and nothing to read.
            var streamed = new List<TranscriptSegment>();

            var progress = new Progress<TranscriptionProgress>(update =>
            {
                Progress = update.Fraction;
                Status = $"Transcribing… {update.ChunksCompleted} of {update.ChunksTotal} windows";

                if (update.LatestText.Length > 0)
                {
                    streamed.Add(new TranscriptSegment(
                        update.LatestText,
                        update.ChunksCompleted * AudioChunker.WindowSeconds,
                        (update.ChunksCompleted + 1) * AudioChunker.WindowSeconds));

                    SetTranscript(streamed.ToList());
                }
            });

            var result = await pipeline.RunAsync(
                audio,
                Glossary,
                RefinementOutputs.Everything,
                progress,
                _cancellation.Token);

            _segmentsBeforeSpeakers = result.Transcript.Segments;

            var segments = await AttributeSpeakersAsync(result.Transcript.Segments, audio);

            SetTranscript(segments);
            Summary = Format(result);
            Status = "Done.";
        }
        catch (OperationCanceledException)
        {
            Status = "Cancelled.";
        }
        catch (Exception exception)
        {
            // Surfacing the message beats a silent failure: most problems here are a missing
            // model file or an export whose signature does not match, and both say so plainly.
            Status = $"Failed: {exception.Message}";
        }
        finally
        {
            IsBusy = false;
            _cancellation?.Dispose();
            _cancellation = null;
        }
    }

    /// <summary>
    /// Starts live transcription from the microphone.
    /// <para>
    /// The model is opened before the microphone, off the UI thread, and the user is told to
    /// wait until it is. This used to run the load synchronously on the click, which froze the
    /// window for about five seconds and started capturing only afterwards — so the opening
    /// words of every recording were lost, with nothing on screen to suggest they would be.
    /// </para>
    /// </summary>
    public async Task StartRecordingAsync()
    {
        if (_plan is null || IsRecording || IsPreparing)
        {
            return;
        }

        // Live work wants a smaller model than batch, so the plan is recomputed for this mode.
        var livePlan = AcceleratorPlanner.Plan(
            _capabilities ?? DeviceCapabilities.Unknown,
            PerformanceProfile.Considerate,
            WorkloadMode.Live);

        IsPreparing = true;
        Status = "Loading the model — wait for the go-ahead before speaking…";

        ITranscriber transcriber;

        try
        {
            transcriber = await Task.Run(() => TranscriberFor(livePlan));
        }
        catch (Exception exception)
        {
            IsPreparing = false;
            Status = $"Could not start listening: {exception.Message}";
            return;
        }

        _liveSession = new LiveTranscriptionSession(transcriber);
        _cancellation = new CancellationTokenSource();
        _diagnostics = SessionDiagnostics.StartIfEnabled(livePlan.Summary, transcriber.Description);
        _liveCapture = [];

        Player.Clear();
        SourceName = $"Recording {DateTime.Now:yyyy-MM-dd HH.mm}";

        _microphone = new MicrophoneCapture();
        _microphone.SamplesAvailable += OnSamplesAvailable;
        _microphone.Start();

        // Only now is anything the user says actually being captured, so this is the first
        // moment it is honest to say so.
        IsPreparing = false;
        IsRecording = true;
        Status = $"Listening — go ahead. ({transcriber.Description})";
    }

    /// <summary>
    /// The cached live transcriber, rebuilt only when the placement or model size changes.
    /// Called on a worker thread; opening ONNX sessions is slow and blocking.
    /// </summary>
    /// <summary>
    /// The loaded transcriber for a plan, opening it only if what is already loaded will not do.
    /// <para>
    /// One slot rather than a cache. Holding a second model would double the memory a QNN
    /// context binary occupies, which is over a gigabyte, to save a load that only happens when
    /// the user switches between recording and transcribing a file.
    /// </para>
    /// <para>
    /// Keyed on the resolved model directory rather than the plan's requested size. The plan
    /// asks for small.en when listening and medium.en for a file, and both may well resolve to
    /// the same weights on disk — reloading a gigabyte because a string differed would be a poor
    /// reason.
    /// </para>
    /// </summary>
    private ITranscriber TranscriberFor(ExecutionPlan plan)
    {
        var directory = ModelDirectoryFor(plan);
        var key = $"{directory}|{plan.Encoder.Device}|{plan.Decoder.Device}|"
            + $"{plan.CpuBudget.IntraOpThreads}|{plan.StrictProviderCheck}";

        // Two callers can arrive at once — a preload running while the user clicks record — and
        // opening the same model twice wastes both the time and the memory.
        _transcriberLock.Wait();

        try
        {
            if (_transcriber is not null && _transcriberKey == key)
            {
                return _transcriber;
            }

            _transcriber?.Dispose();
            _transcriber = null;
            _transcriberKey = null;

            var opened = WhisperOnnxTranscriber.Load(directory, plan);

            _transcriber = opened;
            _transcriberKey = key;

            return opened;
        }
        finally
        {
            _transcriberLock.Release();
        }
    }

    /// <summary>
    /// Opens the model this machine will use before anything asks for it.
    /// <para>
    /// A QNN context binary takes several seconds to load, and doing it on the click meant the
    /// first recording of a session began with a wait, and the first words after it were spoken
    /// into a microphone that was not running yet. Startup has that time going spare.
    /// </para>
    /// <para>
    /// The cost is memory: the model stays resident from launch rather than from first use. Set
    /// LOCALSCRIBE_NO_PRELOAD to trade the seconds back.
    /// </para>
    /// </summary>
    public async Task PreloadAsync()
    {
        if (_plan is null || Environment.GetEnvironmentVariable("LOCALSCRIBE_NO_PRELOAD") is { Length: > 0 })
        {
            return;
        }

        // The listening plan, because that is the path where waiting costs words rather than
        // patience. A file transcription reuses it when the weights resolve the same.
        var livePlan = AcceleratorPlanner.Plan(
            _capabilities ?? DeviceCapabilities.Unknown,
            PerformanceProfile.Considerate,
            WorkloadMode.Live);

        IsWarmingUp = true;
        Status = "Warming up…";

        try
        {
            var transcriber = await Task.Run(() => TranscriberFor(livePlan));

            IsModelReady = true;
            Status = $"Ready. {transcriber.Description}";
        }
        catch (Exception exception)
        {
            // Not fatal. The model will be opened on demand, slowly, and the failure will be
            // reported then with the context of what the user was trying to do.
            Status = $"Ready, but the model could not be preloaded: {exception.Message}";
        }
        finally
        {
            IsWarmingUp = false;
        }
    }

    /// <summary>
    /// Where this plan's weights live.
    /// <para>
    /// Keyed on the planned device, not the chip: a Snapdragon that ended up on the CPU needs
    /// the portable export, not the chipset binaries it does not have. And located rather than
    /// resolved, because the plan names a Whisper size while what is installed is whatever the
    /// user obtained — NPU builds arrive under their own names, and demanding an exact match
    /// hides a working model and sends the work to the CPU.
    /// </para>
    /// </summary>
    private string ModelDirectoryFor(ExecutionPlan plan)
    {
        var family = _capabilities?.Family ?? SocFamily.Unknown;

        return ModelLayout.Locate(_modelRoot, family, plan.Encoder.Device, plan.WhisperModel)
            ?? throw new DirectoryNotFoundException(
                $"No Whisper weights under {Path.Combine(_modelRoot, ModelLayout.FolderFor(family, plan.Encoder.Device))}. "
                + "Run 'localscribe-doctor --fetch-models' to download a portable set.");
    }

    /// <summary>Stops listening and commits the trailing words.</summary>
    public async Task StopRecordingAsync()
    {
        if (!IsRecording || _liveSession is null)
        {
            return;
        }

        IsRecording = false;

        // Unsubscribe before stopping. StopRecording does not promise that no further buffers
        // arrive — ones already captured still raise the event — and a handler that is mid-await
        // is running regardless.
        _microphone!.SamplesAvailable -= OnSamplesAvailable;
        _microphone.Stop();
        _microphone.Dispose();
        _microphone = null;

        var session = _liveSession;

        // Cleared first so any callback still in flight sees no session and returns rather than
        // pushing audio into one that is being torn down.
        _liveSession = null;

        try
        {
            var committed = await session.FinishAsync();
            SetTranscript(committed);
            ProvisionalText = string.Empty;

            // The recording becomes playable the moment it stops, on the same path a
            // transcribed file takes.
            if (_liveCapture is { Count: > 0 } captured)
            {
                _audio = new PcmAudio([.. captured]);
                _segmentsBeforeSpeakers = committed;

                Player.Load(_audio);
            }

            _liveCapture = null;

            if (_diagnostics is not null)
            {
                _diagnostics.Finished(committed);
                Status = $"Stopped. Diagnostics written to {_diagnostics.Directory}";
                _diagnostics = null;
            }
            else
            {
                Status = "Stopped.";
            }
        }
        catch (OperationCanceledException)
        {
            Status = "Stopped.";
        }
        finally
        {
            await session.DisposeAsync();

            _cancellation?.Dispose();
            _cancellation = null;
        }
    }

    private async void OnSamplesAvailable(float[] samples)
    {
        var session = _liveSession;
        if (session is null)
        {
            return;
        }

        try
        {
            _diagnostics?.Captured(samples);
            _liveCapture?.AddRange(samples);

            var update = await session.PushAsync(samples, _cancellation?.Token ?? default);
            if (update is not null)
            {
                ProvisionalText = update.Text;
                SetTranscript(session.CommittedSegments);

                _diagnostics?.Pass(update.Text, session.CommittedSegments);
            }
        }
        catch (OperationCanceledException)
        {
            // Expected when the user stops recording mid-pass.
        }
        catch (Exception exception)
        {
            Status = $"Live transcription failed: {exception.Message}";
        }
    }

    /// <summary>
    /// The backend found at startup, reused rather than reconnected per pass. Null when none is
    /// running, which disables the cleanup stage without disabling transcription.
    /// </summary>
    private ILanguageModel? _languageModel;

    private TranscriptRefiner? BuildRefiner() =>
        _languageModel is null ? null : new TranscriptRefiner(_languageModel);

    /// <summary>
    /// The cleanup model's summary and action items, shown above the transcript rather than
    /// spliced into it. They are about the recording; the transcript is the recording.
    /// </summary>
    private static string Format(TranscriptionResult result)
    {
        var builder = new StringBuilder();

        if (result.Refinement?.Summary is { Length: > 0 } summary)
        {
            builder.AppendLine("Summary").AppendLine(summary);
        }

        if (result.Refinement?.ActionItems is { Count: > 0 } actions)
        {
            if (builder.Length > 0)
            {
                builder.AppendLine();
            }

            builder.AppendLine("Action items");
            foreach (var action in actions)
            {
                builder.AppendLine($"- {action}");
            }
        }

        return builder.ToString().TrimEnd();
    }

    /// <summary>Cancels whatever is running.</summary>
    public void Cancel() => _cancellation?.Cancel();

    /// <summary>
    /// Releases the cached transcriber and anything still capturing. The transcriber holds ONNX
    /// sessions and, on the NPU path, a context binary well over a gigabyte, so it is worth
    /// letting go of rather than leaving to process exit.
    /// </summary>
    public void Dispose()
    {
        _microphone?.Dispose();
        _microphone = null;

        _transcriber?.Dispose();
        _transcriber = null;
        _transcriberKey = null;
        _transcriberLock.Dispose();

        _cancellation?.Dispose();
        _cancellation = null;

        (_languageModel as IDisposable)?.Dispose();
        _languageModel = null;
    }

    private void Raise(string name) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));

    private void Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return;
        }

        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
