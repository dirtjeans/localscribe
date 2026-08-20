using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text;
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
    private ITranscriber? _liveTranscriber;
    private string? _liveTranscriberKey;

    private string _status = "Starting up…";
    private string _hardwareSummary = string.Empty;
    private string _transcript = string.Empty;
    private string _provisionalText = string.Empty;
    private double _progress;
    private bool _busy;
    private bool _recording;
    private bool _preparing;

    public MainViewModel(string? modelRoot = null)
    {
        _modelRoot = modelRoot ?? Path.Combine(AppContext.BaseDirectory, "models");
    }

    public event PropertyChangedEventHandler? PropertyChanged;

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
        Transcript = string.Empty;
        _cancellation = new CancellationTokenSource();

        try
        {
            Status = "Loading audio…";
            var audio = await Task.Run(() => AudioFileLoader.Load(path), _cancellation.Token);

            Status = "Loading the model…";
            using var transcriber = await Task.Run(() => OpenTranscriber(_plan), _cancellation.Token);
            var pipeline = new TranscriptionPipeline(transcriber, BuildRefiner());

            Status = $"Transcribing with {transcriber.Description}…";

            var progress = new Progress<TranscriptionProgress>(update =>
            {
                Progress = update.Fraction;
                Status = $"Transcribing… {update.ChunksCompleted} of {update.ChunksTotal} windows";
            });

            var result = await pipeline.RunAsync(
                audio,
                Glossary,
                RefinementOutputs.Everything,
                progress,
                _cancellation.Token);

            Transcript = Format(result);
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
            transcriber = await Task.Run(() => LiveTranscriber(livePlan));
        }
        catch (Exception exception)
        {
            IsPreparing = false;
            Status = $"Could not start listening: {exception.Message}";
            return;
        }

        _liveSession = new LiveTranscriptionSession(transcriber);
        _cancellation = new CancellationTokenSource();

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
    private ITranscriber LiveTranscriber(ExecutionPlan plan)
    {
        var key = $"{plan.Encoder.Device}/{plan.Decoder.Device}/{plan.WhisperModel}";

        if (_liveTranscriber is not null && _liveTranscriberKey == key)
        {
            return _liveTranscriber;
        }

        _liveTranscriber?.Dispose();
        _liveTranscriber = OpenTranscriber(plan);
        _liveTranscriberKey = key;

        return _liveTranscriber;
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
            Transcript = new Transcript(committed).FullText;
            ProvisionalText = string.Empty;
            Status = "Stopped.";
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
            var update = await session.PushAsync(samples, _cancellation?.Token ?? default);
            if (update is not null)
            {
                ProvisionalText = update.Text;
                Transcript = new Transcript(session.CommittedSegments).FullText;
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

    private ITranscriber OpenTranscriber(ExecutionPlan plan)
    {
        var family = _capabilities?.Family ?? SocFamily.Unknown;

        // Keyed on the planned device, not the chip. A Snapdragon that ended up on the CPU
        // needs the portable export, not the chipset binaries it does not have.
        //
        // Locate rather than Resolve: the plan names a Whisper size, but what is installed is
        // whatever the user obtained, and NPU builds arrive under their own names. Demanding an
        // exact match hides a working model and sends the work to the CPU.
        var directory = ModelLayout.Locate(_modelRoot, family, plan.Encoder.Device, plan.WhisperModel)
            ?? throw new DirectoryNotFoundException(
                $"No Whisper weights under {Path.Combine(_modelRoot, ModelLayout.FolderFor(family, plan.Encoder.Device))}. "
                + "Run 'localscribe-doctor --fetch-models' to download a portable set.");

        return WhisperOnnxTranscriber.Load(directory, plan);
    }

    /// <summary>
    /// The backend found at startup, reused rather than reconnected per pass. Null when none is
    /// running, which disables the cleanup stage without disabling transcription.
    /// </summary>
    private ILanguageModel? _languageModel;

    private TranscriptRefiner? BuildRefiner() =>
        _languageModel is null ? null : new TranscriptRefiner(_languageModel);

    private static string Format(TranscriptionResult result)
    {
        var builder = new StringBuilder();

        if (result.Refinement?.Summary is { Length: > 0 } summary)
        {
            builder.AppendLine("Summary").AppendLine(summary).AppendLine();
        }

        if (result.Refinement?.ActionItems is { Count: > 0 } actions)
        {
            builder.AppendLine("Action items");
            foreach (var action in actions)
            {
                builder.AppendLine($"- {action}");
            }

            builder.AppendLine();
        }

        builder.AppendLine("Transcript").AppendLine(result.BestText);
        return builder.ToString();
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

        _liveTranscriber?.Dispose();
        _liveTranscriber = null;
        _liveTranscriberKey = null;

        _cancellation?.Dispose();
        _cancellation = null;

        (_languageModel as IDisposable)?.Dispose();
        _languageModel = null;
    }

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
