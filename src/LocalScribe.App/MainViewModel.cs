using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Pipeline;
using LocalScribe.Core.Refinement;
using LocalScribe.Core.Transcription;
using LocalScribe.Onnx;

namespace LocalScribe.App;

/// <summary>
/// Drives the window. Holds no UI types, so the interesting behaviour can be reasoned about
/// (and later tested) without standing up a WinUI host.
/// </summary>
public sealed class MainViewModel : INotifyPropertyChanged
{
    private readonly string _modelRoot;
    private ExecutionPlan? _plan;
    private DeviceCapabilities? _capabilities;
    private CancellationTokenSource? _cancellation;
    private CancellationTokenSource? _liveCancellation;
    private MicrophoneCapture? _microphone;
    private LiveTranscriptionSession? _liveSession;

    // The session borrows the transcriber rather than owning it, matching TranscriptionPipeline.
    // Holding it here is what lets stopping a recording release the ONNX sessions it opened.
    private ITranscriber? _liveTranscriber;

    private string _status = "Starting up…";
    private string _hardwareSummary = string.Empty;
    private string _transcript = string.Empty;
    private string _provisionalText = string.Empty;
    private double _progress;
    private bool _busy;
    private bool _recording;

    public MainViewModel(string? modelRoot = null)
    {
        _modelRoot = modelRoot ?? Path.Combine(AppContext.BaseDirectory, "models");
        Setup = new SetupViewModel(_modelRoot);
    }

    /// <summary>
    /// What this machine is missing and how to get it. Owned here rather than by the window so
    /// the probe runs once and transcription reads the same answer setup reported.
    /// </summary>
    public SetupViewModel Setup { get; }

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

    /// <summary>Terms the cleanup model should spell correctly. Edited by the user in the UI.</summary>
    public List<string> Glossary { get; } = [];

    /// <summary>
    /// Probes the machine and works out the plan. Runs once at startup; the result is shown in
    /// the UI rather than hidden, because "is it using the NPU?" is the question everyone asks.
    /// </summary>
    public async Task InitialiseAsync()
    {
        Status = "Checking hardware…";

        await Setup.RefreshAsync();

        _capabilities = Setup.Capabilities;
        _plan = Setup.Plan;

        HardwareSummary = _plan?.Summary ?? "Could not read this machine's hardware.";

        Status = Setup.CanTranscribe
            ? "Ready."
            : "Setup needed before LocalScribe can transcribe.";
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

            using var transcriber = OpenTranscriber(_plan);

            // The refiner borrows the client, so the client has to outlive the run and be closed
            // by whoever opened it. It holds an HttpClient, which a per-file leak does notice.
            using FoundryLocalClient? languageModel = _capabilities?.FoundryLocalPresent == true
                ? new FoundryLocalClient(endpoint: Setup.FoundryEndpoint)
                : null;

            var refiner = languageModel is null ? null : new TranscriptRefiner(languageModel);
            var pipeline = new TranscriptionPipeline(transcriber, refiner);

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

    /// <summary>Starts live transcription from the microphone.</summary>
    public Task StartRecordingAsync()
    {
        if (_plan is null || IsRecording)
        {
            return Task.CompletedTask;
        }

        // Live work wants a smaller model than batch, so the plan is recomputed for this mode.
        var livePlan = AcceleratorPlanner.Plan(
            _capabilities ?? DeviceCapabilities.Unknown,
            PerformanceProfile.Considerate,
            WorkloadMode.Live);

        ITranscriber transcriber;
        try
        {
            transcriber = OpenTranscriber(livePlan);
        }
        catch (Exception exception)
        {
            // Loading throws when no model has been downloaded yet, which is the state every
            // machine starts in. Letting it escape an async void handler would close the app.
            Status = $"Could not start listening: {exception.Message}";
            return Task.CompletedTask;
        }

        _liveTranscriber = transcriber;
        _liveSession = new LiveTranscriptionSession(transcriber);
        _liveCancellation = new CancellationTokenSource();

        _microphone = new MicrophoneCapture();
        _microphone.SamplesAvailable += OnSamplesAvailable;
        _microphone.Start();

        IsRecording = true;
        Status = $"Listening with {transcriber.Description}…";
        return Task.CompletedTask;
    }

    /// <summary>Stops listening and commits the trailing words.</summary>
    public async Task StopRecordingAsync()
    {
        if (!IsRecording || _liveSession is null)
        {
            return;
        }

        IsRecording = false;
        _microphone!.Stop();
        _microphone.SamplesAvailable -= OnSamplesAvailable;
        _microphone.Dispose();
        _microphone = null;

        var committed = await _liveSession.FinishAsync();
        Transcript = new Transcript(committed).FullText;
        ProvisionalText = string.Empty;

        await _liveSession.DisposeAsync();
        _liveSession = null;

        _liveTranscriber?.Dispose();
        _liveTranscriber = null;

        // Cleared before disposal so an in-flight capture callback reads null rather than a
        // disposed source.
        var cancellation = _liveCancellation;
        _liveCancellation = null;
        cancellation?.Dispose();

        Status = "Stopped.";
    }

    private async void OnSamplesAvailable(float[] samples)
    {
        var session = _liveSession;
        var cancellation = _liveCancellation;
        if (session is null || cancellation is null)
        {
            return;
        }

        try
        {
            var update = await session.PushAsync(samples, cancellation.Token);
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
        var directory = Path.Combine(_modelRoot, DeviceProbe.AssetFolderFor(family), plan.WhisperModel);
        return WhisperOnnxTranscriber.Load(directory, plan);
    }

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
