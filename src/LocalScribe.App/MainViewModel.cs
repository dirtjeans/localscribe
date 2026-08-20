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
public sealed class MainViewModel : INotifyPropertyChanged
{
    private readonly string _modelRoot;
    private ExecutionPlan? _plan;
    private DeviceCapabilities? _capabilities;
    private CancellationTokenSource? _cancellation;
    private MicrophoneCapture? _microphone;
    private LiveTranscriptionSession? _liveSession;

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

        using var foundry = new FoundryLocalClient();
        capabilities = capabilities with { FoundryLocalPresent = await foundry.IsAvailableAsync() };

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

            using var transcriber = OpenTranscriber(_plan);
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

        var transcriber = OpenTranscriber(livePlan);
        _liveSession = new LiveTranscriptionSession(transcriber);
        _cancellation = new CancellationTokenSource();

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
        Status = "Stopped.";
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
        var directory = ModelLayout.Resolve(_modelRoot, family, plan.Encoder.Device, plan.WhisperModel);
        return WhisperOnnxTranscriber.Load(directory, plan);
    }

    private TranscriptRefiner? BuildRefiner() =>
        _capabilities?.FoundryLocalPresent == true
            ? new TranscriptRefiner(new FoundryLocalClient())
            : null;

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
