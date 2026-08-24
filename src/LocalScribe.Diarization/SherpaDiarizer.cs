using LocalScribe.Core.Audio;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Provisioning;
using SherpaOnnx;

namespace LocalScribe.Diarization;

/// <summary>
/// Speaker diarization through sherpa-onnx, using pyannote's segmentation model.
/// <para>
/// <b>This runs on the CPU.</b> The weights are pyannote's, but the runtime is sherpa-onnx's own
/// bundled ONNX Runtime, which does not carry the QNN execution provider — and pyannote
/// segmentation is a variable-length graph, which the Hexagon backend would not take as-is
/// anyway. So the NPU stays reserved for the Whisper encoder, and this work lands on the same
/// CPU budget as everything else. That is why the thread count comes from the execution plan
/// rather than from the core count.
/// </para>
/// </summary>
public sealed class SherpaDiarizer : IDiarizer
{
    private readonly OfflineSpeakerDiarization _diarization;
    private readonly SpeakerAssigner _assigner;
    private readonly DiarizationOptions _options;
    private readonly int _threads;
    private bool _disposed;

    private SherpaDiarizer(
        OfflineSpeakerDiarization diarization,
        DiarizationOptions options,
        int threads)
    {
        _diarization = diarization;
        _options = options;
        _threads = threads;
        _assigner = new SpeakerAssigner(options);
    }

    /// <summary>
    /// Opens the segmentation and embedding models from a directory prepared by
    /// <see cref="DiarizationModelInstaller"/>.
    /// </summary>
    /// <param name="modelDirectory">Directory holding the two ONNX models.</param>
    /// <param name="plan">Execution plan, consulted only for its CPU budget.</param>
    /// <param name="options">Clustering and turn-shaping options.</param>
    public static SherpaDiarizer Load(
        string modelDirectory,
        ExecutionPlan plan,
        DiarizationOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(plan);

        var resolved = options ?? DiarizationOptions.Default;
        resolved.Validate();

        var segmentation = Path.Combine(modelDirectory, DiarizationModelInstaller.SegmentationFileName);
        var embedding = Path.Combine(modelDirectory, DiarizationModelInstaller.EmbeddingFileName);

        if (!File.Exists(segmentation) || !File.Exists(embedding))
        {
            throw new FileNotFoundException(
                $"Diarization models are missing from {modelDirectory}. Run "
                + "'localscribe-doctor --install' to fetch them.",
                segmentation);
        }

        // Diarization shares the machine with a transcription run, so it takes the same capped
        // budget rather than the whole CPU. Halving it again would be over-cautious: the two
        // stages run in sequence, not at once.
        var threads = Math.Max(1, plan.CpuBudget.IntraOpThreads);

        var config = new OfflineSpeakerDiarizationConfig();
        config.Segmentation.Pyannote.Model = segmentation;
        config.Segmentation.NumThreads = threads;
        config.Segmentation.Provider = "cpu";
        config.Embedding.Model = embedding;
        config.Embedding.NumThreads = threads;
        config.Embedding.Provider = "cpu";

        // A known speaker count beats a distance threshold whenever it is actually known, so
        // setting one switches clustering out of guess-how-many mode entirely.
        config.Clustering.NumClusters = resolved.SpeakerCount ?? -1;
        config.Clustering.Threshold = resolved.ClusteringThreshold;

        config.MinDurationOn = resolved.MinimumTurnSeconds;
        config.MinDurationOff = resolved.MinimumGapSeconds;

        return new SherpaDiarizer(new OfflineSpeakerDiarization(config), resolved, threads);
    }

    public string Description =>
        $"pyannote segmentation via sherpa-onnx on CPU ({_threads} threads), "
        + (_options.SpeakerCount is { } n ? $"{n} speakers" : "speaker count auto-detected");

    public Task<IReadOnlyList<SpeakerTurn>> DiarizeAsync(
        PcmAudio audio,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (audio.SampleRate != _diarization.SampleRate)
        {
            throw new ArgumentException(
                $"Diarization needs {_diarization.SampleRate} Hz audio but was given {audio.SampleRate} Hz.",
                nameof(audio));
        }

        return Task.Run(() => Diarize(audio, progress, cancellationToken), cancellationToken);
    }

    private IReadOnlyList<SpeakerTurn> Diarize(
        PcmAudio audio,
        IProgress<double>? progress,
        CancellationToken cancellationToken)
    {
        // The native API takes the entire recording in one array, because clustering has to
        // compare voices across the whole thing. An hour of 16 kHz mono is about 230 MB of
        // floats, which is worth knowing before someone feeds it a day of audio.
        var samples = audio.Samples;

        OfflineSpeakerDiarizationSegment[] raw;

        if (progress is null)
        {
            raw = _diarization.Process(samples);
        }
        else
        {
            raw = _diarization.ProcessWithCallback(
                samples,
                (processed, total, _) =>
                {
                    progress.Report(total == 0 ? 0 : processed / (double)total);

                    // The callback's return value is the native API's cancellation channel:
                    // anything non-zero asks it to stop.
                    return cancellationToken.IsCancellationRequested ? 1 : 0;
                },
                IntPtr.Zero);
        }

        cancellationToken.ThrowIfCancellationRequested();

        var turns = raw
            .Select(s => new SpeakerTurn(FormatSpeaker(s.Speaker), s.Start, s.End))
            .ToList();

        return _assigner.Consolidate(turns);
    }

    /// <summary>
    /// Turns the native zero-based speaker index into a label. One-based, because a transcript
    /// reading "Speaker 0" looks like a bug to everyone who is not a programmer.
    /// </summary>
    private static string FormatSpeaker(int index) => $"Speaker {index + 1}";

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _diarization.Dispose();
    }
}
