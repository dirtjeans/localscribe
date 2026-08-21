namespace LocalScribe.Core.Hardware;

/// <summary>Where a single stage of the pipeline will run.</summary>
public enum ComputeDevice
{
    Cpu = 0,

    /// <summary>Adreno GPU, reached through DirectML.</summary>
    Gpu,

    /// <summary>Hexagon NPU, reached through the QNN execution provider.</summary>
    Npu,
}

/// <summary>
/// How aggressively we are allowed to use the machine.
/// </summary>
public enum PerformanceProfile
{
    /// <summary>
    /// Default. Prefer accelerators over the CPU, and when the CPU is unavoidable, leave
    /// enough of it free that the rest of the desktop stays responsive.
    /// </summary>
    Considerate = 0,

    /// <summary>Use everything available. Transcription finishes sooner, the machine feels slower.</summary>
    Maximum,
}

/// <summary>Batch transcription and live transcription want different tradeoffs.</summary>
public enum WorkloadMode
{
    /// <summary>A file on disk. Accuracy matters more than latency.</summary>
    Batch = 0,

    /// <summary>A live microphone. Latency caps how large a model we can afford.</summary>
    Live,
}

/// <summary>The decision for one pipeline stage, plus why we made it.</summary>
/// <param name="Device">Which processor runs this stage.</param>
/// <param name="ExecutionProvider">
/// The ONNX Runtime execution provider name to register, or <c>null</c> for the built-in CPU provider.
/// </param>
/// <param name="Reason">Plain-language justification, surfaced in the app's setup and status panels.</param>
public sealed record StagePlan(ComputeDevice Device, string? ExecutionProvider, string Reason);

/// <summary>
/// How much CPU the pipeline may consume. This is the knob that keeps transcription from
/// making the rest of Windows feel sticky.
/// </summary>
/// <param name="IntraOpThreads">Threads ONNX Runtime may use inside a single operator.</param>
/// <param name="InterOpThreads">
/// Threads for running independent operators concurrently. Held at 1 because Whisper's graph is
/// essentially sequential, so extra threads here buy contention rather than speed.
/// </param>
/// <param name="BelowNormalPriority">
/// When true, worker threads drop below normal priority so foreground apps win the scheduler.
/// </param>
public sealed record CpuBudget(int IntraOpThreads, int InterOpThreads, bool BelowNormalPriority);

/// <summary>
/// The full set of decisions for one run: what goes where, how much CPU we may take,
/// which Whisper weights to load, and anything the user should know about.
/// </summary>
public sealed record ExecutionPlan
{
    public required StagePlan Encoder { get; init; }

    public required StagePlan Decoder { get; init; }

    public required StagePlan LanguageModel { get; init; }

    public required CpuBudget CpuBudget { get; init; }

    /// <summary>Whisper variant to load, e.g. <c>base.en</c>.</summary>
    public required string WhisperModel { get; init; }

    /// <summary>
    /// When true, ONNX Runtime is told to throw rather than quietly fall back to the CPU.
    /// Used by setup to prove the NPU is real; left off during transcription so a driver
    /// problem degrades instead of crashing.
    /// </summary>
    public bool StrictProviderCheck { get; init; }

    /// <summary>Things worth telling the user: missing drivers, absent model assets, CPU-only fallback.</summary>
    public IReadOnlyList<string> Warnings { get; init; } = [];

    /// <summary>True when nothing at all was offloaded and the CPU is doing every stage.</summary>
    public bool IsCpuOnly =>
        Encoder.Device == ComputeDevice.Cpu &&
        Decoder.Device == ComputeDevice.Cpu &&
        LanguageModel.Device == ComputeDevice.Cpu;

    /// <summary>One-line summary for the status bar.</summary>
    public string Summary =>
        $"Whisper {WhisperModel}: encoder on {Encoder.Device}, decoder on {Decoder.Device}, " +
        $"cleanup on {LanguageModel.Device} ({CpuBudget.IntraOpThreads} CPU threads)";
}
