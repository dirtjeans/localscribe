namespace LocalScribe.Core.Hardware;

/// <summary>
/// Which Snapdragon generation we are running on. The distinction matters because
/// Qualcomm ships <em>precompiled</em> QNN context binaries per chipset, and a binary
/// built for one SoC will not load on another.
/// </summary>
public enum SocFamily
{
    /// <summary>Probe could not identify the SoC. Treat as CPU-only.</summary>
    Unknown = 0,

    /// <summary>Not a Qualcomm part (Intel, AMD, Apple). No QNN path exists.</summary>
    NonQualcomm,

    /// <summary>Snapdragon X Plus (X1P series).</summary>
    SnapdragonXPlus,

    /// <summary>Snapdragon X Elite (X1E series).</summary>
    SnapdragonXElite,

    /// <summary>Snapdragon X2 series.</summary>
    SnapdragonX2,
}

/// <summary>
/// A snapshot of what the current machine can actually do, as observed at startup.
/// <para>
/// Every field is something we <em>measured</em>, not something we assumed. The planner
/// (<see cref="AcceleratorPlanner"/>) turns this into a decision; keeping the two apart
/// is what makes the policy testable without a Snapdragon in the room.
/// </para>
/// </summary>
public sealed record DeviceCapabilities
{
    /// <summary>Human-readable processor name, straight from the OS.</summary>
    public string SocName { get; init; } = "unknown";

    public SocFamily Family { get; init; } = SocFamily.Unknown;

    /// <summary>
    /// Count of high-performance cores. On Snapdragon X Elite this is 12 Oryon cores;
    /// X Plus ships 8 or 10. Used to size the CPU thread budget.
    /// </summary>
    public int PerformanceCoreCount { get; init; }

    /// <summary>Total logical processors reported by the OS.</summary>
    public int TotalCoreCount { get; init; }

    /// <summary>Installed physical memory. Snapdragon X uses a unified pool shared with the NPU.</summary>
    public long TotalMemoryBytes { get; init; }

    /// <summary>
    /// Execution providers ONNX Runtime reports as registered. Note that a provider
    /// appearing here means the DLL loaded, not that a model will run on it.
    /// </summary>
    public IReadOnlySet<string> OnnxProviders { get; init; } = new HashSet<string>();

    /// <summary>
    /// True when the QNN execution provider libraries resolved. Without the separate
    /// Hexagon NPU runtime driver this can be true while the NPU stays unusable.
    /// </summary>
    public bool QnnProviderPresent { get; init; }

    /// <summary>
    /// True when the Qualcomm Hexagon NPU runtime driver is installed. This is a distinct
    /// install from the display driver Windows ships with, and is the single most common
    /// reason NPU inference silently falls back to CPU.
    /// </summary>
    public bool HexagonDriverPresent { get; init; }

    /// <summary>True when a chipset-matched Whisper QNN context binary is on disk.</summary>
    public bool WhisperQnnAssetsPresent { get; init; }

    /// <summary>True when DirectML is available, which reaches the Adreno GPU.</summary>
    public bool DirectMlPresent { get; init; }

    /// <summary>True when a Foundry Local service is reachable for the language-model stage.</summary>
    public bool FoundryLocalPresent { get; init; }

    /// <summary>True when the machine is running on battery rather than mains power.</summary>
    public bool OnBattery { get; init; }

    /// <summary>
    /// The NPU is only genuinely usable when the provider, the driver, and a chipset-matched
    /// model asset are all present. Any one missing means CPU work dressed up as NPU work.
    /// </summary>
    public bool NpuUsable => QnnProviderPresent && HexagonDriverPresent && WhisperQnnAssetsPresent;

    /// <summary>Total memory in gibibytes, rounded down.</summary>
    public int TotalMemoryGib => (int)(TotalMemoryBytes / (1024L * 1024 * 1024));

    /// <summary>A conservative stand-in used by tests and by non-Windows hosts.</summary>
    public static DeviceCapabilities Unknown { get; } = new()
    {
        SocName = "unknown",
        Family = SocFamily.Unknown,
        PerformanceCoreCount = 4,
        TotalCoreCount = 4,
        TotalMemoryBytes = 8L * 1024 * 1024 * 1024,
    };
}
