namespace LocalScribe.Core.Hardware;

/// <summary>How ready an engine is on this machine.</summary>
public enum EngineAvailability
{
    /// <summary>Everything it needs is present. It can run now.</summary>
    Ready,

    /// <summary>The hardware supports it, but something has to be installed or fetched first.</summary>
    NeedsSetup,

    /// <summary>This machine cannot run it at all.</summary>
    Unsupported,
}

/// <summary>One way of running transcription, and what it would mean here.</summary>
/// <param name="Name">What to call it, e.g. "ONNX Runtime on the Hexagon NPU".</param>
/// <param name="Device">Where the encoder would run.</param>
/// <param name="ModelSize">The Whisper size this engine could afford.</param>
/// <param name="Availability">Whether it can run now, could after setup, or not at all.</param>
/// <param name="Rationale">Why it ranks where it does on this machine.</param>
/// <param name="Requirements">What is still missing. Empty when <see cref="Availability"/> is Ready.</param>
public sealed record EngineOption(
    string Name,
    ComputeDevice Device,
    string ModelSize,
    EngineAvailability Availability,
    string Rationale,
    IReadOnlyList<string> Requirements);

/// <summary>The advice as a whole.</summary>
/// <param name="Recommended">The best engine that can run right now. Never null: the CPU always can.</param>
/// <param name="Better">
/// A better engine this hardware supports but that is not set up yet, or null when the
/// recommendation is already the best this machine could do.
/// </param>
/// <param name="All">Every engine considered, best first.</param>
public sealed record EngineAdvice(
    EngineOption Recommended,
    EngineOption? Better,
    IReadOnlyList<EngineOption> All)
{
    /// <summary>One line naming the engine to use, and what is being left on the table.</summary>
    public string Summary => Better is null
        ? $"Use {Recommended.Name} with Whisper {Recommended.ModelSize}."
        : $"Use {Recommended.Name} with Whisper {Recommended.ModelSize} for now. "
          + $"{Better.Name} would be the better engine on this machine.";
}

/// <summary>
/// Recommends which engine to transcribe with on a given machine.
/// <para>
/// <see cref="AcceleratorPlanner"/> answers "given what is installed, where does the work go".
/// This answers the question a user asks first, which is "what should I be running, and what
/// would I gain by setting something up". They share a ranking rule, so the two can never
/// contradict each other.
/// </para>
/// <para>
/// That rule is the one the whole project runs on: prefer the processor that leaves the machine
/// usable, not the one that finishes first. The NPU wins not because it beats the Oryon cores on
/// raw throughput — often it does not — but because work placed there is work the user never
/// feels.
/// </para>
/// <para>
/// Pure, like the planner. No hardware, no files, no network, so the whole policy is testable.
/// </para>
/// </summary>
public static class TranscriptionEngineAdvisor
{
    /// <summary>Ranks the engines for this machine and picks one.</summary>
    public static EngineAdvice Advise(DeviceCapabilities caps, WorkloadMode mode = WorkloadMode.Batch)
    {
        ArgumentNullException.ThrowIfNull(caps);

        // Declared best-first. Everything below relies on that order, and it is the same
        // preference the planner encodes.
        var ranked = new List<EngineOption>
        {
            Npu(caps, mode),
            Gpu(caps, mode),
            Cpu(caps, mode),
        };

        var recommended = ranked.First(o => o.Availability == EngineAvailability.Ready);

        // Only engines ranked above the recommendation count as "better". One that needs setup
        // but would still be slower is not worth sending anyone to go and install.
        var better = ranked
            .TakeWhile(o => o != recommended)
            .FirstOrDefault(o => o.Availability == EngineAvailability.NeedsSetup);

        return new EngineAdvice(recommended, better, ranked);
    }

    private static EngineOption Npu(DeviceCapabilities caps, WorkloadMode mode)
    {
        var size = AcceleratorPlanner.ChooseWhisperModel(caps, ComputeDevice.Npu, mode);

        if (caps.Family is SocFamily.NonQualcomm or SocFamily.Unknown)
        {
            return new EngineOption(
                "ONNX Runtime on the Hexagon NPU",
                ComputeDevice.Npu,
                size,
                EngineAvailability.Unsupported,
                "Needs Qualcomm silicon, which this machine does not have.",
                []);
        }

        var missing = new List<string>();

        if (!caps.QnnProviderPresent)
        {
            missing.Add(
                "The QNN execution provider is not registered. Build and run as native arm64: "
                + "under x64 emulation it cannot load at all.");
        }

        if (!caps.HexagonDriverPresent)
        {
            missing.Add(
                "The Hexagon NPU Runtime Driver is missing. It is a separate download from "
                + "Qualcomm Software Center, not the driver Windows ships with.");
        }

        if (!caps.WhisperQnnAssetsPresent)
        {
            missing.Add(
                "No precompiled QNN weights for this chipset. They are built per chip and come "
                + "from Qualcomm AI Hub; see docs/setup-snapdragon.md.");
        }

        return new EngineOption(
            "ONNX Runtime on the Hexagon NPU",
            ComputeDevice.Npu,
            size,
            missing.Count == 0 ? EngineAvailability.Ready : EngineAvailability.NeedsSetup,
            "The encoder runs off-CPU entirely, so a long transcription costs almost no "
            + "responsiveness, and the larger model it affords is more accurate than anything the "
            + "CPU path would tolerate.",
            missing);
    }

    private static EngineOption Gpu(DeviceCapabilities caps, WorkloadMode mode)
    {
        var size = AcceleratorPlanner.ChooseWhisperModel(caps, ComputeDevice.Gpu, mode);

        // DirectML reaches any DirectX 12 device, so unlike the NPU this is not Qualcomm-only.
        // Whether it registered is the one thing we can say without probing further.
        return new EngineOption(
            "ONNX Runtime on the GPU via DirectML",
            ComputeDevice.Gpu,
            size,
            caps.DirectMlPresent ? EngineAvailability.Ready : EngineAvailability.NeedsSetup,
            "Slower than the NPU, and it does occupy the GPU, but it still leaves the CPU mostly "
            + "free — which is what decides whether the machine feels busy.",
            caps.DirectMlPresent
                ? []
                : new[]
                {
                    "The DirectML execution provider is not registered. Add the "
                    + "Microsoft.ML.OnnxRuntime.DirectML package and rebuild.",
                });
    }

    private static EngineOption Cpu(DeviceCapabilities caps, WorkloadMode mode)
    {
        var size = AcceleratorPlanner.ChooseWhisperModel(caps, ComputeDevice.Cpu, mode);

        // Unconditionally Ready, and deliberately so: it is what makes the recommendation total.
        // Every other branch may decline; this one may not.
        return new EngineOption(
            "ONNX Runtime on the CPU",
            ComputeDevice.Cpu,
            size,
            EngineAvailability.Ready,
            "Always works and needs nothing installed. It is the slowest option and the one the "
            + "user feels most, so the model size steps down to keep the machine usable.",
            []);
    }
}
