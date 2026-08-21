namespace LocalScribe.Core.Hardware;

/// <summary>
/// Turns observed hardware into an execution plan.
/// <para>
/// The guiding rule comes from how these laptops are actually used: prefer the NPU and GPU
/// not because they are always faster, but because work placed there does not compete with
/// the user's other apps. A transcription that finishes in four minutes without anyone
/// noticing beats one that finishes in three while the machine stutters.
/// </para>
/// <para>
/// This class is deliberately pure. It touches no hardware and no files, so the whole policy
/// can be tested on any machine.
/// </para>
/// </summary>
public static class AcceleratorPlanner
{
    /// <summary>ONNX Runtime's name for the Qualcomm AI Engine Direct provider.</summary>
    public const string QnnProvider = "QNN";

    /// <summary>ONNX Runtime's name for the DirectML provider, which reaches the Adreno GPU.</summary>
    public const string DirectMlProvider = "DML";

    /// <summary>
    /// Fraction of the performance cores we are willing to occupy in
    /// <see cref="PerformanceProfile.Considerate"/> mode while plugged in.
    /// Two-thirds keeps a 12-core Snapdragon X Elite at eight busy cores and four free.
    /// </summary>
    private const double ConsiderateCoreShare = 0.66;

    /// <summary>The same share on battery, where thermal and power headroom are tighter.</summary>
    private const double BatteryCoreShare = 0.4;

    /// <summary>
    /// Threads needed when every model stage runs off-CPU. All that remains on the CPU is
    /// audio decoding and mel-spectrogram work, which two threads absorb comfortably.
    /// </summary>
    private const int OffloadedCpuThreads = 2;

    public static ExecutionPlan Plan(
        DeviceCapabilities caps,
        PerformanceProfile profile = PerformanceProfile.Considerate,
        WorkloadMode mode = WorkloadMode.Batch,
        bool strictProviderCheck = false)
    {
        ArgumentNullException.ThrowIfNull(caps);

        var warnings = new List<string>();
        var encoder = PlanEncoder(caps, warnings);
        var decoder = PlanDecoder(caps, encoder);
        var languageModel = PlanLanguageModel(caps, warnings);

        var everythingOffloaded =
            encoder.Device != ComputeDevice.Cpu && decoder.Device != ComputeDevice.Cpu;

        return new ExecutionPlan
        {
            Encoder = encoder,
            Decoder = decoder,
            LanguageModel = languageModel,
            CpuBudget = PlanCpuBudget(caps, profile, everythingOffloaded),
            WhisperModel = ChooseWhisperModel(caps, encoder.Device, mode),
            StrictProviderCheck = strictProviderCheck,
            Warnings = warnings,
        };
    }

    /// <summary>
    /// The encoder is the expensive half of Whisper and takes a fixed-shape input, which is
    /// exactly the workload the Hexagon NPU handles well. This is the stage worth fighting for.
    /// </summary>
    private static StagePlan PlanEncoder(DeviceCapabilities caps, List<string> warnings)
    {
        if (caps.NpuUsable)
        {
            return new StagePlan(
                ComputeDevice.Npu,
                QnnProvider,
                "Hexagon NPU via QNN. The encoder runs off-CPU, so the rest of the system stays free.");
        }

        CollectNpuWarnings(caps, warnings);

        if (caps.DirectMlPresent)
        {
            return new StagePlan(
                ComputeDevice.Gpu,
                DirectMlProvider,
                "Adreno GPU via DirectML. Slower than the NPU but still keeps the CPU mostly idle.");
        }

        warnings.Add("No accelerator available. Transcription will run on the CPU and take noticeably longer.");
        return new StagePlan(ComputeDevice.Cpu, null, "CPU. No NPU or GPU path was usable on this machine.");
    }

    /// <summary>
    /// The decoder runs one short autoregressive step at a time. Per-call dispatch overhead
    /// dominates, so a GPU is usually a net loss here even when the encoder gains from one.
    /// We follow the NPU when it is available and otherwise stay on the CPU.
    /// </summary>
    private static StagePlan PlanDecoder(DeviceCapabilities caps, StagePlan encoder)
    {
        if (encoder.Device == ComputeDevice.Npu)
        {
            return new StagePlan(
                ComputeDevice.Npu,
                QnnProvider,
                "Hexagon NPU via QNN, using the precompiled decoder graph that ships beside the encoder.");
        }

        return new StagePlan(
            ComputeDevice.Cpu,
            null,
            "CPU. The decoder issues many small steps, and dispatch overhead outweighs any GPU gain.");
    }

    /// <summary>
    /// The cleanup model is optional. When Foundry Local is running we hand the work to it,
    /// since it picks its own hardware variant. Otherwise the pipeline still produces a
    /// transcript, just without punctuation repair or a summary.
    /// </summary>
    private static StagePlan PlanLanguageModel(DeviceCapabilities caps, List<string> warnings)
    {
        if (!caps.FoundryLocalPresent)
        {
            warnings.Add(
                "Foundry Local is not reachable. Transcripts will be raw: no punctuation repair, " +
                "glossary correction, or summary. Run 'foundry service start' to enable them.");
            return new StagePlan(ComputeDevice.Cpu, null, "Disabled. No local language-model service was found.");
        }

        if (caps.NpuUsable)
        {
            return new StagePlan(
                ComputeDevice.Npu,
                null,
                "Foundry Local on the NPU. Cleanup runs off-CPU alongside transcription.");
        }

        return new StagePlan(
            ComputeDevice.Cpu,
            null,
            "Foundry Local on the CPU. The Oryon cores handle a small model at reading speed.");
    }

    /// <summary>
    /// Explains precisely which of the three NPU prerequisites is missing. Being specific here
    /// saves a great deal of guesswork, because the failure modes look identical from outside.
    /// </summary>
    private static void CollectNpuWarnings(DeviceCapabilities caps, List<string> warnings)
    {
        if (caps.Family is SocFamily.NonQualcomm)
        {
            warnings.Add("This is not a Qualcomm processor, so there is no QNN path. Falling back.");
            return;
        }

        if (!caps.QnnProviderPresent)
        {
            warnings.Add(
                "The QNN execution provider did not load. Check that the ONNX Runtime QNN package " +
                "is installed and that the app is running as native arm64, not under x64 emulation.");
        }
        else if (!caps.HexagonDriverPresent)
        {
            warnings.Add(
                "The Hexagon NPU runtime driver is missing. This is a separate install from the driver " +
                "Windows ships with. Get it from Qualcomm Software Center, then restart LocalScribe.");
        }
        else if (!caps.WhisperQnnAssetsPresent)
        {
            warnings.Add(
                $"No Whisper QNN model assets found for {caps.SocName}. Precompiled binaries are " +
                "chipset-specific. LocalScribe offers to download a matching set during setup.");
        }
    }

    /// <summary>
    /// Sizes the CPU thread pool. When the models are off-CPU this stays small on purpose:
    /// spending more threads on mel-spectrogram work would only add scheduler pressure.
    /// </summary>
    private static CpuBudget PlanCpuBudget(
        DeviceCapabilities caps,
        PerformanceProfile profile,
        bool everythingOffloaded)
    {
        var cores = Math.Max(1, caps.PerformanceCoreCount);

        if (profile == PerformanceProfile.Maximum)
        {
            var maxThreads = everythingOffloaded ? OffloadedCpuThreads : cores;
            return new CpuBudget(maxThreads, 1, BelowNormalPriority: false);
        }

        if (everythingOffloaded)
        {
            return new CpuBudget(Math.Min(OffloadedCpuThreads, cores), 1, BelowNormalPriority: true);
        }

        var share = caps.OnBattery ? BatteryCoreShare : ConsiderateCoreShare;

        // Two threads is the floor for reasonable throughput, but never more threads than the
        // machine has cores. Order matters here: a two-core machine wants 2, not a clamp whose
        // lower bound has climbed above its upper one.
        var target = (int)Math.Floor(cores * share);
        var threads = Math.Min(cores, Math.Max(2, target));
        return new CpuBudget(threads, 1, BelowNormalPriority: true);
    }

    /// <summary>
    /// Picks Whisper weights against three limits at once: how much memory the machine has,
    /// whether the encoder is offloaded, and whether we owe the user low latency.
    /// </summary>
    private static string ChooseWhisperModel(DeviceCapabilities caps, ComputeDevice encoderDevice, WorkloadMode mode)
    {
        var memoryGib = caps.TotalMemoryGib;

        if (encoderDevice == ComputeDevice.Npu)
        {
            // Off-CPU work is close to free from the user's point of view, so we can afford
            // a larger model than the CPU path would tolerate.
            if (mode == WorkloadMode.Live)
            {
                return memoryGib >= 16 ? "small.en" : "base.en";
            }

            return memoryGib >= 16 ? "medium.en" : "small.en";
        }

        if (mode == WorkloadMode.Live)
        {
            // Live on the CPU is the tightest budget in the app. Anything larger than base
            // drifts behind the speaker and never catches up.
            return memoryGib >= 16 ? "base.en" : "tiny.en";
        }

        return memoryGib >= 16 ? "small.en" : "base.en";
    }
}
