using LocalScribe.Core.Hardware;
using Xunit;

namespace LocalScribe.Core.Tests;

public class TranscriptionEngineAdvisorTests
{
    private static DeviceCapabilities Snapdragon(
        bool qnn = true,
        bool driver = true,
        bool assets = true,
        bool directMl = false,
        int memoryGib = 32) => new()
        {
            SocName = "Snapdragon(R) X 12-core X1E80100",
            Family = SocFamily.SnapdragonXElite,
            PerformanceCoreCount = 12,
            TotalCoreCount = 12,
            TotalMemoryBytes = (long)memoryGib * 1024 * 1024 * 1024,
            OnnxProviders = new HashSet<string>(),
            QnnProviderPresent = qnn,
            HexagonDriverPresent = driver,
            WhisperQnnAssetsPresent = assets,
            DirectMlPresent = directMl,
            LocalLanguageModelPresent = false,
            OnBattery = false,
        };

    [Fact]
    public void FullyConfiguredSnapdragonIsSentToTheNpu()
    {
        var advice = TranscriptionEngineAdvisor.Advise(Snapdragon());

        Assert.Equal(ComputeDevice.Npu, advice.Recommended.Device);
        Assert.Equal(EngineAvailability.Ready, advice.Recommended.Availability);

        // Nothing outranks it, so there is nothing to nag about.
        Assert.Null(advice.Better);
    }

    /// <summary>
    /// The case this advisor exists for. The machine can do better than it currently is, and
    /// saying only "you are on the CPU" would not tell anyone what to do about it.
    /// </summary>
    [Fact]
    public void AnUnconfiguredSnapdragonIsToldWhatItIsMissing()
    {
        var advice = TranscriptionEngineAdvisor.Advise(Snapdragon(assets: false));

        Assert.Equal(ComputeDevice.Cpu, advice.Recommended.Device);
        Assert.NotNull(advice.Better);
        Assert.Equal(ComputeDevice.Npu, advice.Better!.Device);
        Assert.Equal(EngineAvailability.NeedsSetup, advice.Better.Availability);
        Assert.Contains(advice.Better.Requirements, r => r.Contains("AI Hub", StringComparison.Ordinal));
    }

    [Fact]
    public void EachMissingPieceIsReportedSeparately()
    {
        var advice = TranscriptionEngineAdvisor.Advise(
            Snapdragon(qnn: false, driver: false, assets: false));

        var npu = advice.All.Single(o => o.Device == ComputeDevice.Npu);

        // Three independent problems, three independent fixes. Folding them into one verdict is
        // exactly what makes a silent CPU fallback hard to diagnose.
        Assert.Equal(3, npu.Requirements.Count);
        Assert.Contains(npu.Requirements, r => r.Contains("arm64", StringComparison.Ordinal));
        Assert.Contains(npu.Requirements, r => r.Contains("Hexagon", StringComparison.Ordinal));
        Assert.Contains(npu.Requirements, r => r.Contains("AI Hub", StringComparison.Ordinal));
    }

    [Fact]
    public void TheGpuIsPreferredOverTheCpuWhenTheNpuIsOut()
    {
        var advice = TranscriptionEngineAdvisor.Advise(
            Snapdragon(assets: false, directMl: true));

        Assert.Equal(ComputeDevice.Gpu, advice.Recommended.Device);
    }

    [Fact]
    public void NonQualcommHardwareIsToldTheNpuIsNotAnOptionRatherThanToInstallThings()
    {
        var caps = Snapdragon() with { Family = SocFamily.NonQualcomm, SocName = "Intel Core i7" };

        var advice = TranscriptionEngineAdvisor.Advise(caps);
        var npu = advice.All.Single(o => o.Device == ComputeDevice.Npu);

        Assert.Equal(EngineAvailability.Unsupported, npu.Availability);
        Assert.Empty(npu.Requirements);

        // Unsupported is not "needs setup", so it must never be offered as an upgrade path.
        Assert.True(advice.Better is null || advice.Better.Device != ComputeDevice.Npu);
    }

    /// <summary>
    /// There is always an answer. A machine with nothing installed still gets a recommendation
    /// it can act on today.
    /// </summary>
    [Fact]
    public void ThereIsAlwaysARunnableRecommendation()
    {
        var advice = TranscriptionEngineAdvisor.Advise(DeviceCapabilities.Unknown);

        Assert.Equal(EngineAvailability.Ready, advice.Recommended.Availability);
        Assert.Equal(ComputeDevice.Cpu, advice.Recommended.Device);
    }

    /// <summary>
    /// The advisor and the planner must not disagree: being told to use the NPU by one and put
    /// on the CPU by the other would be worse than either answer alone.
    /// </summary>
    [Theory]
    [InlineData(true, true, true)]
    [InlineData(true, true, false)]
    [InlineData(false, true, true)]
    [InlineData(true, false, true)]
    public void TheAdvisorAgreesWithThePlanner(bool qnn, bool driver, bool assets)
    {
        var caps = Snapdragon(qnn: qnn, driver: driver, assets: assets);

        var advice = TranscriptionEngineAdvisor.Advise(caps);
        var plan = AcceleratorPlanner.Plan(caps);

        Assert.Equal(plan.Encoder.Device, advice.Recommended.Device);
        Assert.Equal(plan.WhisperModel, advice.Recommended.ModelSize);
    }

    /// <summary>
    /// Live work steps the model down, because falling behind the speaker is unrecoverable. The
    /// advice has to reflect the mode it was asked about.
    /// </summary>
    [Fact]
    public void LiveWorkIsAdvisedASmallerModelThanBatch()
    {
        var caps = Snapdragon();

        var batch = TranscriptionEngineAdvisor.Advise(caps, WorkloadMode.Batch);
        var live = TranscriptionEngineAdvisor.Advise(caps, WorkloadMode.Live);

        Assert.NotEqual(batch.Recommended.ModelSize, live.Recommended.ModelSize);
    }
}
