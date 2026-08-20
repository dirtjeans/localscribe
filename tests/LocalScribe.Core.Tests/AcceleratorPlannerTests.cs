using LocalScribe.Core.Hardware;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// The planner is the only place that decides where work runs, so it carries most of the
/// product's behaviour. These tests pin down the policy on machines nobody here owns.
/// </summary>
public sealed class AcceleratorPlannerTests
{
    private static DeviceCapabilities SnapdragonXElite(
        bool qnn = true,
        bool driver = true,
        bool assets = true,
        bool directMl = true,
        bool foundry = true,
        bool onBattery = false,
        int memoryGib = 32) => new()
        {
            SocName = "Snapdragon X Elite X1E-78-100",
            Family = SocFamily.SnapdragonXElite,
            PerformanceCoreCount = 12,
            TotalCoreCount = 12,
            TotalMemoryBytes = memoryGib * 1024L * 1024 * 1024,
            QnnProviderPresent = qnn,
            HexagonDriverPresent = driver,
            WhisperQnnAssetsPresent = assets,
            DirectMlPresent = directMl,
            FoundryLocalPresent = foundry,
            OnBattery = onBattery,
        };

    [Fact]
    public void FullyEquippedSnapdragon_PutsBothWhisperStagesOnTheNpu()
    {
        var plan = AcceleratorPlanner.Plan(SnapdragonXElite());

        Assert.Equal(ComputeDevice.Npu, plan.Encoder.Device);
        Assert.Equal(ComputeDevice.Npu, plan.Decoder.Device);
        Assert.Equal(AcceleratorPlanner.QnnProvider, plan.Encoder.ExecutionProvider);
        Assert.False(plan.IsCpuOnly);
        Assert.Empty(plan.Warnings);
    }

    [Fact]
    public void WhenModelsRunOffCpu_TheCpuBudgetStaysSmall()
    {
        // The whole point of offloading is to leave the machine usable. Claiming eight threads
        // for mel-spectrogram work would give that back.
        var plan = AcceleratorPlanner.Plan(SnapdragonXElite());

        Assert.Equal(2, plan.CpuBudget.IntraOpThreads);
        Assert.True(plan.CpuBudget.BelowNormalPriority);
    }

    [Fact]
    public void MissingHexagonDriver_FallsBackToGpuAndSaysWhy()
    {
        var plan = AcceleratorPlanner.Plan(SnapdragonXElite(driver: false));

        Assert.Equal(ComputeDevice.Gpu, plan.Encoder.Device);
        Assert.Contains(plan.Warnings, w => w.Contains("Hexagon NPU runtime driver", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void MissingModelAssets_MentionsTheChipsetInTheWarning()
    {
        // Precompiled QNN binaries are per-chipset, so naming the SoC is the difference between
        // a useful message and a scavenger hunt.
        var plan = AcceleratorPlanner.Plan(SnapdragonXElite(assets: false));

        Assert.Contains(plan.Warnings, w => w.Contains("Snapdragon X Elite", StringComparison.Ordinal));
    }

    [Fact]
    public void MissingQnnProvider_WarnsAboutEmulation()
    {
        // Running the arm64 app under x64 emulation is a common self-inflicted cause, and it
        // presents exactly like a broken install.
        var plan = AcceleratorPlanner.Plan(SnapdragonXElite(qnn: false));

        Assert.Contains(plan.Warnings, w => w.Contains("emulation", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void NoAcceleratorAtAll_RunsEverythingOnCpuAndSaysSo()
    {
        var plan = AcceleratorPlanner.Plan(SnapdragonXElite(qnn: false, directMl: false, foundry: false));

        Assert.True(plan.IsCpuOnly);
        Assert.Contains(plan.Warnings, w => w.Contains("No accelerator available", StringComparison.Ordinal));
    }

    [Fact]
    public void OnCpu_ConsiderateProfileLeavesCoresFree()
    {
        var plan = AcceleratorPlanner.Plan(
            SnapdragonXElite(qnn: false, directMl: false),
            PerformanceProfile.Considerate);

        Assert.Equal(ComputeDevice.Cpu, plan.Encoder.Device);
        Assert.True(plan.CpuBudget.IntraOpThreads < 12, "Considerate mode must not claim every core.");
        Assert.True(plan.CpuBudget.IntraOpThreads >= 2, "Two threads is the floor, even when being polite.");
    }

    [Fact]
    public void OnCpu_MaximumProfileUsesEveryPerformanceCore()
    {
        var plan = AcceleratorPlanner.Plan(
            SnapdragonXElite(qnn: false, directMl: false),
            PerformanceProfile.Maximum);

        Assert.Equal(12, plan.CpuBudget.IntraOpThreads);
        Assert.False(plan.CpuBudget.BelowNormalPriority);
    }

    [Fact]
    public void OnBattery_TakesLessCpuThanOnMains()
    {
        var mains = AcceleratorPlanner.Plan(SnapdragonXElite(qnn: false, directMl: false, onBattery: false));
        var battery = AcceleratorPlanner.Plan(SnapdragonXElite(qnn: false, directMl: false, onBattery: true));

        Assert.True(battery.CpuBudget.IntraOpThreads < mains.CpuBudget.IntraOpThreads);
    }

    [Fact]
    public void InterOpThreadsAlwaysOne()
    {
        // Whisper's graph is sequential. Extra inter-op threads buy contention, not throughput.
        foreach (var profile in new[] { PerformanceProfile.Considerate, PerformanceProfile.Maximum })
        {
            Assert.Equal(1, AcceleratorPlanner.Plan(SnapdragonXElite(), profile).CpuBudget.InterOpThreads);
        }
    }

    [Fact]
    public void DecoderNeverGoesToTheGpu()
    {
        // Dispatch overhead per decode step outweighs the compute saved, so a GPU encoder does
        // not imply a GPU decoder.
        var plan = AcceleratorPlanner.Plan(SnapdragonXElite(driver: false));

        Assert.Equal(ComputeDevice.Gpu, plan.Encoder.Device);
        Assert.Equal(ComputeDevice.Cpu, plan.Decoder.Device);
    }

    [Theory]
    [InlineData(WorkloadMode.Batch, 32, "medium.en")]
    [InlineData(WorkloadMode.Batch, 8, "small.en")]
    [InlineData(WorkloadMode.Live, 32, "small.en")]
    [InlineData(WorkloadMode.Live, 8, "base.en")]
    public void NpuPath_PicksLargerModelsThanTheCpuPathCould(WorkloadMode mode, int memoryGib, string expected)
    {
        var plan = AcceleratorPlanner.Plan(
            SnapdragonXElite(memoryGib: memoryGib),
            PerformanceProfile.Considerate,
            mode);

        Assert.Equal(expected, plan.WhisperModel);
    }

    [Theory]
    [InlineData(WorkloadMode.Batch, 32, "small.en")]
    [InlineData(WorkloadMode.Batch, 8, "base.en")]
    [InlineData(WorkloadMode.Live, 32, "base.en")]
    [InlineData(WorkloadMode.Live, 8, "tiny.en")]
    public void CpuPath_StepsDownToKeepUp(WorkloadMode mode, int memoryGib, string expected)
    {
        var plan = AcceleratorPlanner.Plan(
            SnapdragonXElite(qnn: false, directMl: false, memoryGib: memoryGib),
            PerformanceProfile.Considerate,
            mode);

        Assert.Equal(expected, plan.WhisperModel);
    }

    [Fact]
    public void NonQualcommMachine_SkipsTheDriverAdvice()
    {
        // Telling an Intel laptop to install a Hexagon driver would be actively misleading.
        var caps = DeviceCapabilities.Unknown with { Family = SocFamily.NonQualcomm };
        var plan = AcceleratorPlanner.Plan(caps);

        Assert.Contains(plan.Warnings, w => w.Contains("not a Qualcomm processor", StringComparison.Ordinal));
        Assert.DoesNotContain(plan.Warnings, w => w.Contains("Hexagon", StringComparison.Ordinal));
    }

    [Fact]
    public void WithoutFoundryLocal_CleanupIsDisabledButTranscriptionStillPlanned()
    {
        var plan = AcceleratorPlanner.Plan(SnapdragonXElite(foundry: false));

        Assert.Equal(ComputeDevice.Npu, plan.Encoder.Device);
        Assert.Contains("Disabled", plan.LanguageModel.Reason, StringComparison.Ordinal);
        Assert.Contains(plan.Warnings, w => w.Contains("Foundry Local", StringComparison.Ordinal));
    }

    [Fact]
    public void EveryStagePlanExplainsItself()
    {
        // These strings are shown to the user in the doctor tool, so an empty one is a bug.
        var plan = AcceleratorPlanner.Plan(SnapdragonXElite());

        Assert.NotEmpty(plan.Encoder.Reason);
        Assert.NotEmpty(plan.Decoder.Reason);
        Assert.NotEmpty(plan.LanguageModel.Reason);
        Assert.NotEmpty(plan.Summary);
    }

    [Fact]
    public void SingleCoreMachine_StillProducesAUsableBudget()
    {
        var caps = DeviceCapabilities.Unknown with { PerformanceCoreCount = 1, TotalCoreCount = 1 };
        var plan = AcceleratorPlanner.Plan(caps);

        Assert.True(plan.CpuBudget.IntraOpThreads >= 1);
    }
}
