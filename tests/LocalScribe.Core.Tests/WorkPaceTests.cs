using LocalScribe.Core.Hardware;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// The pace slider. What matters is that the default changes nothing, that the stops only ever
/// change how hard the work runs and never where, and that each stop really is a step.
/// </summary>
public class WorkPaceTests
{
    private const int Cores = 12;

    private static DeviceCapabilities Machine(bool npu) => new()
    {
        SocName = "Snapdragon X Elite X1E-78-100",
        Family = SocFamily.SnapdragonXElite,
        PerformanceCoreCount = Cores,
        TotalCoreCount = Cores,
        TotalMemoryBytes = 32L * 1024 * 1024 * 1024,
        QnnProviderPresent = npu,
        HexagonDriverPresent = npu,
        WhisperQnnAssetsPresent = npu,
        DirectMlPresent = false,
        LocalLanguageModelPresent = true,
    };

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void BalancedIsExactlyThePlannersAnswer(bool npu)
    {
        var planned = AcceleratorPlanner.Plan(Machine(npu));

        Assert.Equal(planned, WorkPaces.Apply(planned, WorkPace.Balanced, Cores));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void APaceNeverMovesWorkToADifferentProcessor(bool npu)
    {
        var planned = AcceleratorPlanner.Plan(Machine(npu));

        foreach (var pace in WorkPaces.All)
        {
            var paced = WorkPaces.Apply(planned, pace, Cores);

            Assert.Equal(planned.Encoder, paced.Encoder);
            Assert.Equal(planned.Decoder, paced.Decoder);
            Assert.Equal(planned.LanguageModel, paced.LanguageModel);
            Assert.Equal(planned.WhisperModel, paced.WhisperModel);
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void EveryStopIsAStep(bool npu)
    {
        var planned = AcceleratorPlanner.Plan(Machine(npu));

        var threads = WorkPaces.All
            .Select(pace => WorkPaces.Apply(planned, pace, Cores).CpuBudget.IntraOpThreads)
            .ToList();

        // Strictly more threads at every stop to the right, on the machine with the NPU and on
        // the one without — whose default is already two-thirds of the cores.
        for (var i = 1; i < threads.Count; i++)
        {
            Assert.True(threads[i] > threads[i - 1], string.Join(", ", threads));
        }

        Assert.Equal(WorkPaces.MostThreads(Cores), threads[^1]);
    }

    [Fact]
    public void TheTopStopLeavesTwoCoresForEverythingElse()
    {
        // Measured: on twelve cores, ten threads beat twelve, because every parallel step waits
        // for whichever thread is sharing its core with another program.
        Assert.Equal(10, WorkPaces.MostThreads(12));
        Assert.Equal(6, WorkPaces.MostThreads(8));
        Assert.Equal(4, WorkPaces.MostThreads(4));
        Assert.Equal(1, WorkPaces.MostThreads(1));
    }

    [Fact]
    public void LightNeverTakesTheLastThread()
    {
        var planned = AcceleratorPlanner.Plan(Machine(npu: true)) with
        {
            CpuBudget = new CpuBudget(1, 1, BelowNormalPriority: true),
        };

        Assert.Equal(1, WorkPaces.Apply(planned, WorkPace.Light, Cores).CpuBudget.IntraOpThreads);
    }

    [Fact]
    public void OnlyLightEasesOffTheNpu()
    {
        var planned = AcceleratorPlanner.Plan(Machine(npu: true));

        Assert.Equal(NpuPower.Balanced, WorkPaces.Apply(planned, WorkPace.Light, Cores).NpuPower);
        Assert.Equal(NpuPower.Burst, WorkPaces.Apply(planned, WorkPace.Balanced, Cores).NpuPower);
        Assert.Equal(NpuPower.Burst, WorkPaces.Apply(planned, WorkPace.Full, Cores).NpuPower);
    }

    [Fact]
    public void TheFasterStopsGiveUpTheLowPriority()
    {
        var planned = AcceleratorPlanner.Plan(Machine(npu: true));

        Assert.True(WorkPaces.Apply(planned, WorkPace.Light, Cores).CpuBudget.BelowNormalPriority);
        Assert.False(WorkPaces.Apply(planned, WorkPace.Fast, Cores).CpuBudget.BelowNormalPriority);
        Assert.False(WorkPaces.Apply(planned, WorkPace.Full, Cores).CpuBudget.BelowNormalPriority);
    }

    [Fact]
    public void TheDescriptionMentionsTheNpuOnlyWhenItIsUsed()
    {
        var withNpu = AcceleratorPlanner.Plan(Machine(npu: true));
        var without = AcceleratorPlanner.Plan(Machine(npu: false));

        Assert.Contains("NPU", WorkPaces.Describe(WorkPace.Balanced, withNpu, Cores));
        Assert.DoesNotContain("NPU", WorkPaces.Describe(WorkPace.Balanced, without, Cores));
        Assert.Contains("10 of 12 CPU cores", WorkPaces.Describe(WorkPace.Full, WorkPaces.Apply(without, WorkPace.Full, Cores), Cores));
    }

    [Theory]
    [InlineData("light", WorkPace.Light)]
    [InlineData(" FULL\n", WorkPace.Full)]
    [InlineData("fast", WorkPace.Fast)]
    [InlineData("ludicrous", WorkPace.Balanced)]
    [InlineData("7", WorkPace.Balanced)]
    [InlineData("", WorkPace.Balanced)]
    [InlineData(null, WorkPace.Balanced)]
    public void AnythingUnreadableIsTheDefault(string? text, WorkPace expected)
    {
        Assert.Equal(expected, WorkPaces.Parse(text));
    }

    [Fact]
    public void AChoiceSurvivesARestart()
    {
        var path = Path.Combine(Path.GetTempPath(), $"localscribe-pace-{Guid.NewGuid():N}", "work-pace.txt");

        try
        {
            Assert.Equal(WorkPace.Balanced, WorkPaces.Read(path));

            WorkPaces.Write(path, WorkPace.Fast);

            Assert.Equal(WorkPace.Fast, WorkPaces.Read(path));
        }
        finally
        {
            Directory.Delete(Path.GetDirectoryName(path)!, recursive: true);
        }
    }
}
