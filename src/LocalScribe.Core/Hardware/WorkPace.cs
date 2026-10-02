namespace LocalScribe.Core.Hardware;

/// <summary>
/// How much of the machine the user is willing to give a transcription, chosen on a slider.
/// <para>
/// The planner's own budget is a guess about what the user is doing with the rest of the
/// computer, and it guesses cautiously: with the speech model on the NPU it leaves the CPU two
/// threads, because a transcription nobody notices beats a faster one that makes the machine
/// stutter. That is the right default and the wrong ceiling. Someone who has walked away from
/// the laptop wants the hour of audio done, and only they know they have walked away.
/// </para>
/// <para>
/// <see cref="Balanced"/> is exactly the planner's answer, so the default changes nothing.
/// </para>
/// </summary>
public enum WorkPace
{
    /// <summary>Half the default CPU threads, and the NPU in its balanced power mode.</summary>
    Light = 0,

    /// <summary>The planner's own budget. The default.</summary>
    Balanced = 1,

    /// <summary>
    /// Halfway from the default to <see cref="Full"/>, at normal priority. Halfway rather than a
    /// fixed share, because the default differs by machine: with the speech model on the NPU it
    /// is two threads, on a CPU-only machine two-thirds of the cores, and a fixed half would be
    /// no faster than the default on the second. On a small machine the two can coincide.
    /// </summary>
    Fast = 2,

    /// <summary>
    /// Every core but two, at normal priority.
    /// <para>
    /// Not every core, because that measured slower. On the 12-core Snapdragon the word aligner
    /// took 18.1 s on ten threads and 25.4 s on twelve, and Nemotron 3.6 s against 4.1 s; during
    /// a burst of Windows Recall's background indexing, twelve threads took Nemotron from 36 s to
    /// 130 s. Each parallel step waits for its slowest thread, so a thread that shares its core
    /// with anything else stalls all of them. Two cores left over absorb the rest of the machine.
    /// </para>
    /// </summary>
    Full = 3,
}

/// <summary>
/// How hard the NPU is driven. A power setting rather than a share: the Hexagon runs one graph
/// at a time, so what can be traded is clock speed against heat and battery, not cores.
/// </summary>
public enum NpuPower
{
    /// <summary>Highest clocks while work is running. What the app has always used.</summary>
    Burst = 0,

    /// <summary>Moderate clocks: slower, cooler, kinder to a battery.</summary>
    Balanced,
}

/// <summary>Applies a <see cref="WorkPace"/> to a plan, and remembers the user's choice.</summary>
public static class WorkPaces
{
    /// <summary>Every pace, in slider order.</summary>
    public static IReadOnlyList<WorkPace> All { get; } =
        [WorkPace.Light, WorkPace.Balanced, WorkPace.Fast, WorkPace.Full];

    /// <summary>
    /// The plan with its CPU budget and NPU power set for a pace. Placement, model size and
    /// everything else the planner decided are left alone: a pace changes how hard the work
    /// runs, never where.
    /// </summary>
    /// <param name="plan">The planner's answer, which is what <see cref="WorkPace.Balanced"/> keeps.</param>
    /// <param name="cores">Cores the machine has for this work.</param>
    public static ExecutionPlan Apply(ExecutionPlan plan, WorkPace pace, int cores)
    {
        ArgumentNullException.ThrowIfNull(plan);

        cores = Math.Max(1, cores);
        var planned = plan.CpuBudget;
        var most = MostThreads(cores);
        var middle = planned.IntraOpThreads + ((Math.Max(0, most - planned.IntraOpThreads) + 1) / 2);

        var (threads, belowNormal, npu) = pace switch
        {
            WorkPace.Light => (Math.Max(1, planned.IntraOpThreads / 2), true, NpuPower.Balanced),
            WorkPace.Fast => (Math.Min(most, middle), false, NpuPower.Burst),
            WorkPace.Full => (Math.Max(planned.IntraOpThreads, most), false, NpuPower.Burst),
            _ => (planned.IntraOpThreads, planned.BelowNormalPriority, plan.NpuPower),
        };

        return plan with
        {
            CpuBudget = planned with { IntraOpThreads = threads, BelowNormalPriority = belowNormal },
            NpuPower = npu,
        };
    }

    /// <summary>
    /// The most threads worth giving: every core but two, so the rest of the machine never
    /// shares a core with a thread the others are waiting on. A machine of four cores or fewer
    /// has none to spare and gets all of them. See <see cref="WorkPace.Full"/>.
    /// </summary>
    public static int MostThreads(int cores) => cores > 4 ? cores - 2 : Math.Max(1, cores);

    /// <summary>One line for under the slider: what this pace will actually do here.</summary>
    public static string Describe(WorkPace pace, ExecutionPlan paced, int cores)
    {
        ArgumentNullException.ThrowIfNull(paced);

        var threads = paced.CpuBudget.IntraOpThreads;
        var cpu = threads >= cores
            ? $"all {cores} CPU cores"
            : $"{threads} of {cores} CPU cores";

        var npu = paced.Encoder.Device == ComputeDevice.Npu
            ? paced.NpuPower == NpuPower.Burst ? ", with the NPU at full speed" : ", with the NPU in power-saving mode"
            : string.Empty;

        return pace switch
        {
            WorkPace.Light => $"Lighter on this PC: {cpu}{npu}. Slower, but the rest of the computer barely notices.",
            WorkPace.Fast => $"Faster: {cpu}{npu}. Other apps may feel slower while it works.",
            WorkPace.Full => $"Fastest: {cpu}{npu}. Best when you are not using the computer for anything else.",
            _ => $"Balanced, the default: {cpu}{npu}.",
        };
    }

    /// <summary>Where the choice is kept: beside the user's other app data, not the models.</summary>
    public static string DefaultPath =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "LocalScribe",
            "work-pace.txt");

    /// <summary>The pace recorded at a path, or <see cref="WorkPace.Balanced"/> when none is.</summary>
    public static WorkPace Read(string path)
    {
        try
        {
            return File.Exists(path) ? Parse(File.ReadAllText(path)) : WorkPace.Balanced;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            // A preference that cannot be read is not worth a failed launch. The default is
            // the planner's own answer, so falling back to it is always safe.
            return WorkPace.Balanced;
        }
    }

    /// <summary>Records a pace. Failure is silent: the slider still applies for this session.</summary>
    public static void Write(string path, WorkPace pace)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, pace.ToString().ToLowerInvariant());
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
        }
    }

    /// <summary>The pace a name refers to, falling back to <see cref="WorkPace.Balanced"/>.</summary>
    public static WorkPace Parse(string? name) =>
        Enum.TryParse<WorkPace>(name?.Trim(), ignoreCase: true, out var pace) && Enum.IsDefined(pace)
            ? pace
            : WorkPace.Balanced;
}
