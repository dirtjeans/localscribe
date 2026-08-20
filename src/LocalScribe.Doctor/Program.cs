using System.Runtime.InteropServices;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Refinement;
using LocalScribe.Onnx;

namespace LocalScribe.Doctor;

/// <summary>
/// Reports what this machine can do and what the app will do with it.
/// <para>
/// This exists because every hard problem in running Whisper on a Snapdragon presents the same
/// way: it works, but slowly, and nothing says why. The doctor turns that into a list of
/// specific findings, each with the specific fix.
/// </para>
/// </summary>
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        var modelDirectory = ArgumentValue(args, "--models")
            ?? Path.Combine(AppContext.BaseDirectory, "models");

        var live = args.Contains("--live");
        var maximum = args.Contains("--max");

        Heading("Machine");

        var capabilities = DeviceProbe.Probe(modelDirectory);
        var foundry = new FoundryLocalClient();
        var foundryAvailable = await foundry.IsAvailableAsync().ConfigureAwait(false);
        capabilities = capabilities with { FoundryLocalPresent = foundryAvailable };

        Report("Processor", capabilities.SocName);
        Report("Detected family", capabilities.Family.ToString());
        Report("Process architecture", RuntimeInformation.ProcessArchitecture.ToString());
        Report("Logical processors", capabilities.TotalCoreCount.ToString());
        Report("Memory", $"{capabilities.TotalMemoryGib} GiB");
        Report("Power", capabilities.OnBattery ? "battery" : "mains");

        if (RuntimeInformation.ProcessArchitecture != Architecture.Arm64 && OperatingSystem.IsWindows())
        {
            Warn(
                "This process is not running as arm64. Under x64 emulation the QNN provider "
                + "cannot load at all. Rebuild with -r win-arm64.");
        }

        Heading("Acceleration");

        Check("QNN execution provider", capabilities.QnnProviderPresent,
            "Install the Microsoft.ML.OnnxRuntime.QNN package and build for win-arm64.");
        Check("Hexagon NPU runtime driver", capabilities.HexagonDriverPresent,
            "Install the Hexagon NPU Runtime Driver from Qualcomm Software Center. This is a "
            + "separate download from the driver Windows ships with, and is the usual reason "
            + "the NPU sits idle.");
        Check($"Whisper assets for {DeviceProbe.AssetFolderFor(capabilities.Family)}",
            capabilities.WhisperQnnAssetsPresent,
            $"Place encoder.onnx, decoder.onnx and vocab.json under "
            + $"{Path.Combine(modelDirectory, DeviceProbe.AssetFolderFor(capabilities.Family))}. "
            + "Precompiled QNN binaries are chipset-specific; see docs/setup-snapdragon.md.");
        Check("DirectML (Adreno GPU)", capabilities.DirectMlPresent,
            "Optional. Used as the fallback when the NPU is unavailable.");
        Check("Foundry Local", capabilities.FoundryLocalPresent,
            "Optional. Run 'foundry service start' to enable punctuation repair and summaries.");

        if (capabilities.OnnxProviders.Count > 0)
        {
            Report("Providers registered", string.Join(", ", capabilities.OnnxProviders.Order()));
        }

        Heading("Plan");

        var plan = AcceleratorPlanner.Plan(
            capabilities,
            maximum ? PerformanceProfile.Maximum : PerformanceProfile.Considerate,
            live ? WorkloadMode.Live : WorkloadMode.Batch);

        Report("Whisper model", plan.WhisperModel);
        Report("Encoder", $"{plan.Encoder.Device} — {plan.Encoder.Reason}");
        Report("Decoder", $"{plan.Decoder.Device} — {plan.Decoder.Reason}");
        Report("Cleanup", $"{plan.LanguageModel.Device} — {plan.LanguageModel.Reason}");
        Report("CPU budget", $"{plan.CpuBudget.IntraOpThreads} intra-op, "
            + $"{plan.CpuBudget.InterOpThreads} inter-op, "
            + (plan.CpuBudget.BelowNormalPriority ? "below-normal priority" : "normal priority"));

        if (plan.Warnings.Count > 0)
        {
            Heading("Findings");
            foreach (var warning in plan.Warnings)
            {
                Warn(warning);
            }
        }

        Heading("Verdict");

        if (plan.IsCpuOnly)
        {
            Console.WriteLine(
                "Everything will run on the CPU. The app works, but it will be slower and the "
                + "machine will feel busier. Fix the findings above to change that.");
            return 1;
        }

        Console.WriteLine(plan.Summary);
        Console.WriteLine(
            capabilities.NpuUsable
                ? "The NPU path is fully configured."
                : "Running on the GPU. The NPU would be quieter still; see the findings above.");

        return 0;
    }

    private static string? ArgumentValue(string[] args, string name)
    {
        var index = Array.IndexOf(args, name);
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }

    private static void Heading(string title)
    {
        Console.WriteLine();
        Console.WriteLine(title);
        Console.WriteLine(new string('-', title.Length));
    }

    private static void Report(string label, string value) =>
        Console.WriteLine($"  {label,-32} {value}");

    private static void Check(string label, bool ok, string remedy)
    {
        Console.WriteLine($"  [{(ok ? "ok" : "--")}] {label}");
        if (!ok)
        {
            Console.WriteLine($"       {remedy}");
        }
    }

    private static void Warn(string message) => Console.WriteLine($"  ! {message}");
}
