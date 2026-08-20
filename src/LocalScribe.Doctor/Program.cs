using System.Runtime.InteropServices;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Provisioning;
using LocalScribe.Core.Refinement;
using LocalScribe.Onnx;

namespace LocalScribe.Doctor;

/// <summary>
/// Reports what this machine can do, and installs what is missing.
/// <para>
/// This exists because every hard problem in running Whisper on a Snapdragon presents the same
/// way: it works, but slowly, and nothing says why. The doctor turns that into a list of
/// specific findings, each with a specific fix — and where the fix is an ordinary download,
/// performs it.
/// </para>
/// </summary>
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        if (args.Contains("--help") || args.Contains("-h"))
        {
            PrintUsage();
            return 0;
        }

        var modelRoot = ArgumentValue(args, "--models")
            ?? Path.Combine(AppContext.BaseDirectory, "models");

        var live = args.Contains("--live");
        var maximum = args.Contains("--max");
        var install = args.Contains("--install");

        using var cancellation = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cancellation.Cancel();
            Console.WriteLine();
            Console.WriteLine("Cancelling…");
        };

        try
        {
            return await RunAsync(modelRoot, live, maximum, install, cancellation.Token)
                .ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("Cancelled. Partial downloads were discarded.");
            return 130;
        }
    }

    private static async Task<int> RunAsync(
        string modelRoot,
        bool live,
        bool maximum,
        bool install,
        CancellationToken cancellationToken)
    {
        Heading("Machine");

        var foundryManager = new FoundryLocalManager();
        var foundryInstalled = await foundryManager.IsInstalledAsync(cancellationToken).ConfigureAwait(false);

        var capabilities = await ProbeAsync(modelRoot, foundryManager, cancellationToken).ConfigureAwait(false);

        Report("Processor", capabilities.SocName);
        Report("Detected family", capabilities.Family.ToString());
        Report("Process architecture", RuntimeInformation.ProcessArchitecture.ToString());
        Report("Logical processors", capabilities.TotalCoreCount.ToString());
        Report("Memory", $"{capabilities.TotalMemoryGib} GiB");
        Report("Power", capabilities.OnBattery ? "battery" : "mains");

        var plan = AcceleratorPlanner.Plan(
            capabilities,
            maximum ? PerformanceProfile.Maximum : PerformanceProfile.Considerate,
            live ? WorkloadMode.Live : WorkloadMode.Batch);

        var chipsetSlug = DeviceProbe.AssetFolderFor(capabilities.Family);
        var modelDirectory = Path.Combine(modelRoot, chipsetSlug, plan.WhisperModel);

        var provisioner = new Provisioner();
        var provisioning = Provisioner.BuildPlan(
            capabilities,
            plan,
            modelDirectory,
            foundryInstalled,
            Provisioner.ProcessIsArm64);

        Heading("Prerequisites");
        foreach (var component in provisioning.Components)
        {
            PrintComponent(component);
        }

        if (install && provisioning.HasWorkToDo)
        {
            Heading("Installing");
            Console.WriteLine(
                "  Downloading from Hugging Face and Microsoft. Audio never leaves this machine, "
                + "but setup does reach the network.");
            Console.WriteLine();

            var progress = new Progress<InstallProgress>(PrintProgress);
            var results = await provisioner.InstallAsync(
                provisioning,
                modelDirectory,
                plan.WhisperModel,
                chipsetSlug,
                progress,
                cancellationToken).ConfigureAwait(false);

            Console.WriteLine();
            foreach (var result in results)
            {
                Console.WriteLine($"  [{(result.Succeeded ? "ok" : "!!")}] {result.ComponentId}: {result.Message}");
            }

            // Re-probe so the plan below reflects what was just installed rather than the
            // state we started from.
            capabilities = await ProbeAsync(modelRoot, foundryManager, cancellationToken).ConfigureAwait(false);
            plan = AcceleratorPlanner.Plan(
                capabilities,
                maximum ? PerformanceProfile.Maximum : PerformanceProfile.Considerate,
                live ? WorkloadMode.Live : WorkloadMode.Batch);
        }
        else if (provisioning.HasWorkToDo)
        {
            Console.WriteLine();
            Console.WriteLine($"  {provisioning.Installable.Count} item(s) can be installed automatically.");
            Console.WriteLine("  Re-run with --install to download them.");
        }

        if (provisioning.ManualActions.Count > 0)
        {
            Heading("Needs you");
            foreach (var component in provisioning.ManualActions)
            {
                Console.WriteLine($"  {component.Title}");
                Console.WriteLine($"    {component.ManualInstructions}");
                Console.WriteLine();
            }
        }

        Heading("Plan");
        Report("Whisper model", plan.WhisperModel);
        Report("Model directory", modelDirectory);
        Report("Encoder", $"{plan.Encoder.Device} — {plan.Encoder.Reason}");
        Report("Decoder", $"{plan.Decoder.Device} — {plan.Decoder.Reason}");
        Report("Cleanup", $"{plan.LanguageModel.Device} — {plan.LanguageModel.Reason}");
        Report("CPU budget", $"{plan.CpuBudget.IntraOpThreads} intra-op, "
            + $"{plan.CpuBudget.InterOpThreads} inter-op, "
            + (plan.CpuBudget.BelowNormalPriority ? "below-normal priority" : "normal priority"));

        Heading("Verdict");

        if (plan.IsCpuOnly)
        {
            Console.WriteLine(
                "  Everything will run on the CPU. The app works, but it will be slower and the "
                + "machine will feel busier. Address the items above to change that.");
            return 1;
        }

        Console.WriteLine($"  {plan.Summary}");
        Console.WriteLine(
            capabilities.NpuUsable
                ? "  The NPU path is fully configured."
                : "  Running on the GPU. The NPU would be quieter still; see the items above.");

        return 0;
    }

    /// <summary>
    /// Probes the hardware and asks Foundry Local where it is listening, since the port is
    /// dynamic and a hard-coded guess reports a running service as absent.
    /// </summary>
    private static async Task<DeviceCapabilities> ProbeAsync(
        string modelRoot,
        FoundryLocalManager foundryManager,
        CancellationToken cancellationToken)
    {
        var capabilities = DeviceProbe.Probe(modelRoot);

        var endpoint = await foundryManager.DiscoverEndpointAsync(cancellationToken).ConfigureAwait(false);
        using var client = new FoundryLocalClient(endpoint: endpoint);
        var available = await client.IsAvailableAsync(cancellationToken).ConfigureAwait(false);

        return capabilities with { FoundryLocalPresent = available };
    }

    private static void PrintComponent(ComponentStatus component)
    {
        var mark = component.Installed ? "ok" : component.Required ? "!!" : "--";
        var optional = component.Required ? string.Empty : " (optional)";

        Console.WriteLine($"  [{mark}] {component.Title}{optional}");
        Console.WriteLine($"       {component.Detail}");
    }

    private static void PrintProgress(InstallProgress progress)
    {
        var percent = progress.Fraction is { } fraction
            ? $" {fraction * 100:F0}%"
            : string.Empty;

        Console.WriteLine($"  {progress.Message}{percent}");
    }

    private static void PrintUsage()
    {
        Console.WriteLine("""
            localscribe-doctor — report what this machine can do, and install what is missing.

              --install         Download missing models and install the local inference engine.
                                Without this, nothing is changed and nothing is downloaded.
              --models <path>   Where model assets live. Defaults to ./models beside the binary.
              --live            Plan for live transcription rather than batch files.
              --max             Plan for maximum speed rather than a responsive machine.
              --help            This text.

            Exit codes: 0 when an accelerator will be used, 1 when everything falls back to CPU.
            """);
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
        Console.WriteLine($"  {label,-24} {value}");
}
