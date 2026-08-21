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
        var fetch = args.Contains("--fetch-models");

        if (args.Contains("--help") || args.Contains("-h"))
        {
            PrintUsage();
            return 0;
        }

        Heading("Machine");

        var capabilities = DeviceProbe.Probe(modelDirectory);
        var languageModel = await LocalLanguageModel.ResolveAsync().ConfigureAwait(false);
        using var languageModelHandle = languageModel as IDisposable;
        capabilities = capabilities with { LocalLanguageModelPresent = languageModel is not null };

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
        Check(
            languageModel is null
                ? "Local language model"
                : $"Local language model ({languageModel.Description})",
            capabilities.LocalLanguageModelPresent,
            "Optional; enables punctuation repair, glossary correction and summaries. Start "
            + $"one of: {string.Join(", ", LocalLanguageModel.BackendNames)}. GenieX is "
            + "preferred on Snapdragon; see docs/setup-snapdragon.md.");

        if (capabilities.OnnxProviders.Count > 0)
        {
            Report("Providers registered", string.Join(", ", capabilities.OnnxProviders.Order()));
        }

        Heading("Plan");

        var plan = AcceleratorPlanner.Plan(
            capabilities,
            maximum ? PerformanceProfile.Maximum : PerformanceProfile.Considerate,
            live ? WorkloadMode.Live : WorkloadMode.Batch,
            strictProviderCheck: args.Contains("--strict"));

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

        Heading("Recommended engine");

        var advice = TranscriptionEngineAdvisor.Advise(
            capabilities,
            live ? WorkloadMode.Live : WorkloadMode.Batch);

        Console.WriteLine($"  {advice.Summary}");
        Console.WriteLine();

        foreach (var option in advice.All)
        {
            var mark = option.Availability switch
            {
                EngineAvailability.Ready => "ok",
                EngineAvailability.NeedsSetup => "--",
                _ => "no",
            };

            var here = ReferenceEquals(option, advice.Recommended) ? "  <- using this" : string.Empty;
            Console.WriteLine($"  [{mark}] {option.Name} ({option.ModelSize}){here}");
            Console.WriteLine($"       {option.Rationale}");

            foreach (var requirement in option.Requirements)
            {
                Console.WriteLine($"       needs: {requirement}");
            }
        }

        var matrixAudio = ArgumentValue(args, "--speaker-matrix");
        if (matrixAudio is not null)
        {
            return DiarizeCommand.Matrix(
                matrixAudio,
                ArgumentValue(args, "--diarization-models") ?? Path.Combine(modelDirectory, "diarization"),
                ArgumentValue(args, "--spans") ?? string.Empty);
        }

        var diarizeAudio = ArgumentValue(args, "--diarize");
        if (diarizeAudio is not null)
        {
            if (args.Contains("--sweep", StringComparer.Ordinal))
            {
                return DiarizeCommand.Sweep(
                    diarizeAudio,
                    ArgumentValue(args, "--diarization-models")
                        ?? Path.Combine(modelDirectory, "diarization"));
            }

            return DiarizeCommand.Run(
                diarizeAudio,
                ArgumentValue(args, "--diarization-models")
                    ?? Path.Combine(modelDirectory, "diarization"),
                ArgumentValue(args, "--speakers"),
                ArgumentValue(args, "--threshold"));
        }

        var liveAudio = ArgumentValue(args, "--transcribe-live");
        if (liveAudio is not null)
        {
            return await TranscribeCommand.RunLiveAsync(
                    liveAudio, modelDirectory, capabilities, plan, ArgumentValue(args, "--model-dir"))
                .ConfigureAwait(false);
        }

        var audio = ArgumentValue(args, "--transcribe");
        if (audio is not null)
        {
            return await TranscribeCommand.RunAsync(
                    audio,
                    modelDirectory,
                    capabilities,
                    plan,
                    ArgumentValue(args, "--model-dir"))
                .ConfigureAwait(false);
        }

        if (fetch)
        {
            return await FetchModels.RunAsync(
                    modelDirectory,
                    capabilities,
                    plan,
                    ArgumentValue(args, "--model"),
                    args.Contains("--force"))
                .ConfigureAwait(false);
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

    private static void PrintUsage()
    {
        Console.WriteLine("localscribe-doctor — reports what this machine can do, and why.");
        Console.WriteLine();
        Console.WriteLine("  --models <dir>   Where model weights live. Defaults to ./models.");
        Console.WriteLine("  --fetch-models   Download the portable Whisper export this machine will use.");
        Console.WriteLine("  --model <size>   Size to fetch. Defaults to the one the plan chose.");
        Console.WriteLine("  --force          Re-download files that are already present.");
        Console.WriteLine("  --transcribe <f> Transcribe a PCM WAV file and report what happened.");
        Console.WriteLine("  --transcribe-live <f>  Feed a WAV through the live path, as the microphone does.");
        Console.WriteLine("  --diarize <f>    Work out who spoke when in a WAV, and print the turns.");
        Console.WriteLine("  --speakers <n>   Upper bound on speakers for --diarize.");
        Console.WriteLine("  --threshold <d>  Cosine distance at which two voices are different people.");
        Console.WriteLine("  --sweep          With --diarize: try every threshold and show where the answer changes.");
        Console.WriteLine("  --model-dir <d>  Model directory for --transcribe, overriding the layout.");
        Console.WriteLine("  --strict         Refuse to let a requested provider quietly fall back to the CPU.");
        Console.WriteLine("  --live           Plan for live transcription rather than a batch pass.");
        Console.WriteLine("  --max            Plan for maximum speed rather than a considerate CPU share.");
        Console.WriteLine("  --help, -h       This text.");
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
