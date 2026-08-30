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

        if (RuntimeInformation.ProcessArchitecture != Architecture.Arm64 && OperatingSystem.IsMacOS())
        {
            // The same trap as x64-under-emulation on Windows, wearing a different coat: under
            // Rosetta the Apple Neural Engine is unreachable and the symptoms look like a
            // missing framework rather than a wrong build.
            Warn(
                "This process is not running as arm64. Under Rosetta the Apple Neural Engine "
                + "cannot be reached at all. Rebuild with -r osx-arm64.");
        }

        Heading("Acceleration");

        if (OperatingSystem.IsMacOS())
        {
            // The Qualcomm checklist would be three permanent failures here, each with a remedy
            // that cannot be followed. What is worth knowing on a Mac is whether the Core ML
            // provider loaded — the only accelerator ONNX Runtime can reach on this platform.
            Check("Core ML execution provider",
                capabilities.OnnxProviders.Contains("CoreMLExecutionProvider"),
                "This build of ONNX Runtime is missing the Core ML provider; the stock "
                + "osx-arm64 package carries it, so check the package and the RuntimeIdentifier.");
            Check($"Whisper assets for {DeviceProbe.AssetFolderFor(capabilities.Family)}",
                capabilities.WhisperQnnAssetsPresent,
                $"Place encoder.onnx, decoder.onnx and vocab.json under "
                + $"{Path.Combine(modelDirectory, DeviceProbe.AssetFolderFor(capabilities.Family))}. "
                + "Run 'localscribe-doctor --fetch-models' to download a portable set.");
        }
        else
        {
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
        }
        Check(
            languageModel is null
                ? "Local language model"
                : $"Local language model ({languageModel.Description})",
            capabilities.LocalLanguageModelPresent,
            OperatingSystem.IsMacOS()
                ? "Optional; enables punctuation repair, glossary correction and summaries. "
                  + "Foundry Local ships for Apple silicon: 'brew tap microsoft/foundrylocal && "
                  + "brew install foundrylocal', then 'foundry service start'."
                : "Optional; enables punctuation repair, glossary correction and summaries. Start "
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
                ArgumentValue(args, "--threshold"),
                args.Contains("--tracking", StringComparer.Ordinal));
        }

        var liveAudio = ArgumentValue(args, "--transcribe-live");
        if (liveAudio is not null)
        {
            return await TranscribeCommand.RunLiveAsync(
                    liveAudio, modelDirectory, capabilities, plan, ArgumentValue(args, "--model-dir"))
                .ConfigureAwait(false);
        }

        if (args.Contains("--diarizer"))
        {
            return DiarizerCommand.Run(modelDirectory, ArgumentValue(args, "--diarizer"));
        }

        if (args.Contains("--segmentation-model"))
        {
            return await SegmentationModelCommand
                .RunAsync(modelDirectory, ArgumentValue(args, "--segmentation-model"))
                .ConfigureAwait(false);
        }

        if (args.Contains("--speaker-model"))
        {
            return await SpeakerModelCommand
                .RunAsync(modelDirectory, ArgumentValue(args, "--speaker-model"))
                .ConfigureAwait(false);
        }

        var compare = ArgumentValue(args, "--speaker-models");
        if (compare is not null)
        {
            return SpeakerModelsCommand.Run(
                compare,
                ArgumentValue(args, "--candidates") ?? Path.Combine(modelDirectory, "candidates"));
        }

        var archive = ArgumentValue(args, "--check-words");
        if (archive is not null)
        {
            return CheckWordsCommand.Run(
                archive,
                ArgumentValue(args, "--alignment-models") ?? modelDirectory,
                plan);
        }

        var replay = ArgumentValue(args, "--replay");
        if (replay is not null)
        {
            if (ArgumentValue(args, "--audio") is not { } replayAudio)
            {
                Console.Error.WriteLine("--replay needs --audio <wav> to align against.");
                return 1;
            }

            return ReplayCommand.Run(
                replay,
                replayAudio,
                ArgumentValue(args, "--alignment-models") ?? modelDirectory,
                plan);
        }

        var alignAudio = ArgumentValue(args, "--align");
        if (alignAudio is not null)
        {
            return AlignCommand.Run(
                alignAudio,
                ArgumentValue(args, "--alignment-models") ?? modelDirectory,
                plan,
                ArgumentValue(args, "--window"));
        }

        var audio = ArgumentValue(args, "--transcribe");
        if (audio is not null)
        {
            return await TranscribeCommand.RunAsync(
                    audio,
                    modelDirectory,
                    capabilities,
                    plan,
                    ArgumentValue(args, "--model-dir"),
                    ArgumentValue(args, "--engine"))
                .ConfigureAwait(false);
        }

        if (fetch)
        {
            return await FetchModels.RunAsync(
                    modelDirectory,
                    capabilities,
                    plan,
                    ArgumentValue(args, "--model"),
                    args.Contains("--force"),
                    alignment: !args.Contains("--no-alignment"))
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
        Console.WriteLine("  --no-alignment   With --fetch-models: skip the 602 MB word aligner.");
        Console.WriteLine("                   Speaker models are always fetched; they are small.");
        Console.WriteLine("  --transcribe <f> Transcribe a PCM WAV file and report what happened.");
        Console.WriteLine("  --transcribe-live <f>  Feed a WAV through the live path, as the microphone does.");
        Console.WriteLine("  --diarize <f>    Work out who spoke when in a WAV, and print the turns.");
        Console.WriteLine("  --speakers <n>   Upper bound on speakers for --diarize.");
        Console.WriteLine("  --threshold <d>  Cosine distance at which two voices are different people.");
        Console.WriteLine("  --sweep          With --diarize: try every threshold and show where the answer changes.");
        Console.WriteLine("  --tracking       With --diarize: follow speakers between windows instead of comparing voices.");
        Console.WriteLine("  --align <f>      Scan a WAV with the alignment model and read the grid back.");
        Console.WriteLine("  --window <a-b>   With --align: the seconds to decode, as 12.5-18.5.");
        Console.WriteLine("  --check-words <f>  Check a saved .scrb transcript against its own audio.");
        Console.WriteLine("  --speaker-model [n]  Show or switch which voice model is used. No name lists them.");
        Console.WriteLine("  --segmentation-model [n]  Show or switch what decides where speakers change.");
        Console.WriteLine("  --diarizer [n]   Show or switch how speakers are worked out: tracking or voices.");
        Console.WriteLine("  --speaker-models <f>  Compare embedding models on a saved .scrb transcript.");
        Console.WriteLine("  --candidates <d>   With --speaker-models: directory of model folders to try.");
        Console.WriteLine("  --model-dir <d>  Model directory for --transcribe, overriding the layout.");
        Console.WriteLine("  --engine <e>     Transcription engine for --transcribe: onnx (default) or");
        Console.WriteLine("                   whispercpp, which reads ggml-*.bin from <models>/whisper-cpp.");
        Console.WriteLine("  --strict         Refuse to let a requested provider quietly fall back to the CPU.");
        Console.WriteLine("  --live           Plan for live transcription rather than a batch pass.");
        Console.WriteLine("  --max            Plan for maximum speed rather than a considerate CPU share.");
        Console.WriteLine("  --help, -h       This text.");
    }

    /// <summary>
    /// The value following a named argument, or null when it has none.
    /// <para>
    /// A word beginning with two dashes is the next option, not this one's value. Without that,
    /// <c>--speaker-model --models .</c> reads "--models" as the name of a model and reports it
    /// as unknown instead of listing what there is.
    /// </para>
    /// </summary>
    private static string? ArgumentValue(string[] args, string name)
    {
        var index = Array.IndexOf(args, name);

        if (index < 0 || index + 1 >= args.Length)
        {
            return null;
        }

        var value = args[index + 1];

        return value.StartsWith("--", StringComparison.Ordinal) ? null : value;
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
