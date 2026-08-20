using System.Diagnostics;
using System.Runtime.InteropServices;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Models;

namespace LocalScribe.Onnx;

/// <summary>
/// Works out what the current machine can do.
/// <para>
/// Every check here is a heuristic over something observable, and each one is reported
/// separately rather than folded into a single verdict. That is on purpose: when the NPU turns
/// out to be unusable, the useful question is always <em>which</em> prerequisite is missing,
/// and a single boolean cannot answer it.
/// </para>
/// </summary>
public static class DeviceProbe
{
    /// <summary>
    /// Inspects the machine. Takes a second or so on Windows because it shells out for the
    /// processor name and device list, so call it once at startup and keep the result.
    /// </summary>
    /// <param name="modelDirectory">Where Whisper model assets are expected to live.</param>
    public static DeviceCapabilities Probe(string? modelDirectory = null)
    {
        var providers = OnnxSessionFactory.AvailableProviders();
        var processorName = ReadProcessorName();
        var family = ClassifySoc(processorName);

        return new DeviceCapabilities
        {
            SocName = processorName,
            Family = family,
            PerformanceCoreCount = Environment.ProcessorCount,
            TotalCoreCount = Environment.ProcessorCount,
            TotalMemoryBytes = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes,
            OnnxProviders = providers,
            QnnProviderPresent = providers.Contains("QNNExecutionProvider") || providers.Contains("QNN"),
            HexagonDriverPresent = family is not SocFamily.NonQualcomm && HasHexagonRuntime(),
            WhisperQnnAssetsPresent = HasWhisperAssets(modelDirectory, family),
            DirectMlPresent = providers.Contains("DmlExecutionProvider") || providers.Contains("DML"),
            LocalLanguageModelPresent = false, // Filled in asynchronously by the caller; see LocalLanguageModel.
            OnBattery = IsOnBattery(),
        };
    }

    /// <summary>
    /// Maps a processor name onto a family. Matching on the marketing name is crude, but it is
    /// what the OS actually exposes, and the distinction only needs to be good enough to pick
    /// the right precompiled model directory.
    /// </summary>
    internal static SocFamily ClassifySoc(string processorName)
    {
        if (processorName.Length == 0)
        {
            return SocFamily.Unknown;
        }

        var name = processorName.ToLowerInvariant();

        if (!name.Contains("snapdragon") && !name.Contains("qualcomm") && !name.Contains("oryon"))
        {
            // ARM parts from other vendors and every x86 chip land here.
            return name.Contains("intel") || name.Contains("amd") || name.Contains("apple")
                ? SocFamily.NonQualcomm
                : SocFamily.Unknown;
        }

        if (name.Contains("x2") || name.Contains("x2e") || name.Contains("x2p"))
        {
            return SocFamily.SnapdragonX2;
        }

        if (name.Contains("elite") || name.Contains("x1e"))
        {
            return SocFamily.SnapdragonXElite;
        }

        if (name.Contains("plus") || name.Contains("x1p"))
        {
            return SocFamily.SnapdragonXPlus;
        }

        return SocFamily.Unknown;
    }

    /// <summary>The directory name model assets for a given family are expected under.</summary>
    public static string AssetFolderFor(SocFamily family) => ModelLayout.ChipsetFolder(family);

    private static bool HasWhisperAssets(string? modelDirectory, SocFamily family)
    {
        if (modelDirectory is null || !Directory.Exists(modelDirectory))
        {
            return false;
        }

        // Precompiled QNN binaries are chipset-specific, so a set built for a different
        // Snapdragon is worse than none: it fails to load at the least convenient moment.
        // A non-Qualcomm machine has no chipset folder of its own and must not be told it
        // has NPU weights just because portable ones were fetched.
        if (ModelLayout.ChipsetFolder(family) == ModelLayout.PortableFolder)
        {
            return false;
        }

        var chipsetDirectory = Path.Combine(modelDirectory, AssetFolderFor(family));

        return Directory.Exists(chipsetDirectory) && ContainsWeights(chipsetDirectory);
    }

    /// <summary>
    /// True when a directory holds a Whisper pair, either directly or in a per-size
    /// subdirectory. Both layouts are accepted because the size a plan will ask for is not
    /// known here: it depends on where the encoder lands, which in turn depends on this
    /// answer. Asking only whether any usable set exists breaks that circle.
    /// </summary>
    private static bool ContainsWeights(string directory)
    {
        if (HasPair(directory))
        {
            return true;
        }

        return Directory.EnumerateDirectories(directory).Any(HasPair);
    }

    /// <summary>
    /// True when a directory holds both halves, in either layout the transcriber accepts:
    /// <c>encoder.onnx</c> beside <c>decoder.onnx</c>, or AI Hub's <c>encoder/model.onnx</c>
    /// and <c>decoder/model.onnx</c>. The two must agree, or a correctly laid out AI Hub set
    /// reports as missing and the planner routes around weights that are sitting right there.
    /// </summary>
    private static bool HasPair(string directory) =>
        HasHalf(directory, "encoder") && HasHalf(directory, "decoder");

    private static bool HasHalf(string directory, string half) =>
        File.Exists(Path.Combine(directory, $"{half}.onnx"))
        || File.Exists(Path.Combine(directory, half, "model.onnx"));

    /// <summary>
    /// Looks for the Hexagon NPU runtime. The library shipping beside the app is the strongest
    /// signal available without loading it, which would be slow and can crash on a bad install.
    /// </summary>
    private static bool HasHexagonRuntime()
    {
        if (!OperatingSystem.IsWindows())
        {
            return false;
        }

        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "QnnHtp.dll"),
            Path.Combine(Environment.SystemDirectory, "QnnHtp.dll"),
        };

        return candidates.Any(File.Exists);
    }

    /// <summary>
    /// Reads the processor's marketing name. On Windows this needs a CIM query, since the
    /// environment variable only reports the ARM architecture revision.
    /// </summary>
    private static string ReadProcessorName()
    {
        if (OperatingSystem.IsWindows())
        {
            var name = RunAndCapture(
                "powershell",
                "-NoProfile -NonInteractive -Command \"(Get-CimInstance Win32_Processor).Name\"");

            if (!string.IsNullOrWhiteSpace(name))
            {
                return name.Trim();
            }

            return Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "unknown";
        }

        if (OperatingSystem.IsLinux() && File.Exists("/proc/cpuinfo"))
        {
            foreach (var line in File.ReadLines("/proc/cpuinfo"))
            {
                if (line.StartsWith("model name", StringComparison.OrdinalIgnoreCase))
                {
                    return line[(line.IndexOf(':') + 1)..].Trim();
                }
            }
        }

        return RuntimeInformation.ProcessArchitecture.ToString();
    }

    /// <summary>
    /// True when running on battery. Used to take less CPU, since the cost of a slow machine is
    /// felt more sharply when it is also draining.
    /// </summary>
    private static bool IsOnBattery()
    {
        if (!OperatingSystem.IsWindows())
        {
            return false;
        }

        var status = RunAndCapture(
            "powershell",
            "-NoProfile -NonInteractive -Command \"(Get-CimInstance Win32_Battery).BatteryStatus\"");

        // BatteryStatus 1 means discharging. Anything else, including no battery at all, counts
        // as mains power for our purposes.
        return status.Trim() == "1";
    }

    private static string RunAndCapture(string fileName, string arguments)
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo(fileName, arguments)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            });

            if (process is null)
            {
                return string.Empty;
            }

            var output = process.StandardOutput.ReadToEnd();
            return process.WaitForExit(10_000) ? output : string.Empty;
        }
        catch (Exception exception) when (exception is System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            // A missing shell is not worth failing startup over; the caller degrades gracefully.
            return string.Empty;
        }
    }
}
