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
        var (whisperCppModel, whisperCppCoreMl) = WhisperCppAssets(modelDirectory);

        return new DeviceCapabilities
        {
            SocName = processorName,
            Platform = OperatingSystem.IsWindows() ? DevicePlatform.Windows
                : OperatingSystem.IsMacOS() ? DevicePlatform.MacOS
                : DevicePlatform.Linux,
            Family = family,
            PerformanceCoreCount = Environment.ProcessorCount,
            TotalCoreCount = Environment.ProcessorCount,
            TotalMemoryBytes = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes,
            OnnxProviders = providers,
            QnnProviderPresent = providers.Contains("QNNExecutionProvider") || providers.Contains("QNN"),
            HexagonDriverPresent = family is not SocFamily.NonQualcomm && HasHexagonRuntime(),
            WhisperQnnAssetsPresent = HasWhisperAssets(modelDirectory, family),
            DirectMlPresent = providers.Contains("DmlExecutionProvider") || providers.Contains("DML"),
            WhisperCppModelPresent = whisperCppModel,
            WhisperCppCoreMlEncoderPresent = whisperCppCoreMl,
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

    /// <summary>
    /// Whether whisper.cpp weights are on disk, and whether the Core ML encoder bundle is
    /// beside them. Observed on every platform — files present is files present — and left to
    /// the advisor to interpret, since only Apple silicon can use the bundle.
    /// </summary>
    private static (bool Model, bool CoreMl) WhisperCppAssets(string? modelDirectory)
    {
        if (modelDirectory is null)
        {
            return (false, false);
        }

        var directory = Path.Combine(modelDirectory, WhisperCppModelSource.DirectoryName);

        if (!Directory.Exists(directory))
        {
            return (false, false);
        }

        return (
            Directory.EnumerateFiles(directory, "ggml-*.bin").Any(),
            Directory.EnumerateDirectories(directory, "*-encoder.mlmodelc").Any());
    }

    private static bool HasWhisperAssets(string? modelDirectory, SocFamily family) =>
        ModelLayout.HasChipsetModels(modelDirectory, family);

    /// <summary>
    /// Looks for a working Hexagon NPU by asking Windows for the device itself.
    /// <para>
    /// An earlier version of this checked for <c>QnnHtp.dll</c> beside the binary. That was
    /// wrong in the worst possible direction: the ONNX Runtime QNN package <em>ships</em> that
    /// library into the output directory, so the check passed on every machine, including ones
    /// with no Qualcomm driver at all. A prerequisite check that always passes is worse than no
    /// check, because it sends you looking somewhere else.
    /// </para>
    /// <para>
    /// The device query is still only a strong hint. The definitive test is loading a model with
    /// <c>StrictProviderCheck</c> enabled and seeing whether it throws.
    /// </para>
    /// </summary>
    private static bool HasHexagonRuntime()
    {
        if (!OperatingSystem.IsWindows())
        {
            return false;
        }

        // Status is part of the test, not decoration: a driver present but in an error state is
        // exactly the situation that produces silent CPU fallback later.
        var query = RunAndCapture(
            "powershell",
            "-NoProfile -NonInteractive -Command \"Get-PnpDevice -PresentOnly "
            + "| Where-Object { $_.FriendlyName -match 'Hexagon|NPU|Neural' -and $_.Status -eq 'OK' } "
            + "| Select-Object -First 1 -ExpandProperty FriendlyName\"");

        return !string.IsNullOrWhiteSpace(query);
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

        if (OperatingSystem.IsMacOS())
        {
            // The environment reports only the architecture; the marketing name ("Apple M2")
            // lives in sysctl, and it is what ClassifySoc needs to file Apple silicon under
            // NonQualcomm rather than Unknown.
            var name = RunAndCapture("/usr/sbin/sysctl", "-n machdep.cpu.brand_string");

            if (!string.IsNullOrWhiteSpace(name))
            {
                return name.Trim();
            }
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
        if (OperatingSystem.IsMacOS())
        {
            // pmset names the active power source; "Battery Power" appears only when
            // discharging, which is the same line the planner draws on Windows.
            return RunAndCapture("/usr/bin/pmset", "-g batt")
                .Contains("Battery Power", StringComparison.OrdinalIgnoreCase);
        }

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
