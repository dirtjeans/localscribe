using LocalScribe.Core.Hardware;
using LocalScribe.Core.Models;

namespace LocalScribe.Doctor;

/// <summary>
/// Implements <c>--fetch-models</c>: downloads the portable Whisper export this machine will
/// actually open, and says plainly what it has and has not done.
/// <para>
/// The honesty matters here. This fetches ordinary ONNX exports, which serve the CPU and
/// DirectML paths. It cannot fetch the NPU path's weights, because precompiled QNN context
/// binaries are built per chipset and come from AI Hub behind an account. Leaving that
/// unsaid would recreate the problem the doctor exists to solve: a machine that looks fully
/// configured and is quietly running on the wrong processor.
/// </para>
/// </summary>
internal static class FetchModels
{
    public static async Task<int> RunAsync(
        string modelRoot,
        DeviceCapabilities capabilities,
        ExecutionPlan plan,
        string? requestedSize,
        bool force)
    {
        var size = requestedSize ?? plan.WhisperModel;

        if (!WhisperModelSource.IsSupported(size))
        {
            Console.Error.WriteLine(
                $"Unknown Whisper size '{size}'. Known sizes: "
                + $"{string.Join(", ", WhisperModelSource.SupportedSizes)}.");
            return 1;
        }

        // Portable weights, so the portable folder, whatever chip this is. Passing Cpu rather
        // than the planned device is deliberate: these files must never land where the probe
        // would mistake them for chipset binaries.
        var directory = ModelLayout.Resolve(modelRoot, capabilities.Family, ComputeDevice.Cpu, size);

        Console.WriteLine();
        Console.WriteLine($"Fetching Whisper {size} into {directory}");
        Console.WriteLine("These are portable ONNX exports for the CPU and DirectML paths.");
        Console.WriteLine();

        var fetcher = new WhisperModelFetcher();
        var progress = new ConsoleProgress();

        try
        {
            var results = await fetcher
                .FetchAsync(directory, size, progress, force)
                .ConfigureAwait(false);

            progress.Finish();

            Console.WriteLine();
            foreach (var result in results)
            {
                var note = result.Outcome switch
                {
                    FetchOutcome.Downloaded => $"downloaded, {Mib(result.Bytes)}",
                    FetchOutcome.AlreadyPresent => $"already present, {Mib(result.Bytes)}",
                    _ => "not published for this export, skipped",
                };
                Console.WriteLine($"  {result.FileName,-20} {note}");
            }
        }
        catch (HttpRequestException exception)
        {
            progress.Finish();
            Console.Error.WriteLine();
            Console.Error.WriteLine($"Download failed: {exception.Message}");
            Console.Error.WriteLine("Nothing was left half-written; run the command again to retry.");
            return 1;
        }
        catch (IOException exception)
        {
            progress.Finish();
            Console.Error.WriteLine();
            Console.Error.WriteLine($"Could not write to {directory}: {exception.Message}");
            return 1;
        }

        Console.WriteLine();
        Console.WriteLine("Done. The CPU and GPU paths can now run.");

        ReportNpuGap(capabilities);
        return 0;
    }

    /// <summary>
    /// Says what this command could not do, and only when it is actually relevant. On a
    /// machine with no Hexagon NPU the AI Hub instructions are noise.
    /// </summary>
    private static void ReportNpuGap(DeviceCapabilities capabilities)
    {
        if (ModelLayout.ChipsetFolder(capabilities.Family) == ModelLayout.PortableFolder)
        {
            return;
        }

        var chipset = ModelLayout.ChipsetFolder(capabilities.Family);

        Console.WriteLine();
        Console.WriteLine("The NPU path still needs weights this command cannot fetch.");
        Console.WriteLine(
            "  Precompiled QNN context binaries are built for one chipset and are not");
        Console.WriteLine(
            "  published for download. Export them from Qualcomm AI Hub, which needs a free");
        Console.WriteLine("  account:");
        Console.WriteLine();
        Console.WriteLine("    pip install qai-hub-models");
        Console.WriteLine("    python -m qai_hub_models.models.whisper_base_en.export \\");
        Console.WriteLine($"      --chipset qualcomm-{chipset} \\");
        Console.WriteLine("      --target-runtime precompiled_qnn_onnx \\");
        Console.WriteLine("      --components HfWhisperEncoder HfWhisperDecoder");
        Console.WriteLine();
        Console.WriteLine($"  Then place encoder.onnx and decoder.onnx under models/{chipset}/<size>/,");
        Console.WriteLine("  copying vocab.json across from the portable set fetched just now.");
    }

    /// <summary>
    /// Sizes in the unit that shows something. added_tokens.json is 2 KiB, and reporting it as
    /// "0.0 MiB" reads like a failed download.
    /// </summary>
    private static string Mib(long bytes) => bytes < 1024 * 1024
        ? $"{bytes / 1024.0:F0} KiB"
        : $"{bytes / (1024.0 * 1024):F1} MiB";

    /// <summary>
    /// Single-line progress. Rewrites one line per file rather than scrolling, and degrades to
    /// silence when output is redirected, where carriage returns produce unreadable logs.
    /// </summary>
    private sealed class ConsoleProgress : IProgress<FetchProgress>
    {
        private readonly bool _interactive = !Console.IsOutputRedirected;
        private string _current = string.Empty;
        private int _lastPercent = -1;
        private bool _dirty;

        public void Report(FetchProgress value)
        {
            if (value.FileName != _current)
            {
                Finish();
                _current = value.FileName;
                _lastPercent = -1;
            }

            if (value.Done)
            {
                Finish();
                return;
            }

            if (!_interactive)
            {
                return;
            }

            var percent = value.TotalBytes > 0
                ? (int)(100 * value.BytesRead / value.TotalBytes)
                : -1;

            // Redrawing on every 80 KiB buffer wastes far more time than it reports.
            if (percent == _lastPercent)
            {
                return;
            }

            _lastPercent = percent;
            _dirty = true;

            var shown = percent >= 0
                ? $"{percent,3}%"
                : $"{value.BytesRead / (1024.0 * 1024):F1} MiB";

            Console.Write($"\r  {value.FileName,-20} {shown}   ");
        }

        public void Finish()
        {
            if (_dirty)
            {
                Console.WriteLine();
                _dirty = false;
            }
        }
    }
}
