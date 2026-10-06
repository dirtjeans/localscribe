using LocalScribe.Core.Hardware;
using LocalScribe.Core.Models;
using Microsoft.ML.OnnxRuntime;

namespace LocalScribe.Onnx;

/// <summary>
/// The word aligner compiled for the Hexagon NPU, where Whisper runs.
/// <para>
/// On the CPU the aligner's scan is the longest stage of a Snapdragon run and the one that
/// decides when the transcript becomes clickable. On the NPU it is 1.3 s per window and leaves
/// the CPU to the speaker model; measured by the doctor's <c>--aligner-npu</c>, its word times
/// match the CPU's fp16 reference — 99% within 0.1 s on the podcast, 93% on the debate, every
/// fifth free of drift — and a seven-minute podcast scans in about a minute even while Whisper
/// shares the NPU.
/// </para>
/// <para>
/// The NPU compiles a graph for one exact input shape, so the scan uses one window length: 14 s,
/// 10 s kept and 2 s of margin either side. Longer windows were tried and could not be compiled
/// — 34 s took 33 GB and never finished. Compiling even this one takes four to five minutes and
/// about 8 GB of memory, which is why it happens once, in the background, on machines with
/// memory to spare, and is kept: until it exists, the CPU scans exactly as before.
/// </para>
/// <para>
/// The compile starts from the fp16 build with its GELUs rewritten (<see cref="GeluRewrite"/>),
/// because the provider cannot run <c>Erf</c>. The rewritten copy is deleted once compiled.
/// </para>
/// </summary>
public static class NpuAligner
{
    /// <summary>Samples per window: 14 s at 16 kHz, the one shape the compiled graph accepts.</summary>
    public const int WindowSamples = 224_000;

    /// <summary>Recordings shorter than one window are scanned on the CPU.</summary>
    public const double ShortestSeconds = WindowSamples / 16_000.0;

    private const string DirectoryName = "npu";
    private const string CompiledName = "aligner.onnx";
    private const string SourceName = "model_fp16.onnx";
    private const string CompleteMarker = "compiled";

    /// <summary>
    /// Whether this machine should have the aligner on its NPU: the Qualcomm provider took
    /// Whisper's encoder, and the user has not switched it off.
    /// </summary>
    /// <remarks>
    /// Requires the QNN provider specifically. On a Mac the plan also says NPU, and means the
    /// Neural Engine through Core ML, which this compiled graph has nothing to do with.
    /// </remarks>
    public static bool Applies(ExecutionPlan plan) =>
        plan.Encoder.Device == ComputeDevice.Npu
        && plan.Encoder.ExecutionProvider == AcceleratorPlanner.QnnProvider
        && Environment.GetEnvironmentVariable("LOCALSCRIBE_NPU_ALIGNER") != "0";

    /// <summary>Whether the compiled aligner is on disk, complete, and can be opened.</summary>
    public static bool IsReady(string alignmentDirectory) =>
        File.Exists(Path.Combine(alignmentDirectory, DirectoryName, CompleteMarker));

    /// <summary>
    /// Compiles the aligner for the NPU, fetching the fp16 build first if only the 4-bit one is
    /// installed. Returns true when the compiled graph is ready.
    /// <para>
    /// Finished only when a marker is written after the graph: a compile cut short — the app
    /// closed, memory ran out, the driver refused — leaves no marker, so nothing mistakes it for
    /// a finished one, and the next attempt clears it and starts over. A marker rather than
    /// compiling elsewhere and moving the folder into place, because the move is what fails:
    /// a folder under OneDrive or an indexer is held open for a while after its files close.
    /// </para>
    /// </summary>
    public static async Task<bool> PrepareAsync(
        string alignmentDirectory,
        ExecutionPlan plan,
        Action<string>? log = null,
        CancellationToken cancellationToken = default)
    {
        if (IsReady(alignmentDirectory))
        {
            return true;
        }

        var source = Path.Combine(alignmentDirectory, SourceName);
        if (!File.Exists(source))
        {
            log?.Invoke("fetching the fp16 aligner to compile for the NPU");
            await new ModelFetcher()
                .FetchAsync(alignmentDirectory, [.. AlignmentModelSource.Files.Where(f => f.FileName == SourceName)], cancellationToken: cancellationToken)
                .ConfigureAwait(false);
        }

        var directory = Path.Combine(alignmentDirectory, DirectoryName);

        return await Task.Factory.StartNew(
            () =>
            {
                // Below the transcription and the window: this runs once and nobody is waiting on
                // it, and it holds a core for minutes.
                Thread.CurrentThread.Priority = ThreadPriority.Lowest;

                Clear(directory);
                Directory.CreateDirectory(directory);
                var rewritten = Path.Combine(directory, "gelu.onnx");
                var compiled = Path.Combine(directory, CompiledName);
                var finished = false;

                try
                {
                    log?.Invoke("rewriting the aligner's GELUs for the NPU");
                    if (GeluRewrite.RewriteFile(source, rewritten) == 0)
                    {
                        log?.Invoke("the aligner has no Erf GELUs to rewrite; not compiling");
                        return false;
                    }

                    cancellationToken.ThrowIfCancellationRequested();
                    log?.Invoke("compiling the aligner for the NPU");

                    using (var options = Options(plan, writeCacheTo: compiled))
                    using (new InferenceSession(rewritten, options))
                    {
                        // Opening is the compile; the graph is written as it finishes.
                    }

                    if (!File.Exists(compiled) || !Directory.EnumerateFiles(directory, "*.bin").Any())
                    {
                        log?.Invoke("the compile finished without writing its graph");
                        return false;
                    }

                    File.WriteAllText(Path.Combine(directory, CompleteMarker), $"{WindowSamples}{Environment.NewLine}");
                    finished = true;
                    log?.Invoke("the aligner is compiled for the NPU");
                    return true;
                }
                finally
                {
                    TryDelete(rewritten);

                    if (!finished)
                    {
                        Clear(directory);
                    }
                }
            },
            cancellationToken,
            TaskCreationOptions.LongRunning,
            TaskScheduler.Default).ConfigureAwait(false);
    }

    /// <summary>Opens the compiled aligner, or throws when the NPU will not take it.</summary>
    public static InferenceSession Open(string alignmentDirectory, ExecutionPlan plan)
    {
        using var options = Options(plan, writeCacheTo: null);
        return new InferenceSession(Path.Combine(alignmentDirectory, DirectoryName, CompiledName), options);
    }

    /// <summary>
    /// Throws away a compiled aligner the NPU no longer accepts — after a driver update, say —
    /// so the next launch compiles a fresh one instead of failing the same way every run.
    /// </summary>
    public static void Forget(string alignmentDirectory)
    {
        // The marker first, so a graph whose files are still held open is at least not trusted.
        var directory = Path.Combine(alignmentDirectory, DirectoryName);
        TryDelete(Path.Combine(directory, CompleteMarker));
        Clear(directory);
    }

    /// <summary>Removes what a compile left, as far as the file system allows right now.</summary>
    private static void Clear(string directory)
    {
        if (!Directory.Exists(directory))
        {
            return;
        }

        // Files one by one rather than the folder: the files are all that matters, and a folder
        // something else has open refuses to go when its contents would not have.
        foreach (var file in Directory.EnumerateFiles(directory))
        {
            TryDelete(file);
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            // Left for the next attempt, which clears the folder before it starts.
        }
    }

    private static SessionOptions Options(ExecutionPlan plan, string? writeCacheTo)
    {
        var options = new SessionOptions();

        // The export leaves both dimensions free; the NPU compiles for exact ones.
        options.AddFreeDimensionOverrideByName("batch_size", 1);
        options.AddFreeDimensionOverrideByName("sequence_length", WindowSamples);

        // Entirely on the NPU or not at all. A graph split across the two would be slower than
        // the CPU alone, and the CPU path is right there.
        options.AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1");

        if (writeCacheTo is not null)
        {
            options.AddSessionConfigEntry("ep.context_enable", "1");
            options.AddSessionConfigEntry("ep.context_file_path", writeCacheTo);

            // The compiled graph beside its description rather than inside it: 750 MB in a
            // protobuf would have to be read whole to open it.
            options.AddSessionConfigEntry("ep.context_embed_mode", "0");
        }

        options.AppendExecutionProvider("QNN", new Dictionary<string, string>
        {
            ["backend_path"] = "QnnHtp.dll",
            ["htp_performance_mode"] = plan.NpuPower == NpuPower.Balanced ? "balanced" : "burst",
            ["enable_htp_fp16_precision"] = "1",
        });

        return options;
    }
}
