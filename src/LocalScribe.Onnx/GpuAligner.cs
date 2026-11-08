using System.Runtime.InteropServices;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Models;
using Microsoft.ML.OnnxRuntime;

namespace LocalScribe.Onnx;

/// <summary>
/// The word aligner on an Apple-silicon Mac's GPU, through ONNX Runtime's Core ML provider.
/// <para>
/// On the Mac's CPU the scan is the longest stage of a run, and it competes with everything else
/// for the cores. On the GPU the full-precision model scanned the podcast in 47 s against the
/// CPU's 4-bit 168 s, back to back on the same machine, and closer to the fp16 reference (100% of
/// words within 0.1 s, against 99%; on the debate 97% against 92%). Six of its 1,391 nodes stay on
/// the CPU, the first feature-extractor convolutions, whose input is longer than Core ML takes, so
/// it still keeps about two cores busy. Measured with the doctor's <c>--aligner-coreml</c>; see
/// docs/model-trials.md. The Neural Engine was tried too and was no faster, took five minutes to
/// compile, and is where Whisper's encoder runs.
/// </para>
/// <para>
/// Core ML plans for exact shapes, so the scan uses one window length: 34 s, 30 kept and 2 s of
/// margin either side, the length that was measured. The model is first rewritten so Core ML can
/// keep its weights in binary (<see cref="GemmRewrite"/>); as exported it took two to three
/// minutes to open every time. Then it is opened once, which is the compile, and Core ML's result
/// kept beside it. All of that happens once, in the background: until it is done, and whenever
/// the GPU fails, the CPU scans exactly as before.
/// </para>
/// </summary>
public static class GpuAligner
{
    /// <summary>Samples per window: 34 s at 16 kHz, the one shape the compiled model accepts.</summary>
    public const int WindowSamples = 544_000;

    /// <summary>Recordings shorter than one window are scanned on the CPU.</summary>
    public const double ShortestSeconds = WindowSamples / 16_000.0;

    private const string DirectoryName = "gpu";
    private const string ModelName = "aligner.onnx";
    private const string CacheName = "coreml-cache";
    private const string SourceName = "model_fp16.onnx";
    private const string CompleteMarker = "compiled";

    /// <summary>Whether this machine should scan on the GPU: an Apple-silicon Mac, unless switched off.</summary>
    public static bool Applies =>
        OperatingSystem.IsMacOS()
        && RuntimeInformation.ProcessArchitecture == Architecture.Arm64
        && Environment.GetEnvironmentVariable("LOCALSCRIBE_GPU_ALIGNER") != "0";

    /// <summary>Whether the rewritten model and Core ML's compile of it are on disk and complete.</summary>
    public static bool IsReady(string alignmentDirectory) =>
        File.Exists(Path.Combine(alignmentDirectory, DirectoryName, CompleteMarker));

    /// <summary>
    /// Rewrites the aligner for Core ML and compiles it, fetching the fp16 build first where only
    /// the 4-bit one is installed. Returns true when it is ready.
    /// <para>
    /// Finished only when a marker is written after the compile, as for the NPU: anything cut
    /// short leaves no marker, and the next attempt clears it and starts over.
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
            log?.Invoke("fetching the fp16 aligner to run on the GPU");
            await new ModelFetcher()
                .FetchAsync(alignmentDirectory, [.. AlignmentModelSource.Files.Where(f => f.FileName == SourceName)], cancellationToken: cancellationToken)
                .ConfigureAwait(false);
        }

        var directory = Path.Combine(alignmentDirectory, DirectoryName);

        return await Task.Factory.StartNew(
            () =>
            {
                // Below the transcription and the window: nobody is waiting on this.
                Thread.CurrentThread.Priority = ThreadPriority.Lowest;

                Clear(directory);
                Directory.CreateDirectory(directory);
                var model = Path.Combine(directory, ModelName);
                var finished = false;

                try
                {
                    log?.Invoke("rewriting the aligner for Core ML");
                    if (GemmRewrite.RewriteFile(source, model) == 0)
                    {
                        log?.Invoke("the aligner has no MatMuls to rewrite; not compiling");
                        return false;
                    }

                    cancellationToken.ThrowIfCancellationRequested();
                    log?.Invoke("compiling the aligner for the GPU");

                    using (Open(alignmentDirectory, plan))
                    {
                        // Opening is the compile; Core ML's result is written to the cache.
                    }

                    File.WriteAllText(Path.Combine(directory, CompleteMarker), $"{WindowSamples}{Environment.NewLine}");
                    finished = true;
                    log?.Invoke("the aligner is compiled for the GPU");
                    return true;
                }
                finally
                {
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

    /// <summary>Opens the rewritten aligner on the GPU, or throws when Core ML will not take it.</summary>
    public static InferenceSession Open(string alignmentDirectory, ExecutionPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);

        var directory = Path.Combine(alignmentDirectory, DirectoryName);
        var cache = Path.Combine(directory, CacheName);
        Directory.CreateDirectory(cache);

        using var options = new SessionOptions
        {
            // For the six convolutions Core ML cannot take, under the same cap as everything else.
            IntraOpNumThreads = plan.CpuBudget.IntraOpThreads,
            InterOpNumThreads = plan.CpuBudget.InterOpThreads,
        };

        options.AddFreeDimensionOverrideByName("batch_size", 1);
        options.AddFreeDimensionOverrideByName("sequence_length", WindowSamples);

        options.AppendExecutionProvider("CoreML", new Dictionary<string, string>
        {
            ["ModelFormat"] = "MLProgram",
            ["MLComputeUnits"] = "CPUAndGPU",
            ["RequireStaticInputShapes"] = "1",
            ["ModelCacheDirectory"] = cache,
        });

        return new InferenceSession(Path.Combine(directory, ModelName), options);
    }

    /// <summary>
    /// Throws away a compiled aligner Core ML no longer opens — after a macOS update, say — so
    /// the next launch prepares a fresh one instead of failing the same way every run.
    /// </summary>
    public static void Forget(string alignmentDirectory)
    {
        var directory = Path.Combine(alignmentDirectory, DirectoryName);
        TryDelete(Path.Combine(directory, CompleteMarker));
        Clear(directory);
    }

    private static void Clear(string directory)
    {
        if (!Directory.Exists(directory))
        {
            return;
        }

        try
        {
            // The cache is a tree of Core ML's own; the model and marker sit beside it.
            Directory.Delete(directory, recursive: true);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            // Left for the next attempt, which clears the folder before it starts.
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
        }
    }
}
