using System.Diagnostics;
using LocalScribe.Core.Audio;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalScribe.Doctor;

/// <summary>
/// Implements <c>--aligner-npu &lt;wav&gt;</c>: can the MMS aligner run on the Hexagon NPU, and is
/// it worth it?
/// <para>
/// The go/no-go before any app code changes. Loads the fp16 aligner through the QNN provider at
/// one fixed window length — the NPU compiles a graph for exact shapes — with the CPU fallback
/// switched off, so it either runs entirely on the NPU or fails saying so. Then scores every
/// window of a recording, times it against the CPU at the same windows, and compares what the
/// two make of each frame, because a faster scan that hears different letters is not a scan.
/// </para>
/// <para>
/// Windows are always full length and always real audio: the first and last slide inward rather
/// than being padded. The model is normalised per window and attends across all of it, so
/// padding with silence would change what it hears in the part that is kept.
/// </para>
/// </summary>
internal static class AlignerNpuCommand
{
    private const int SampleRate = 16_000;
    private const double MarginSeconds = 2;
    private const int SamplesPerFrame = 320;

    public static int Run(string wavPath, string modelRoot, int cpuThreads, double windowSeconds = 30)
    {
        var model = Path.Combine(modelRoot, "alignment", "model_fp16.onnx");
        if (!File.Exists(model))
        {
            Console.Error.WriteLine($"No fp16 aligner at {model}.");
            return 1;
        }

        var audio = WavReader.Read(wavPath);
        audio.EnsureWhisperFormat();

        var length = (int)((windowSeconds + (2 * MarginSeconds)) * SampleRate);
        if (audio.Samples.Length < length)
        {
            Console.Error.WriteLine("The recording is shorter than one window; the NPU path would not apply.");
            return 1;
        }

        var windows = Windows(audio.Samples.Length, length, windowSeconds);

        Console.WriteLine();
        Console.WriteLine($"Aligner on the NPU — {Path.GetFileName(wavPath)}, {audio.DurationSeconds:F0} s, "
            + $"{windows.Count} windows of {length / (double)SampleRate:F0} s");
        Console.WriteLine();

        // The NPU session first, so a refusal is the first thing reported. It may read a
        // rewritten copy of the model (LOCALSCRIBE_NPU_MODEL); the CPU reference never does.
        var npuModel = Environment.GetEnvironmentVariable("LOCALSCRIBE_NPU_MODEL") is { Length: > 0 } rewritten
            ? rewritten
            : model;

        // One cache per model and window length: the compiled graph is for exactly one shape.
        var cache = Path.Combine(
            Path.GetDirectoryName(npuModel)!,
            $"{Path.GetFileNameWithoutExtension(npuModel)}.qnn-{length}.onnx");
        var cached = File.Exists(cache);
        var watch = Stopwatch.StartNew();
        InferenceSession npu;

        Console.WriteLine($"  Model   {npuModel}");

        try
        {
            npu = OpenNpu(cached ? cache : npuModel, length, cached ? null : cache);
        }
        catch (Exception exception)
        {
            Console.WriteLine($"  NPU     refused: {exception.GetType().Name}: {exception.Message}");
            return 1;
        }

        Console.WriteLine($"  NPU     opened in {watch.Elapsed.TotalSeconds:F1} s "
            + $"({(cached ? "from the compiled cache" : "compiled now, cache written")})");

        using (npu)
        {
            using var cpuOptions = new SessionOptions { IntraOpNumThreads = cpuThreads, InterOpNumThreads = 1 };
            using var cpu = new InferenceSession(model, cpuOptions);

            var input = npu.InputMetadata.Keys.First();
            var npuTime = TimeSpan.Zero;
            var cpuTime = TimeSpan.Zero;
            long frames = 0, agree = 0;
            double worst = 0, total = 0;

            foreach (var (from, index) in windows.Select((w, i) => (w, i)))
            {
                var normalised = Normalise(audio.Samples.AsSpan(from, length));

                watch.Restart();
                var onNpu = Logits(npu, input, normalised);
                npuTime += watch.Elapsed;

                // NPU alone, for timing it against something else that wants the NPU.
                if (Environment.GetEnvironmentVariable("LOCALSCRIBE_NPU_ONLY") is { Length: > 0 })
                {
                    continue;
                }

                watch.Restart();
                var onCpu = Logits(cpu, input, normalised);
                cpuTime += watch.Elapsed;

                if (onNpu.Frames != onCpu.Frames || onNpu.Alphabet != onCpu.Alphabet)
                {
                    Console.WriteLine($"  window {index}: shapes differ, NPU {onNpu.Frames}x{onNpu.Alphabet}, CPU {onCpu.Frames}x{onCpu.Alphabet}");
                    return 1;
                }

                var a = LogSoftmax(onNpu.Values, onNpu.Frames, onNpu.Alphabet);
                var b = LogSoftmax(onCpu.Values, onCpu.Frames, onCpu.Alphabet);

                for (var t = 0; t < onNpu.Frames; t++)
                {
                    var row = t * onNpu.Alphabet;
                    var bestA = 0;
                    var bestB = 0;

                    for (var k = 0; k < onNpu.Alphabet; k++)
                    {
                        if (a[row + k] > a[row + bestA]) bestA = k;
                        if (b[row + k] > b[row + bestB]) bestB = k;

                        // Compared where the probability is not negligible; below e^-10 a
                        // difference changes nothing the aligner does.
                        if (b[row + k] > -10)
                        {
                            var difference = Math.Abs(a[row + k] - b[row + k]);
                            worst = Math.Max(worst, difference);
                            total += difference;
                        }
                    }

                    frames++;
                    agree += bestA == bestB ? 1 : 0;
                }
            }

            Console.WriteLine($"  NPU     {npuTime.TotalSeconds,6:F1} s for all windows ({npuTime.TotalMilliseconds / windows.Count:F0} ms each)");
            Console.WriteLine($"  CPU     {cpuTime.TotalSeconds,6:F1} s at {cpuThreads} threads ({cpuTime.TotalMilliseconds / windows.Count:F0} ms each)");
            Console.WriteLine($"  Same most-likely letter on {agree} of {frames} frames ({agree / (double)frames:P2})");
            Console.WriteLine($"  Log-probability difference: worst {worst:F3}, mean {total / Math.Max(1, frames):F4} per frame");
        }

        return 0;
    }

    /// <summary>
    /// Grades the NPU where it counts: places a saved transcript's words on a grid scored by the
    /// NPU, with the app's own aligner, and compares every word with the CPU reference — the
    /// same yardstick as <c>--aligner-trial</c>. Letters per frame can disagree and the words
    /// still land in the same place, because the aligner knows which words to look for.
    /// </summary>
    /// <param name="coreMlUnits">
    /// Null for the NPU. Otherwise the Mac's Core ML provider instead, on these compute units
    /// (CPUAndGPU, CPUAndNeuralEngine or ALL), with the CPU's 4-bit scan timed beside it, since
    /// that is what the Mac runs today. The question there is whether the full-size model on the
    /// GPU can take the scan off the CPU, where it competes with everything else.
    /// </param>
    public static int Grade(string archivePath, string modelRoot, Core.Hardware.ExecutionPlan plan, double windowSeconds, string? coreMlUnits = null)
    {
        var contents = Core.Archive.TranscriptArchive.Load(archivePath);
        var audio = contents.Audio;
        var segments = contents.Segments;
        var directory = Path.Combine(modelRoot, "alignment");

        Console.WriteLine();
        var device = coreMlUnits is null ? "NPU" : $"Core ML {coreMlUnits}";

        // LOCALSCRIBE_OPEN_ONLY: how long Core ML takes to open the model, and nothing else,
        // because that turned out to be the question.
        if (coreMlUnits is not null && Environment.GetEnvironmentVariable("LOCALSCRIBE_OPEN_ONLY") is { Length: > 0 })
        {
            var timer = Stopwatch.StartNew();
            using var session = OpenCoreMl(Environment.GetEnvironmentVariable("LOCALSCRIBE_NPU_MODEL") ?? Path.Combine(directory, "model_fp16.onnx"), (int)((windowSeconds + (2 * MarginSeconds)) * SampleRate), coreMlUnits);
            Console.WriteLine($"  {device} opened in {timer.Elapsed.TotalSeconds:F1} s");
            return 0;
        }
        Console.WriteLine($"Aligner on the {device}, graded — {Path.GetFileName(archivePath)}, {audio.DurationSeconds:F0} s");
        Console.WriteLine();

        var reference = AlignerTrialCommand.RunMms(directory, audio, segments, plan, out var referenceSeconds, "model_fp16.onnx");
        if (reference is null)
        {
            Console.Error.WriteLine("The reference aligner produced no words.");
            return 1;
        }

        Console.WriteLine($"  {"MMS fp16, CPU (reference)",-28} {referenceSeconds,6:F1} s   {reference.Count} words");

        if (coreMlUnits is not null && File.Exists(Path.Combine(directory, "model_q4.onnx")))
        {
            var current = AlignerTrialCommand.RunMms(directory, audio, segments, plan, out var currentSeconds, "model_q4.onnx");
            AlignerTrialCommand.Report("MMS q4, CPU (the app today)", currentSeconds, current, reference);
        }

        var length = (int)((windowSeconds + (2 * MarginSeconds)) * SampleRate);
        var npuModel = Environment.GetEnvironmentVariable("LOCALSCRIBE_NPU_MODEL") is { Length: > 0 } rewritten
            ? rewritten
            : Path.Combine(directory, "model_fp16.onnx");
        var cache = Path.Combine(
            Path.GetDirectoryName(npuModel)!,
            $"{Path.GetFileNameWithoutExtension(npuModel)}.qnn-{length}.onnx");

        var watch = Stopwatch.StartNew();
        using var npu = coreMlUnits is null
            ? OpenNpu(File.Exists(cache) ? cache : npuModel, length, File.Exists(cache) ? null : cache)
            : OpenCoreMl(npuModel, length, coreMlUnits);
        var opened = watch.Elapsed.TotalSeconds;
        watch.Restart();

        // CPU time as well as wall time: the point of an accelerator here is the cores it frees.
        var cpuBefore = Process.GetCurrentProcess().TotalProcessorTime;

        var input = npu.InputMetadata.Keys.First();
        var frames = audio.Samples.Length / SamplesPerFrame;
        var core = (int)(windowSeconds * SampleRate) / SamplesPerFrame;
        var margin = (int)(MarginSeconds * SampleRate) / SamplesPerFrame;
        var lastStart = (audio.Samples.Length - length) / SamplesPerFrame;
        AlignmentScoresBuilder? grid = null;

        for (var at = 0; at < frames; at += core)
        {
            // Full-length windows of real audio, slid inward at the ends, on the frame grid.
            var readFrom = Math.Clamp(at - margin, 0, lastStart);
            var logits = Logits(npu, input, Normalise(audio.Samples.AsSpan(readFrom * SamplesPerFrame, length)));
            var scores = LogSoftmax(logits.Values, logits.Frames, logits.Alphabet);

            grid ??= new AlignmentScoresBuilder(frames, logits.Alphabet, SamplesPerFrame / (double)SampleRate);

            var offset = at - readFrom;
            var rows = Math.Min(Math.Min(core, frames - at), logits.Frames - offset);
            if (rows > 0)
            {
                grid.Scores.Fill(at, scores.AsSpan(offset * logits.Alphabet, rows * logits.Alphabet));
            }
        }

        var scanned = watch.Elapsed.TotalSeconds;
        var cpuSeconds = (Process.GetCurrentProcess().TotalProcessorTime - cpuBefore).TotalSeconds;

        using var aligner = LocalScribe.Onnx.ForcedAligner.Load(directory, plan, "model_fp16.onnx");
        var placed = aligner.AlignAll(grid!.Scores, segments);

        var words = placed
            .Where(w => w is not null)
            .SelectMany(w => w!)
            .Where(w => w.EndSeconds > w.StartSeconds)
            .Select(w => new AlignerTrialCommand.Timed(AlignerTrialCommand.Fold(w.Text), w.StartSeconds))
            .Where(w => w.Word.Length > 0)
            .ToList();

        Console.WriteLine($"  {device} opened in {opened:F1} s; scanned in {scanned:F1} s, using {cpuSeconds:F1} s of CPU time");
        AlignerTrialCommand.Report($"MMS fp16, {device} ({windowSeconds:F0} s windows)", scanned, words, reference);
        return 0;
    }

    /// <summary>
    /// The fp16 aligner through Core ML, at one fixed window length: Core ML plans for exact
    /// shapes as the NPU does. The CPU fallback stays off unless LOCALSCRIBE_ALLOW_FALLBACK is
    /// set, so a graph Core ML takes only in part is reported rather than timed as if it ran there;
    /// LOCALSCRIBE_QNN_VERBOSE prints how many nodes Core ML took.
    /// </summary>
    private static InferenceSession OpenCoreMl(string modelPath, int length, string units)
    {
        var options = new SessionOptions();

        if (Environment.GetEnvironmentVariable("LOCALSCRIBE_QNN_VERBOSE") is { Length: > 0 })
        {
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
        }

        options.AddFreeDimensionOverrideByName("batch_size", 1);
        options.AddFreeDimensionOverrideByName("sequence_length", length);

        if (Environment.GetEnvironmentVariable("LOCALSCRIBE_ALLOW_FALLBACK") is not { Length: > 0 })
        {
            options.AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1");
        }

        // The compiled Core ML model is cached beside the system's own caches, keyed by the
        // model, so a second run times the scan rather than the compile.
        var cache = Path.Combine(Path.GetTempPath(), "localscribe-coreml-cache");
        Directory.CreateDirectory(cache);

        options.AppendExecutionProvider("CoreML", new Dictionary<string, string>
        {
            ["ModelFormat"] = Environment.GetEnvironmentVariable("LOCALSCRIBE_COREML_FORMAT") ?? "MLProgram",
            ["MLComputeUnits"] = units,
            ["RequireStaticInputShapes"] = "1",
            ["ModelCacheDirectory"] = cache,
        });

        return new InferenceSession(modelPath, options);
    }

    /// <summary>
    /// Grades the app's own NPU path — <see cref="LocalScribe.Onnx.NpuAligner"/> preparing the
    /// compiled graph if it is missing, then <see cref="LocalScribe.Onnx.ForcedAligner.LoadNpu"/>
    /// scanning — against the CPU's fp16 reference, so what ships is what was measured.
    /// </summary>
    /// <param name="gpu">The Mac's GPU path (<see cref="LocalScribe.Onnx.GpuAligner"/>) instead of the NPU's.</param>
    public static int GradeApp(string archivePath, string modelRoot, Core.Hardware.ExecutionPlan plan, bool gpu = false)
    {
        var device = gpu ? "GPU" : "NPU";

        var contents = Core.Archive.TranscriptArchive.Load(archivePath);
        var audio = contents.Audio;
        var segments = contents.Segments;
        var directory = Path.Combine(modelRoot, "alignment");

        Console.WriteLine();
        Console.WriteLine($"Aligner on the {device}, the app's path — {Path.GetFileName(archivePath)}, {audio.DurationSeconds:F0} s");
        Console.WriteLine();

        var watch = Stopwatch.StartNew();
        var isReady = gpu ? LocalScribe.Onnx.GpuAligner.IsReady(directory) : LocalScribe.Onnx.NpuAligner.IsReady(directory);
        if (!isReady)
        {
            void Log(string message) => Console.WriteLine($"  {watch.Elapsed.TotalSeconds,6:F1} s  {message}");
            var ready = (gpu
                    ? LocalScribe.Onnx.GpuAligner.PrepareAsync(directory, plan, Log)
                    : LocalScribe.Onnx.NpuAligner.PrepareAsync(directory, plan, Log))
                .GetAwaiter().GetResult();

            if (!ready)
            {
                Console.Error.WriteLine($"The aligner could not be compiled for the {device}.");
                return 1;
            }
        }

        var reference = AlignerTrialCommand.RunMms(directory, audio, segments, plan, out var referenceSeconds, "model_fp16.onnx");
        if (reference is null)
        {
            Console.Error.WriteLine("The reference aligner produced no words.");
            return 1;
        }

        Console.WriteLine($"  {"MMS fp16, CPU (reference)",-28} {referenceSeconds,6:F1} s   {reference.Count} words");

        watch.Restart();
        using var aligner = gpu
            ? LocalScribe.Onnx.ForcedAligner.LoadGpu(directory, plan)
            : LocalScribe.Onnx.ForcedAligner.LoadNpu(directory, plan);
        var opened = watch.Elapsed.TotalSeconds;
        watch.Restart();

        var cpuBefore = Process.GetCurrentProcess().TotalProcessorTime;
        var scores = aligner.Scan(audio);
        var scanned = watch.Elapsed.TotalSeconds;
        var cpuSeconds = (Process.GetCurrentProcess().TotalProcessorTime - cpuBefore).TotalSeconds;

        if (scores is null)
        {
            Console.Error.WriteLine($"The {device} scan returned nothing.");
            return 1;
        }

        var words = aligner.AlignAll(scores, segments)
            .Where(w => w is not null)
            .SelectMany(w => w!)
            .Where(w => w.EndSeconds > w.StartSeconds)
            .Select(w => new AlignerTrialCommand.Timed(AlignerTrialCommand.Fold(w.Text), w.StartSeconds))
            .Where(w => w.Word.Length > 0)
            .ToList();

        Console.WriteLine($"  {device} opened in {opened:F1} s; scanned in {scanned:F1} s, using {cpuSeconds:F1} s of CPU time");
        AlignerTrialCommand.Report($"MMS fp16, {device} (app)", scanned, words, reference);
        return 0;
    }

    /// <summary>Holds the grid being filled; a class so the loop can create it on first sight of the alphabet.</summary>
    private sealed class AlignmentScoresBuilder(int frames, int alphabet, double frameSeconds)
    {
        public Core.Alignment.AlignmentScores Scores { get; } = new(frames, alphabet, frameSeconds);
    }

    private static InferenceSession OpenNpu(string modelPath, int length, string? writeCacheTo)
    {
        var options = new SessionOptions();

        // QNN reports why a graph will not finalize only at verbose level.
        if (Environment.GetEnvironmentVariable("LOCALSCRIBE_QNN_VERBOSE") is { Length: > 0 })
        {
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
        }

        // The NPU compiles for exact shapes; the export leaves both dimensions free.
        options.AddFreeDimensionOverrideByName("batch_size", 1);
        options.AddFreeDimensionOverrideByName("sequence_length", length);

        // Entirely on the NPU or not at all: a silent CPU fallback would time the CPU.
        options.AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1");

        if (writeCacheTo is not null)
        {
            // The compile is the slow part and depends only on the model and the shape, so its
            // result is kept, as the Whisper build ships precompiled.
            options.AddSessionConfigEntry("ep.context_enable", "1");
            options.AddSessionConfigEntry("ep.context_file_path", writeCacheTo);
            options.AddSessionConfigEntry("ep.context_embed_mode", "0");
        }

        options.AppendExecutionProvider("QNN", new Dictionary<string, string>
        {
            ["backend_path"] = "QnnHtp.dll",
            ["htp_performance_mode"] = "burst",
            ["enable_htp_fp16_precision"] = "1",
        });

        return new InferenceSession(modelPath, options);
    }

    /// <summary>Window starts: one per window length, each read 2 s early and slid inward at the ends.</summary>
    private static List<int> Windows(int samples, int length, double windowSeconds)
    {
        var starts = new List<int>();
        var core = (int)(windowSeconds * SampleRate) / SamplesPerFrame * SamplesPerFrame;
        var margin = (int)(MarginSeconds * SampleRate) / SamplesPerFrame * SamplesPerFrame;

        for (var at = 0; at < samples; at += core)
        {
            starts.Add(Math.Clamp(at - margin, 0, samples - length));
        }

        return starts;
    }

    private static (float[] Values, int Frames, int Alphabet) Logits(InferenceSession session, string input, float[] samples)
    {
        using var outputs = session.Run(
        [
            NamedOnnxValue.CreateFromTensor(input, new DenseTensor<float>(samples, [1, samples.Length])),
        ]);

        var logits = outputs.First().AsTensor<float>();
        return (logits.ToArray(), logits.Dimensions[1], logits.Dimensions[2]);
    }

    private static float[] Normalise(ReadOnlySpan<float> samples)
    {
        var mean = 0.0;
        foreach (var sample in samples)
        {
            mean += sample;
        }

        mean /= samples.Length;

        var variance = 0.0;
        foreach (var sample in samples)
        {
            variance += (sample - mean) * (sample - mean);
        }

        var deviation = Math.Sqrt((variance / samples.Length) + 1e-7);
        var scaled = new float[samples.Length];

        for (var i = 0; i < samples.Length; i++)
        {
            scaled[i] = (float)((samples[i] - mean) / deviation);
        }

        return scaled;
    }

    private static float[] LogSoftmax(float[] logits, int frames, int alphabet)
    {
        var scores = new float[logits.Length];

        for (var t = 0; t < frames; t++)
        {
            var row = t * alphabet;
            var largest = float.NegativeInfinity;

            for (var k = 0; k < alphabet; k++)
            {
                largest = Math.Max(largest, logits[row + k]);
            }

            var sum = 0.0;
            for (var k = 0; k < alphabet; k++)
            {
                sum += Math.Exp(logits[row + k] - largest);
            }

            var offset = largest + Math.Log(sum);
            for (var k = 0; k < alphabet; k++)
            {
                scores[row + k] = (float)(logits[row + k] - offset);
            }
        }

        return scores;
    }
}
