using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalScribe.Doctor;

/// <summary>
/// Whether a graph runs on one of QNN's backends at all, and how fast: opened at fixed shapes,
/// fed zeros, timed. For deciding whether a model, or a slice of one, is worth moving before
/// writing the code that would move it. The backend is LOCALSCRIBE_QNN_BACKEND (htp or gpu);
/// LOCALSCRIBE_ALLOW_FALLBACK lets refused nodes run on the CPU, and the verbose log then says
/// how many nodes each provider took.
/// </summary>
internal static class QnnProbeCommand
{
    public static int Run(string modelPath, IReadOnlyDictionary<string, long> dims, int runs)
    {
        var gpu = Environment.GetEnvironmentVariable("LOCALSCRIBE_QNN_BACKEND") is "gpu";
        var options = new SessionOptions();
        if (Environment.GetEnvironmentVariable("LOCALSCRIBE_QNN_VERBOSE") is { Length: > 0 })
        {
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
        }

        foreach (var (name, value) in dims)
        {
            options.AddFreeDimensionOverrideByName(name, value);
        }

        if (Environment.GetEnvironmentVariable("LOCALSCRIBE_ALLOW_FALLBACK") is not { Length: > 0 })
        {
            options.AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1");
        }

        var provider = new Dictionary<string, string> { ["backend_path"] = gpu ? "QnnGpu.dll" : "QnnHtp.dll" };
        if (!gpu)
        {
            provider["htp_performance_mode"] = "burst";
            provider["enable_htp_fp16_precision"] = "1";
        }

        if (Environment.GetEnvironmentVariable("LOCALSCRIBE_QNN_BACKEND") is not "cpu")
        {
            options.AppendExecutionProvider("QNN", provider);
        }

        Console.WriteLine($"Probe {Path.GetFileName(modelPath)} on {Environment.GetEnvironmentVariable("LOCALSCRIBE_QNN_BACKEND") ?? "htp"}");
        var opening = Stopwatch.StartNew();
        InferenceSession session;
        try
        {
            session = new InferenceSession(modelPath, options);
        }
        catch (OnnxRuntimeException ex)
        {
            Console.WriteLine($"  refused after {opening.Elapsed.TotalSeconds:F1} s: {ex.Message}");
            return 1;
        }

        Console.WriteLine($"  opened in {opening.Elapsed.TotalSeconds:F1} s");
        using (session)
        {
            var inputs = session.InputMetadata.Select(pair => Zeros(pair.Key, pair.Value, dims)).ToList();
            var times = new List<double>();
            for (var i = 0; i <= runs; i++)
            {
                var run = Stopwatch.StartNew();
                using var outputs = session.Run(inputs);
                if (i > 0)
                {
                    // The first run warms the backend up; it is not what a scan pays per window.
                    times.Add(run.Elapsed.TotalMilliseconds);
                }
                else
                {
                    Console.WriteLine($"  first run {run.Elapsed.TotalMilliseconds:F0} ms");
                }
            }

            times.Sort();
            Console.WriteLine($"  {runs} runs: median {times[times.Count / 2]:F0} ms, fastest {times[0]:F0} ms");
        }

        return 0;
    }

    private static NamedOnnxValue Zeros(string name, NodeMetadata meta, IReadOnlyDictionary<string, long> dims)
    {
        var shape = meta.Dimensions
            .Select((d, i) => d > 0 ? d : (int)dims.GetValueOrDefault(meta.SymbolicDimensions[i], 1))
            .ToArray();
        var count = shape.Aggregate(1, (a, b) => a * b);

        return meta.ElementDataType == TensorElementType.Float16
            ? NamedOnnxValue.CreateFromTensor(name, new DenseTensor<Float16>(new Float16[count], shape))
            : NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(new float[count], shape));
    }
}
