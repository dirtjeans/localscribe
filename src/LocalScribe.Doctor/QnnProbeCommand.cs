using System.Diagnostics;
using LocalScribe.Onnx;
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

        var provider = new Dictionary<string, string>();
        if (!gpu)
        {
            provider["htp_performance_mode"] = "burst";
            provider["enable_htp_fp16_precision"] = "1";
        }

        // Any provider option, "key=value;key=value", so profiling or another performance mode
        // can be tried without a rebuild.
        foreach (var pair in (Environment.GetEnvironmentVariable("LOCALSCRIBE_QNN_OPTIONS") ?? string.Empty)
                     .Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            var cut = pair.IndexOf('=');
            if (cut > 0)
            {
                provider[pair[..cut].Trim()] = pair[(cut + 1)..].Trim();
            }
        }

        if (Environment.GetEnvironmentVariable("LOCALSCRIBE_QNN_BACKEND") is not "cpu")
        {
            WindowsMl.AppendQnn(options, provider, gpu ? OrtHardwareDeviceType.GPU : OrtHardwareDeviceType.NPU);
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

        return meta.ElementDataType switch
        {
            TensorElementType.Float16 => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<Float16>(new Float16[count], shape)),
            TensorElementType.Int32 => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<int>(new int[count], shape)),
            TensorElementType.Int64 => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(new long[count], shape)),
            TensorElementType.UInt8 => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<byte>(new byte[count], shape)),
            TensorElementType.UInt16 => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<ushort>(new ushort[count], shape)),
            _ => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(new float[count], shape)),
        };
    }
}
