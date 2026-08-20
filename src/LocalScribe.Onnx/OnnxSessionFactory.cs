using LocalScribe.Core.Hardware;
using Microsoft.ML.OnnxRuntime;

namespace LocalScribe.Onnx;

/// <summary>
/// Builds ONNX Runtime sessions that honour an <see cref="ExecutionPlan"/>.
/// <para>
/// This is where the plan stops being advice and starts being configuration. Two details here
/// matter more than the rest: registering QNN correctly, and refusing to let a failed
/// registration pass unnoticed. Silent fallback to the CPU is the single most common way a
/// Snapdragon app ends up slower than the developer thinks, because everything still works —
/// it just quietly uses the wrong processor.
/// </para>
/// </summary>
public static class OnnxSessionFactory
{
    /// <summary>
    /// Creates a session for one stage.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX file. For QNN this is a precompiled context binary.</param>
    /// <param name="stage">The plan entry describing where this stage should run.</param>
    /// <param name="plan">The overall plan, for the CPU budget and strictness setting.</param>
    /// <exception cref="OnnxRuntimeException">
    /// Thrown when <see cref="ExecutionPlan.StrictProviderCheck"/> is set and the requested
    /// provider cannot take the whole graph.
    /// </exception>
    public static InferenceSession Create(string modelPath, StagePlan stage, ExecutionPlan plan)
    {
        ArgumentNullException.ThrowIfNull(stage);
        ArgumentNullException.ThrowIfNull(plan);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(
                $"Model file not found. Run the doctor tool with --fetch-models to download a set " +
                $"matching this machine.",
                modelPath);
        }

        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = plan.CpuBudget.IntraOpThreads,
            InterOpNumThreads = plan.CpuBudget.InterOpThreads,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
        };

        try
        {
            switch (stage.Device)
            {
                case ComputeDevice.Npu:
                    ConfigureQnn(options, plan);
                    break;

                case ComputeDevice.Gpu:
                    options.AppendExecutionProvider_DML(deviceId: 0);
                    break;

                case ComputeDevice.Cpu:
                default:
                    // The built-in CPU provider is always registered; there is nothing to add.
                    break;
            }

            return new InferenceSession(modelPath, options);
        }
        catch
        {
            options.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Registers the Qualcomm provider and points it at the Hexagon backend.
    /// </summary>
    private static void ConfigureQnn(SessionOptions options, ExecutionPlan plan)
    {
        var providerOptions = new Dictionary<string, string>
        {
            // HTP is the Hexagon Tensor Processor: the NPU proper. The alternatives target the
            // CPU or GPU through the same provider, which would defeat the purpose.
            ["backend_path"] = "QnnHtp.dll",

            // Burst gives the shortest time to first token. It also draws more power, which is
            // the right trade for a foreground transcription the user is waiting on.
            ["htp_performance_mode"] = "burst",

            // Precompiled context binaries skip on-device graph compilation. Without this the
            // first run of every session pays several seconds of compile time.
            ["enable_htp_fp16_precision"] = "1",
        };

        options.AppendExecutionProvider("QNN", providerOptions);

        if (plan.StrictProviderCheck)
        {
            // Makes ONNX Runtime throw rather than silently place unsupported nodes on the CPU.
            // The doctor tool switches this on precisely so it can prove the NPU is real; the
            // app leaves it off so that a driver problem degrades instead of crashing.
            options.AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1");
        }
    }

    /// <summary>
    /// Reports which providers this build of ONNX Runtime has available. Note that a provider
    /// listed here has loaded its library — it has not been proven to run a given model.
    /// </summary>
    public static IReadOnlySet<string> AvailableProviders()
    {
        try
        {
            return OrtEnv.Instance().GetAvailableProviders().ToHashSet(StringComparer.OrdinalIgnoreCase);
        }
        catch (Exception exception) when (exception is OnnxRuntimeException or DllNotFoundException)
        {
            return new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        }
    }
}
