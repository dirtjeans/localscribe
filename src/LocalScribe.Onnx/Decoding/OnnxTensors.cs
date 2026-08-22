using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalScribe.Onnx.Decoding;

/// <summary>
/// Builds and reads ONNX tensors without the caller having to care whether a model wants 32-bit
/// or 16-bit floats.
/// <para>
/// This exists because exports disagree. Optimum emits float32; the Qualcomm AI Hub precompiled
/// builds are float16 throughout, which is also what other shipping QNN speech models use. The
/// element type is read from the session's own metadata rather than assumed, because feeding
/// float32 to a float16 input does not fail cleanly — ONNX Runtime reports a type mismatch for
/// some providers and reinterprets the bytes for others.
/// </para>
/// </summary>
internal static class OnnxTensors
{
    /// <summary>The element type a named input expects.</summary>
    public static TensorElementType ElementTypeOf(InferenceSession session, string name) =>
        session.InputMetadata.TryGetValue(name, out var metadata)
            ? metadata.ElementDataType
            : TensorElementType.Float;

    /// <summary>
    /// Wraps float data as a named value in whichever float width the model wants.
    /// </summary>
    public static NamedOnnxValue Float(
        string name,
        ReadOnlySpan<float> data,
        ReadOnlySpan<int> shape,
        TensorElementType elementType)
    {
        var dimensions = shape.ToArray();

        if (elementType == TensorElementType.Float16)
        {
            var half = new Float16[data.Length];
            for (var i = 0; i < data.Length; i++)
            {
                half[i] = (Float16)data[i];
            }

            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<Float16>(half, dimensions));
        }

        return NamedOnnxValue.CreateFromTensor(
            name,
            new DenseTensor<float>(data.ToArray(), dimensions));
    }

    /// <summary>Wraps 32-bit integer data, the width every Whisper export uses for token ids.</summary>
    public static NamedOnnxValue Int32(string name, ReadOnlySpan<int> data, ReadOnlySpan<int> shape) =>
        NamedOnnxValue.CreateFromTensor(name, new DenseTensor<int>(data.ToArray(), shape.ToArray()));

    /// <summary>
    /// Reads a float output back as float32 whatever width it arrived in. Returns the tensor's
    /// values in flat order.
    /// </summary>
    public static float[] ReadFloats(DisposableNamedOnnxValue value)
    {
        ArgumentNullException.ThrowIfNull(value);

        if (value.ElementType == TensorElementType.Float16)
        {
            var source = value.AsTensor<Float16>();
            var result = new float[source.Length];
            var index = 0;
            foreach (var element in source)
            {
                result[index++] = (float)element;
            }

            return result;
        }

        var floats = value.AsTensor<float>();
        var output = new float[floats.Length];
        var position = 0;
        foreach (var element in floats)
        {
            output[position++] = element;
        }

        return output;
    }

    /// <summary>
    /// Re-wraps an output as an input for the next step, preserving its element type. Used to
    /// feed a decoder's updated self-attention cache straight back in.
    /// </summary>
    public static NamedOnnxValue Passthrough(string name, DisposableNamedOnnxValue value)
    {
        ArgumentNullException.ThrowIfNull(value);

        return value.ElementType == TensorElementType.Float16
            ? NamedOnnxValue.CreateFromTensor(name, value.AsTensor<Float16>())
            : NamedOnnxValue.CreateFromTensor(name, value.AsTensor<float>());
    }

    /// <summary>The index of the largest value in a flat span.</summary>
    public static int ArgMax(ReadOnlySpan<float> values)
    {
        var best = 0;
        var bestScore = float.NegativeInfinity;

        for (var i = 0; i < values.Length; i++)
        {
            if (values[i] > bestScore)
            {
                bestScore = values[i];
                best = i;
            }
        }

        return best;
    }

    /// <summary>
    /// How confident the model was in the token it chose, as a log probability.
    /// <para>
    /// Zero means certain, and it falls away steeply from there: around -0.2 for ordinary
    /// speech, past -1 when the model is guessing at something it cannot hear. That is the
    /// number that separates a transcript from an invention, and Whisper will invent fluent
    /// sentences out of noise without any change in tone to warn you.
    /// </para>
    /// <para>
    /// Computed in log space against the maximum, which is the standard way to stop the
    /// exponentials overflowing on a vocabulary of fifty thousand.
    /// </para>
    /// </summary>
    public static float LogProbabilityOf(ReadOnlySpan<float> logits, int token)
    {
        if (token < 0 || token >= logits.Length)
        {
            return 0;
        }

        var largest = float.NegativeInfinity;
        foreach (var value in logits)
        {
            if (value > largest)
            {
                largest = value;
            }
        }

        var total = 0.0;
        foreach (var value in logits)
        {
            total += Math.Exp(value - largest);
        }

        return (float)(logits[token] - largest - Math.Log(total));
    }
}
