using LocalScribe.Core.Transcription;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalScribe.Onnx.Decoding;

/// <summary>
/// Greedy decoding against a Qualcomm AI Hub export, which hands the decode loop's state back
/// and forth explicitly.
/// <para>
/// The encoder here does not produce hidden states. It produces the cross-attention key/value
/// caches directly, one pair per decoder layer, and those stay fixed for the whole window. The
/// decoder then takes exactly one token per call along with its own self-attention caches, and
/// returns updated ones. The graph shifts the caches internally, so each step's outputs become
/// the next step's inputs untouched.
/// </para>
/// <para>
/// Two details are not guessable and are taken from Qualcomm's reference implementation. The
/// attention mask starts fully masked and is opened up one position at a time from the right
/// hand end, so at step <c>n</c> position <c>window - n - 1</c> becomes visible. And the masked
/// value is -100, not negative infinity: these graphs run in float16, where infinities do not
/// survive the arithmetic.
/// </para>
/// <para>
/// The window is fixed at export time — 200 for the published builds — and is a hard cap rather
/// than a safety net, because the caches are sized for it.
/// </para>
/// </summary>
internal sealed class QnnCachedDecodeStrategy(
    InferenceSession encoder,
    InferenceSession decoder,
    WhisperTokenizer tokenizer,
    WhisperModelSignature signature,
    int melBands) : IWhisperDecodeStrategy
{
    /// <summary>
    /// What a masked position scores. Qualcomm's export uses a finite sentinel rather than
    /// negative infinity because the graph is float16 throughout.
    /// </summary>
    private const float MaskedScore = -100.0f;

    public List<int> Decode(float[] mel, int frames, CancellationToken cancellationToken)
    {
        var encoderInput = encoder.InputMetadata.Keys.First();

        // The cross caches are the encoder's whole output and are reused unchanged every step,
        // so this stays alive for the entire window rather than being recomputed.
        using var crossCaches = encoder.Run(
        [
            OnnxTensors.Float(
                encoderInput,
                mel,
                [1, melBands, frames],
                OnnxTensors.ElementTypeOf(encoder, encoderInput)),
        ]);

        return DecodeWithCaches(crossCaches, cancellationToken);
    }

    private List<int> DecodeWithCaches(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> crossCaches,
        CancellationToken cancellationToken)
    {
        var layers = signature.DecoderLayers;
        var window = signature.MaxDecodeLength;

        var prompt = tokenizer.BuildPrompt(withTimestamps: true);
        var tokens = new List<int>(prompt);
        var emitted = new List<int>();

        var mask = new float[window];
        Array.Fill(mask, MaskedScore);

        var selfCaches = InitialSelfCaches(layers);
        var maskType = OnnxTensors.ElementTypeOf(decoder, "attention_mask");

        try
        {
            // One fewer than the window: the last slot belongs to the token being decoded.
            for (var step = 0; step < window - 1; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Open the mask one position further, counting back from the right hand end.
                mask[window - step - 1] = 0.0f;

                var inputs = new List<NamedOnnxValue>(2 + (4 * layers) + 1)
                {
                    OnnxTensors.Int32("input_ids", [tokens[step]], [1, 1]),
                    OnnxTensors.Float("attention_mask", mask, [1, 1, 1, window], maskType),
                };

                for (var layer = 0; layer < layers; layer++)
                {
                    inputs.Add(selfCaches[layer].Key);
                    inputs.Add(selfCaches[layer].Value);
                }

                foreach (var cross in crossCaches)
                {
                    inputs.Add(OnnxTensors.Passthrough(cross.Name, cross));
                }

                inputs.Add(OnnxTensors.Int32("position_ids", [step], [1]));

                var outputs = decoder.Run(inputs);

                try
                {
                    var next = OnnxTensors.ArgMax(OnnxTensors.ReadFloats(Output(outputs, "logits")));

                    if (next == tokenizer.Special.EndOfText)
                    {
                        break;
                    }

                    // While the forced prompt is still being fed the prediction is discarded:
                    // those tokens are ours to choose, not the model's.
                    if (step >= prompt.Count - 1)
                    {
                        tokens.Add(next);
                        emitted.Add(next);
                    }

                    selfCaches = CarryForward(outputs, layers, selfCaches);
                }
                finally
                {
                    outputs.Dispose();
                }
            }
        }
        finally
        {
            foreach (var cache in selfCaches)
            {
                cache.Dispose();
            }
        }

        return emitted;
    }

    /// <summary>
    /// Zeroed caches for the first step. Nothing has been attended to yet, and the mask keeps
    /// these positions invisible until they hold something real.
    /// </summary>
    private SelfCache[] InitialSelfCaches(int layers)
    {
        var window = signature.MaxDecodeLength;
        var caches = new SelfCache[layers];

        for (var layer = 0; layer < layers; layer++)
        {
            var keyName = $"k_cache_self_{layer}_in";
            var valueName = $"v_cache_self_{layer}_in";

            caches[layer] = new SelfCache(
                OnnxTensors.Float(
                    keyName,
                    new float[Elements(keyName)],
                    ShapeOf(keyName),
                    OnnxTensors.ElementTypeOf(decoder, keyName)),
                OnnxTensors.Float(
                    valueName,
                    new float[Elements(valueName)],
                    ShapeOf(valueName),
                    OnnxTensors.ElementTypeOf(decoder, valueName)),
                Owned: null);
        }

        return caches;

        int Elements(string name)
        {
            var shape = ShapeOf(name);
            var total = 1;
            foreach (var dimension in shape)
            {
                total *= dimension;
            }

            return total;
        }
    }

    /// <summary>
    /// The decoder's updated caches become the next step's inputs. They are handed straight
    /// across without conversion: the graph has already done the shifting, and round-tripping
    /// float16 through float32 here would cost time and precision for nothing.
    /// </summary>
    private static SelfCache[] CarryForward(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        int layers,
        SelfCache[] previous)
    {
        var next = new SelfCache[layers];

        for (var layer = 0; layer < layers; layer++)
        {
            var key = Output(outputs, $"k_cache_self_{layer}_out");
            var value = Output(outputs, $"v_cache_self_{layer}_out");

            // The outputs collection is disposed at the end of this step, so the tensors are
            // copied out rather than referenced.
            next[layer] = new SelfCache(
                Copy($"k_cache_self_{layer}_in", key),
                Copy($"v_cache_self_{layer}_in", value),
                Owned: null);
        }

        foreach (var cache in previous)
        {
            cache.Dispose();
        }

        return next;

        static NamedOnnxValue Copy(string name, DisposableNamedOnnxValue source) =>
            source.ElementType == TensorElementType.Float16
                ? NamedOnnxValue.CreateFromTensor(
                    name,
                    new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<Float16>(
                        source.AsTensor<Float16>().ToArray(),
                        source.AsTensor<Float16>().Dimensions.ToArray()))
                : NamedOnnxValue.CreateFromTensor(
                    name,
                    new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(
                        source.AsTensor<float>().ToArray(),
                        source.AsTensor<float>().Dimensions.ToArray()));
    }

    private int[] ShapeOf(string inputName) =>
        decoder.InputMetadata.TryGetValue(inputName, out var metadata)
            ? metadata.Dimensions
            : throw new InvalidOperationException(
                $"The decoder declares no input '{inputName}'. Found: "
                + $"{string.Join(", ", decoder.InputMetadata.Keys)}.");

    private static DisposableNamedOnnxValue Output(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        string name) =>
        outputs.FirstOrDefault(o => o.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
        ?? throw new InvalidOperationException(
            $"The decoder produced no output '{name}'. Found: "
            + $"{string.Join(", ", outputs.Select(o => o.Name))}.");

    /// <summary>One layer's self-attention cache, as inputs ready for the next step.</summary>
    private readonly record struct SelfCache(
        NamedOnnxValue Key,
        NamedOnnxValue Value,
        IDisposable? Owned)
    {
        public void Dispose() => Owned?.Dispose();
    }
}
