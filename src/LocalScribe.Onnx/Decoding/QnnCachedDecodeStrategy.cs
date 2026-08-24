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
    int melBands,
    DecodeSession session) : IWhisperDecodeStrategy
{
    /// <summary>
    /// What a masked position scores. Qualcomm's export uses a finite sentinel rather than
    /// negative infinity because the graph is float16 throughout.
    /// </summary>
    private const float MaskedScore = -100.0f;

    public DecodedWindow Decode(
        float[] mel,
        int frames,
        IReadOnlyList<int>? prompt,
        CancellationToken cancellationToken)
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

        return DecodeWithCaches(crossCaches, prompt, cancellationToken);
    }

    private DecodedWindow DecodeWithCaches(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> crossCaches,
        IReadOnlyList<int>? prompt,
        CancellationToken cancellationToken)
    {
        var layers = signature.DecoderLayers;
        var window = signature.MaxDecodeLength;

        // With a prompt the whole opening is known in advance and fed as one: a prompt is only
        // ever used on a retry, by which point the language has already been established, so
        // there is nothing left to detect.
        var forced = prompt is { Count: > 0 }
            ? new List<int>(tokenizer.BuildPrompt(
                withTimestamps: true,
                languageToken: session.Language ?? -1,
                priorTokens: prompt,
                task: session.Task))
            : [tokenizer.Special.StartOfTranscript];

        var detecting = prompt is not { Count: > 0 };
        var tokens = new List<int>(forced);
        var emitted = new List<int>();
        var confidence = new List<float>();

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
                    var logits = OnnxTensors.ReadFloats(Output(outputs, "logits"));

                    // The token after the start marker is where Whisper says what language it
                    // heard, so this step is a detection rather than a prediction.
                    if (step == 0 && detecting)
                    {
                        forced.AddRange(PromptAfterDetection(logits));
                    }

                    // While the prompt is still being fed the prediction is discarded: those
                    // tokens are ours to choose, not the model's.
                    var next = step + 1 < forced.Count
                        ? forced[step + 1]
                        : OnnxTensors.ArgMax(logits);

                    if (next == tokenizer.Special.EndOfText)
                    {
                        break;
                    }

                    tokens.Add(next);

                    if (step + 1 >= forced.Count)
                    {
                        emitted.Add(next);

                        // Only for tokens the model chose. A forced prompt token says nothing
                        // about how well it could hear.
                        confidence.Add(OnnxTensors.LogProbabilityOf(logits, next));
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

        return new DecodedWindow(emitted, confidence);
    }

    /// <summary>
    /// The rest of the prompt, once the language has been read off the first step's scores.
    /// <para>
    /// Detected once and then reused for the whole session. Re-detecting every pass is what
    /// produced a transcript that opened with "Gracias." over English speech and then changed
    /// its mind; the language of a recording does not change between passes over the same audio,
    /// so neither should the prompt.
    /// </para>
    /// </summary>
    private IReadOnlyList<int> PromptAfterDetection(ReadOnlySpan<float> logits)
    {
        session.Language ??= tokenizer.DetectLanguage(logits);

        var prompt = new List<int>();

        if (session.Language is > 0)
        {
            prompt.Add(session.Language.Value);
        }

        prompt.Add(tokenizer.TaskToken(session.Task));

        return prompt;
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
