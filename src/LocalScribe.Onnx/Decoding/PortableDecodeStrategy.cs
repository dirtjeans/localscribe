using LocalScribe.Core.Transcription;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalScribe.Onnx.Decoding;

/// <summary>
/// Greedy decoding against a stateless decoder: take the highest-scoring token each step and
/// feed the whole sequence back.
/// <para>
/// This re-runs the decoder over the entire prefix every step rather than carrying a key/value
/// cache. That is the slower option and deliberately so — it is correct against any export that
/// takes <c>input_ids</c> and <c>encoder_hidden_states</c>, whatever it calls them, and the
/// decoder is the cheaper half of Whisper, so the cost lands where it hurts least.
/// </para>
/// <para>
/// Exports that carry explicit caches get <see cref="QnnCachedDecodeStrategy"/> instead. There
/// the cache is not an optimisation we may decline: the graph will not run without it.
/// </para>
/// </summary>
internal sealed class PortableDecodeStrategy(
    InferenceSession encoder,
    InferenceSession decoder,
    WhisperTokenizer tokenizer,
    WhisperModelSignature signature,
    int melBands,
    int maxTokensPerWindow,
    DecodeSession session) : IWhisperDecodeStrategy
{
    public DecodedWindow Decode(
        float[] mel,
        int frames,
        IReadOnlyList<int>? prompt,
        CancellationToken cancellationToken)
    {
        var encoderInput = encoder.InputMetadata.Keys.First();

        using var encoderOutputs = encoder.Run(
        [
            OnnxTensors.Float(
                encoderInput,
                mel,
                [1, melBands, frames],
                OnnxTensors.ElementTypeOf(encoder, encoderInput)),
        ]);

        var hiddenStates = encoderOutputs.First();

        var tokenInput = InputNameContaining("input_ids", fallbackIndex: 0);
        var stateInput = InputNameContaining("encoder_hidden_states", fallbackIndex: 1);
        var tokenType = OnnxTensors.ElementTypeOf(decoder, tokenInput);

        // One step with nothing but the start token scores every language at once. Done once
        // per session: the language does not change between passes over the same audio, and
        // asking again is what let the answer wobble.
        session.Language ??= DetectLanguage(hiddenStates, tokenInput, stateInput, tokenType);

        var tokens = new List<int>(tokenizer.BuildPrompt(
            withTimestamps: true,
            languageToken: session.Language ?? -1,
            priorTokens: prompt));

        var promptLength = tokens.Count;
        var confidence = new List<float>();
        var limit = Math.Min(maxTokensPerWindow, signature.MaxDecodeLength);

        for (var step = 0; step < limit; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            using var outputs = decoder.Run(
            [
                TokenSequence(tokenInput, tokens, tokenType),
                OnnxTensors.Passthrough(stateInput, hiddenStates),
            ]);

            var logits = OnnxTensors.ReadFloats(outputs.First());
            var row = LastPosition(logits, tokens.Count);
            var next = OnnxTensors.ArgMax(row.Span);

            if (next == tokenizer.Special.EndOfText)
            {
                break;
            }

            tokens.Add(next);
            confidence.Add(OnnxTensors.LogProbabilityOf(row.Span, next));
        }

        return new DecodedWindow([.. tokens.Skip(promptLength)], confidence);
    }

    /// <summary>
    /// Runs a single step over just the start token and reads the language off the result.
    /// </summary>
    private int DetectLanguage(
        DisposableNamedOnnxValue hiddenStates,
        string tokenInput,
        string stateInput,
        TensorElementType tokenType)
    {
        if (tokenizer.Languages.Count == 0)
        {
            return -1;
        }

        using var outputs = decoder.Run(
        [
            TokenSequence(tokenInput, [tokenizer.Special.StartOfTranscript], tokenType),
            OnnxTensors.Passthrough(stateInput, hiddenStates),
        ]);

        var logits = OnnxTensors.ReadFloats(outputs.First());

        // Only the final position predicts what comes next; with one input token that is the
        // whole row, but an export that returns every position would otherwise be misread.
        var vocabulary = signature.VocabularySize;
        var row = logits.Length >= vocabulary
            ? logits.AsSpan(logits.Length - vocabulary, vocabulary)
            : logits.AsSpan();

        return tokenizer.DetectLanguage(row);
    }

    /// <summary>
    /// Token ids as the decoder wants them. Optimum exports normally declare int64 here, but
    /// some declare int32, so the width follows the metadata.
    /// </summary>
    private static NamedOnnxValue TokenSequence(
        string name,
        List<int> tokens,
        TensorElementType elementType)
    {
        if (elementType == TensorElementType.Int32)
        {
            return OnnxTensors.Int32(name, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(tokens), [1, tokens.Count]);
        }

        var wide = new long[tokens.Count];
        for (var i = 0; i < tokens.Count; i++)
        {
            wide[i] = tokens[i];
        }

        return NamedOnnxValue.CreateFromTensor(
            name,
            new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(wide, [1, tokens.Count]));
    }

    /// <summary>
    /// The final position's row of logits. Logits arrive as (batch, sequence, vocabulary) and
    /// only the last position is a prediction about what comes next.
    /// <para>
    /// Returned rather than reduced to an argmax, because the confidence in the chosen token has
    /// to be measured against the same row it was chosen from.
    /// </para>
    /// </summary>
    private static ReadOnlyMemory<float> LastPosition(float[] logits, int sequenceLength)
    {
        var vocabulary = logits.Length / Math.Max(1, sequenceLength);

        // A decoder that returns only the final position gives one row, not one per token.
        var offset = logits.Length - vocabulary;

        return logits.AsMemory(offset, vocabulary);
    }

    /// <summary>
    /// Finds an input by name, tolerating naming differences between exports and falling back to
    /// position when nothing matches.
    /// </summary>
    private string InputNameContaining(string fragment, int fallbackIndex)
    {
        var names = decoder.InputMetadata.Keys.ToList();

        var match = names.FirstOrDefault(name => name.Contains(fragment, StringComparison.OrdinalIgnoreCase));
        if (match is not null)
        {
            return match;
        }

        if (fallbackIndex < names.Count)
        {
            return names[fallbackIndex];
        }

        throw new InvalidOperationException(
            $"The decoder has no input matching '{fragment}'. Found: {string.Join(", ", names)}. "
            + "This export uses a different signature and needs its own binding.");
    }
}
