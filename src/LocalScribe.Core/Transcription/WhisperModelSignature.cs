using System.Text.RegularExpressions;

namespace LocalScribe.Core.Transcription;

/// <summary>One model input or output, as the runtime reports it.</summary>
/// <param name="Name">The tensor's name.</param>
/// <param name="Shape">Its dimensions. Negative or zero entries mean dynamic.</param>
public sealed record TensorSpec(string Name, IReadOnlyList<int> Shape);

/// <summary>Which decoding contract an export implements.</summary>
public enum WhisperDecoderContract
{
    /// <summary>
    /// Decoder takes <c>input_ids</c> and <c>encoder_hidden_states</c> and is re-run over the
    /// whole prefix each step. What Hugging Face Optimum emits.
    /// </summary>
    Portable,

    /// <summary>
    /// Decoder takes one token at a time alongside explicit self- and cross-attention caches.
    /// What Qualcomm AI Hub emits for the precompiled QNN target.
    /// </summary>
    QnnCached,
}

/// <summary>
/// What a loaded pair of Whisper graphs actually expects.
/// <para>
/// The README predicted this would be the first thing to break on real hardware, and it was
/// right: exports do not merely rename tensors, they disagree about the shape of the decode
/// loop itself. Optimum gives a stateless decoder that re-reads its whole prefix. AI Hub gives
/// a stateful one that takes a single token plus every attention cache explicitly, and caps the
/// window at a fixed length.
/// </para>
/// <para>
/// Guessing between them is not an option, because both load without complaint and only the
/// output tells you which you got. So the contract is read off the metadata, and everything
/// that varies — band count, layer count, window cap, vocabulary — is read with it rather than
/// hard-coded.
/// </para>
/// <para>
/// Pure, and takes plain tensor descriptions rather than live sessions, so the whole detection
/// rule is testable on a machine with no ONNX Runtime and no NPU.
/// </para>
/// </summary>
public sealed record WhisperModelSignature(
    WhisperDecoderContract Contract,
    int MelBands,
    int DecoderLayers,
    int MaxDecodeLength,
    int VocabularySize)
{
    /// <summary>Self-attention cache inputs, e.g. <c>k_cache_self_0_in</c>.</summary>
    private static readonly Regex SelfCacheInput = new(
        @"^k_cache_self_(\d+)_in$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    /// <summary>Cross-attention caches, which the encoder produces and the decoder consumes.</summary>
    private static readonly Regex CrossCache = new(
        @"^k_cache_cross_(\d+)$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    /// <summary>
    /// Whisper never emits more than this for a 30-second window, so it is a runaway-loop guard
    /// rather than a real limit. Only the portable contract needs it; the cached one carries its
    /// own cap in the attention mask.
    /// </summary>
    public const int PortableMaxDecodeLength = 448;

    /// <summary>Works out the contract from what the two graphs declare.</summary>
    public static WhisperModelSignature Detect(
        IReadOnlyList<TensorSpec> encoderInputs,
        IReadOnlyList<TensorSpec> encoderOutputs,
        IReadOnlyList<TensorSpec> decoderInputs,
        IReadOnlyList<TensorSpec> decoderOutputs)
    {
        ArgumentNullException.ThrowIfNull(encoderInputs);
        ArgumentNullException.ThrowIfNull(encoderOutputs);
        ArgumentNullException.ThrowIfNull(decoderInputs);
        ArgumentNullException.ThrowIfNull(decoderOutputs);

        if (encoderInputs.Count == 0)
        {
            throw new InvalidOperationException("The encoder declares no inputs.");
        }

        var melBands = MelBandsFrom(encoderInputs[0]);

        var cachedLayers = decoderInputs.Count(spec => SelfCacheInput.IsMatch(spec.Name));

        if (cachedLayers == 0)
        {
            return new WhisperModelSignature(
                WhisperDecoderContract.Portable,
                melBands,
                DecoderLayers: 0,
                PortableMaxDecodeLength,
                VocabularySize: LastDimension(Find(decoderOutputs, "logits") ?? decoderOutputs[0]));
        }

        var crossLayers = encoderOutputs.Count(spec => CrossCache.IsMatch(spec.Name));

        if (crossLayers != cachedLayers)
        {
            // Mismatched halves are usually two exports from different runs, and would fail
            // deep inside the first decode step with nothing to point at.
            throw new InvalidOperationException(
                $"The encoder produces {crossLayers} cross-attention cache pairs but the decoder "
                + $"expects {cachedLayers}. These are exports of different models.");
        }

        var mask = Find(decoderInputs, "attention_mask")
            ?? throw new InvalidOperationException(
                "A cached decoder must declare attention_mask, which carries the window length.");

        return new WhisperModelSignature(
            WhisperDecoderContract.QnnCached,
            melBands,
            cachedLayers,
            MaxDecodeLength: LastDimension(mask),
            VocabularySize: VocabularyFromCachedLogits(
                Find(decoderOutputs, "logits")
                ?? throw new InvalidOperationException("A cached decoder must declare logits.")));
    }

    /// <summary>
    /// Band count comes from the encoder's own input, because Whisper is not consistent about
    /// it: 80 through large-v2, 128 for large-v3 and turbo.
    /// </summary>
    private static int MelBandsFrom(TensorSpec encoderInput)
    {
        // (batch, bands, frames). A dynamic batch is normal; a dynamic band count is not, since
        // the filterbank has to be built before the model is ever called.
        if (encoderInput.Shape.Count < 2 || encoderInput.Shape[1] <= 0)
        {
            throw new InvalidOperationException(
                $"Cannot read the mel band count from the encoder input '{encoderInput.Name}' "
                + $"with shape [{string.Join(", ", encoderInput.Shape)}].");
        }

        return encoderInput.Shape[1];
    }

    /// <summary>
    /// Cached exports emit logits as (1, vocabulary, 1, 1) rather than (batch, sequence,
    /// vocabulary), so the vocabulary is the largest dimension rather than the last one.
    /// </summary>
    private static int VocabularyFromCachedLogits(TensorSpec logits)
    {
        var largest = logits.Shape.Count == 0 ? 0 : logits.Shape.Max();

        return largest > 1
            ? largest
            : throw new InvalidOperationException(
                $"Cannot read the vocabulary size from logits shaped "
                + $"[{string.Join(", ", logits.Shape)}].");
    }

    private static int LastDimension(TensorSpec spec) =>
        spec.Shape.Count > 0 && spec.Shape[^1] > 0
            ? spec.Shape[^1]
            : throw new InvalidOperationException(
                $"'{spec.Name}' has no usable final dimension: "
                + $"[{string.Join(", ", spec.Shape)}].");

    private static TensorSpec? Find(IReadOnlyList<TensorSpec> specs, string name) =>
        specs.FirstOrDefault(s => s.Name.Equals(name, StringComparison.OrdinalIgnoreCase));

    /// <summary>A one-line description for the doctor, so a mismatch is visible immediately.</summary>
    public string Describe() => Contract == WhisperDecoderContract.Portable
        ? $"portable export, {MelBands} mel bands, vocabulary {VocabularySize}"
        : $"cached QNN export, {MelBands} mel bands, {DecoderLayers} decoder layers, "
          + $"window {MaxDecodeLength}, vocabulary {VocabularySize}";
}
