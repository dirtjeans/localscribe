using LocalScribe.Core.Audio;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Models;
using LocalScribe.Core.Transcription;
using LocalScribe.Onnx.Decoding;
using Microsoft.ML.OnnxRuntime;

namespace LocalScribe.Onnx;

/// <summary>
/// Runs Whisper through ONNX Runtime, placing each half where the plan says it should go.
/// <para>
/// Whisper is exported as two graphs because the NPU cannot run the decoder's loop. The encoder
/// takes a fixed 30-second mel spectrogram; the decoder then emits tokens one at a time. Only
/// the encoder is worth accelerating, which is why the two get separate sessions and separate
/// execution providers.
/// </para>
/// <para>
/// <b>Export compatibility.</b> Two contracts are supported and the right one is detected from
/// the loaded models rather than assumed. Hugging Face Optimum produces a stateless decoder
/// taking <c>input_ids</c> and <c>encoder_hidden_states</c>. Qualcomm AI Hub's precompiled QNN
/// builds produce a stateful one taking a single token plus explicit attention caches. They are
/// not interchangeable, and both load without complaint, so
/// <see cref="WhisperModelSignature"/> reads the difference off the metadata. The doctor prints
/// what it found.
/// </para>
/// </summary>
public sealed class WhisperOnnxTranscriber : ITranscriber
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly WhisperTokenizer _tokenizer;
    private readonly LogMelSpectrogram _spectrogram;
    private readonly ExecutionPlan _plan;
    private readonly IWhisperDecodeStrategy _strategy;

    /// <summary>
    /// Whisper never emits more than 448 tokens for a 30-second window, so this is a safety net
    /// against a runaway loop rather than a real limit. Cached exports carry their own, smaller,
    /// cap and ignore this.
    /// </summary>
    public const int DefaultMaxTokensPerWindow = WhisperModelSignature.PortableMaxDecodeLength;

    /// <summary>What the loaded pair of graphs turned out to be.</summary>
    public WhisperModelSignature Signature { get; }

    private WhisperOnnxTranscriber(
        InferenceSession encoder,
        InferenceSession decoder,
        WhisperTokenizer tokenizer,
        ExecutionPlan plan,
        WhisperModelSignature signature,
        int maxTokensPerWindow)
    {
        _encoder = encoder;
        _decoder = decoder;
        _tokenizer = tokenizer;
        _plan = plan;
        Signature = signature;

        // Band count comes from the encoder itself. large-v3 and its turbo derivative use 128
        // where everything before them uses 80, and feeding the wrong one produces fluent
        // nonsense rather than an error.
        _spectrogram = new LogMelSpectrogram(melBands: signature.MelBands);

        _strategy = signature.Contract == WhisperDecoderContract.QnnCached
            ? new QnnCachedDecodeStrategy(encoder, decoder, tokenizer, signature, signature.MelBands)
            : new PortableDecodeStrategy(
                encoder, decoder, tokenizer, signature, signature.MelBands, maxTokensPerWindow);
    }

    /// <summary>
    /// Opens both sessions from a model directory.
    /// <para>
    /// Two layouts are accepted. <c>encoder.onnx</c> and <c>decoder.onnx</c> side by side is the
    /// portable one. AI Hub ships <c>encoder/model.onnx</c> and <c>decoder/model.onnx</c>
    /// instead, each beside a <c>model.bin</c> holding the actual context binary — those two
    /// files have to stay together and keep their names, because the wrapper references the
    /// binary by relative path.
    /// </para>
    /// </summary>
    public static WhisperOnnxTranscriber Load(
        string modelDirectory,
        ExecutionPlan plan,
        int maxTokensPerWindow = DefaultMaxTokensPerWindow)
    {
        ArgumentNullException.ThrowIfNull(plan);

        var encoder = OnnxSessionFactory.Create(
            ResolveGraph(modelDirectory, "encoder"),
            plan.Encoder,
            plan);

        InferenceSession decoder;
        try
        {
            decoder = OnnxSessionFactory.Create(
                ResolveGraph(modelDirectory, "decoder"),
                plan.Decoder,
                plan);
        }
        catch
        {
            encoder.Dispose();
            throw;
        }

        try
        {
            var signature = DetectSignature(encoder, decoder);
            var tokenizer = WhisperTokenizer.LoadFromDirectory(modelDirectory);

            return new WhisperOnnxTranscriber(
                encoder, decoder, tokenizer, plan, signature, maxTokensPerWindow);
        }
        catch
        {
            encoder.Dispose();
            decoder.Dispose();
            throw;
        }
    }

    /// <summary>Reads both graphs' metadata and works out which contract they implement.</summary>
    public static WhisperModelSignature DetectSignature(InferenceSession encoder, InferenceSession decoder)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(decoder);

        return WhisperModelSignature.Detect(
            Describe(encoder.InputMetadata),
            Describe(encoder.OutputMetadata),
            Describe(decoder.InputMetadata),
            Describe(decoder.OutputMetadata));
    }

    private static IReadOnlyList<TensorSpec> Describe(IReadOnlyDictionary<string, NodeMetadata> metadata) =>
        metadata.Select(pair => new TensorSpec(pair.Key, pair.Value.Dimensions)).ToList();

    private static string ResolveGraph(string modelDirectory, string half) =>
        ModelLayout.GraphPath(modelDirectory, half)
        ?? throw new FileNotFoundException(
            $"No {half} graph in {modelDirectory}. Looked for '{half}.onnx' and "
            + $"'{half}/model.onnx'. Run 'localscribe-doctor --fetch-models' for a portable set, "
            + "or see docs/setup-snapdragon.md for the AI Hub layout.",
            Path.Combine(modelDirectory, $"{half}.onnx"));

    public string Description =>
        $"Whisper {_plan.WhisperModel}: encoder on {_plan.Encoder.Device}, "
        + $"decoder on {_plan.Decoder.Device} ({Signature.Describe()})";

    public Task<IReadOnlyList<TranscriptSegment>> TranscribeChunkAsync(
        AudioChunk chunk,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(chunk);

        // ONNX Runtime's API is synchronous. Running it on a worker thread keeps the UI thread
        // free, which matters more here than it would elsewhere: the whole point of the
        // accelerator policy is that the machine stays responsive during transcription.
        return Task.Run(() => TranscribeChunk(chunk, cancellationToken), cancellationToken);
    }

    private IReadOnlyList<TranscriptSegment> TranscribeChunk(AudioChunk chunk, CancellationToken cancellationToken)
    {
        var mel = _spectrogram.Compute(chunk.Samples);
        var frames = mel.Length / _spectrogram.MelBands;

        var tokens = _strategy.Decode(mel, frames, cancellationToken);

        return BuildSegments(tokens, chunk);
    }

    /// <summary>
    /// Splits the decoded tokens into timed segments. Whisper emits timestamp tokens in pairs
    /// that bracket each utterance; text between a pair belongs to that span.
    /// </summary>
    private List<TranscriptSegment> BuildSegments(List<int> tokens, AudioChunk chunk)
    {
        var segments = new List<TranscriptSegment>();
        var current = new List<int>();
        double? segmentStart = null;

        foreach (var token in tokens)
        {
            if (!_tokenizer.Special.IsTimestamp(token))
            {
                current.Add(token);
                continue;
            }

            var time = _tokenizer.TimestampToSeconds(token);

            if (segmentStart is null)
            {
                segmentStart = time;
                continue;
            }

            AppendSegment(segments, current, segmentStart.Value, time, chunk);
            current.Clear();
            segmentStart = null;
        }

        // A window that ends mid-utterance leaves text with no closing timestamp.
        if (current.Count > 0)
        {
            AppendSegment(
                segments,
                current,
                segmentStart ?? 0,
                Math.Min(chunk.ContentSeconds, AudioChunker.WindowSeconds),
                chunk);
        }

        return segments;
    }

    private void AppendSegment(
        List<TranscriptSegment> segments,
        List<int> tokens,
        double startInWindow,
        double endInWindow,
        AudioChunk chunk)
    {
        var text = _tokenizer.Decode(tokens).Trim();
        if (text.Length == 0)
        {
            return;
        }

        // Text the model placed beyond the real audio is padding-induced invention.
        if (startInWindow > chunk.ContentSeconds)
        {
            return;
        }

        segments.Add(new TranscriptSegment(
            text,
            chunk.StartSeconds + startInWindow,
            chunk.StartSeconds + Math.Min(endInWindow, chunk.ContentSeconds)));
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _decoder.Dispose();
    }
}
