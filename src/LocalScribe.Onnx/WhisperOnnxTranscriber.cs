using LocalScribe.Core.Audio;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Provisioning;
using LocalScribe.Core.Transcription;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalScribe.Onnx;

/// <summary>
/// Runs Whisper through ONNX Runtime, placing each half where the plan says it should go.
/// <para>
/// Whisper is exported as two graphs because the NPU cannot run the decoder's loop. The encoder
/// takes a fixed 30-second mel spectrogram and produces hidden states; the decoder then emits
/// tokens one at a time, attending to those states. Only the encoder is worth accelerating,
/// which is why the two get separate sessions and separate execution providers.
/// </para>
/// <para>
/// <b>Export compatibility.</b> This binds to the signature used by Hugging Face Optimum and
/// Qualcomm AI Hub exports: the encoder takes one mel input, and the decoder takes
/// <c>input_ids</c> plus <c>encoder_hidden_states</c>. Input names are discovered from the model
/// metadata rather than assumed, but an export with a different <em>shape</em> contract — for
/// instance one requiring explicit key/value cache tensors — needs its own binding. A mismatch
/// throws naming every input the export actually declares, so the binding it wants is legible
/// from the error rather than needing a separate tool to dump it.
/// </para>
/// </summary>
public sealed class WhisperOnnxTranscriber : ITranscriber
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly WhisperTokenizer _tokenizer;
    private readonly LogMelSpectrogram _spectrogram;
    private readonly ExecutionPlan _plan;
    private readonly int _maxTokensPerWindow;

    /// <summary>
    /// Whisper never emits more than 448 tokens for a 30-second window, so this is a safety net
    /// against a runaway loop rather than a real limit.
    /// </summary>
    public const int DefaultMaxTokensPerWindow = 448;

    private WhisperOnnxTranscriber(
        InferenceSession encoder,
        InferenceSession decoder,
        WhisperTokenizer tokenizer,
        ExecutionPlan plan,
        int maxTokensPerWindow)
    {
        _encoder = encoder;
        _decoder = decoder;
        _tokenizer = tokenizer;
        _plan = plan;
        _spectrogram = new LogMelSpectrogram();
        _maxTokensPerWindow = maxTokensPerWindow;
    }

    /// <summary>
    /// Opens both sessions from a model directory.
    /// <para>
    /// Which file is the encoder, the decoder, and the vocabulary comes from the directory's
    /// <see cref="ModelLayout"/> manifest, falling back to conventional names. Downloaded assets
    /// keep whatever names their publisher gave them, because large ONNX models reference their
    /// weight sidecars by name from inside the graph and renaming breaks that link.
    /// </para>
    /// </summary>
    public static WhisperOnnxTranscriber Load(
        string modelDirectory,
        ExecutionPlan plan,
        int maxTokensPerWindow = DefaultMaxTokensPerWindow)
    {
        ArgumentNullException.ThrowIfNull(plan);

        var layout = ModelLayout.Discover(modelDirectory)
            ?? throw new FileNotFoundException(
                $"No usable Whisper model in {modelDirectory}. Run setup in LocalScribe to "
                + "download one, or see docs/setup-snapdragon.md to place files by hand.",
                Path.Combine(modelDirectory, ModelLayout.FileName));

        var encoder = OnnxSessionFactory.Create(layout.EncoderPath(modelDirectory), plan.Encoder, plan);

        InferenceSession decoder;
        try
        {
            decoder = OnnxSessionFactory.Create(layout.DecoderPath(modelDirectory), plan.Decoder, plan);
        }
        catch
        {
            encoder.Dispose();
            throw;
        }

        try
        {
            var tokenizer = WhisperTokenizer.LoadFromFile(layout.VocabPath(modelDirectory));
            return new WhisperOnnxTranscriber(encoder, decoder, tokenizer, plan, maxTokensPerWindow);
        }
        catch
        {
            encoder.Dispose();
            decoder.Dispose();
            throw;
        }
    }

    public string Description =>
        $"Whisper {_plan.WhisperModel}: encoder on {_plan.Encoder.Device}, decoder on {_plan.Decoder.Device}";

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
        var frames = mel.Length / LogMelSpectrogram.MelBands;

        var melTensor = new DenseTensor<float>(
            mel,
            [1, LogMelSpectrogram.MelBands, frames]);

        var encoderInput = SingleInputName(_encoder);
        using var encoderOutputs = _encoder.Run(
            [NamedOnnxValue.CreateFromTensor(encoderInput, melTensor)]);

        var hiddenStates = encoderOutputs.First().AsTensor<float>();
        var tokens = DecodeGreedily(hiddenStates, cancellationToken);

        return BuildSegments(tokens, chunk);
    }

    /// <summary>
    /// Greedy decoding: take the highest-scoring token at each step and feed the sequence back.
    /// <para>
    /// This re-runs the decoder over the whole prefix each step rather than carrying a key/value
    /// cache. That is the slower option, and deliberately so — cache tensor names and layouts
    /// differ between exports, and a decode loop that is correct everywhere is worth more than
    /// one that is fast against a single export and wrong against the rest. The decoder is also
    /// the cheaper half of Whisper, so the cost lands where it hurts least.
    /// </para>
    /// </summary>
    private List<int> DecodeGreedily(Tensor<float> encoderHiddenStates, CancellationToken cancellationToken)
    {
        var tokens = new List<int>(_tokenizer.BuildPrompt(withTimestamps: true));
        var promptLength = tokens.Count;

        var decoderTokenInput = InputNameContaining(_decoder, "input_ids", fallbackIndex: 0);
        var decoderStateInput = InputNameContaining(_decoder, "encoder_hidden_states", fallbackIndex: 1);

        for (var step = 0; step < _maxTokensPerWindow; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var idTensor = new DenseTensor<long>([1, tokens.Count]);
            for (var i = 0; i < tokens.Count; i++)
            {
                idTensor[0, i] = tokens[i];
            }

            using var outputs = _decoder.Run(
            [
                NamedOnnxValue.CreateFromTensor(decoderTokenInput, idTensor),
                NamedOnnxValue.CreateFromTensor(decoderStateInput, encoderHiddenStates),
            ]);

            var logits = outputs.First().AsTensor<float>();
            var next = ArgMaxOverLastPosition(logits);

            if (next == _tokenizer.Special.EndOfText)
            {
                break;
            }

            tokens.Add(next);
        }

        return tokens.Skip(promptLength).ToList();
    }

    /// <summary>
    /// Picks the highest-scoring token for the final position. Logits arrive shaped
    /// (batch, sequence, vocabulary) and only the last sequence position is a prediction.
    /// </summary>
    private static int ArgMaxOverLastPosition(Tensor<float> logits)
    {
        var dimensions = logits.Dimensions;
        var sequenceLength = dimensions.Length == 3 ? dimensions[1] : 1;
        var vocabularySize = dimensions[^1];
        var offset = (sequenceLength - 1) * vocabularySize;

        var best = 0;
        var bestScore = float.NegativeInfinity;

        for (var token = 0; token < vocabularySize; token++)
        {
            var score = logits.GetValue(offset + token);
            if (score > bestScore)
            {
                bestScore = score;
                best = token;
            }
        }

        return best;
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

    private static string SingleInputName(InferenceSession session) => session.InputMetadata.Keys.First();

    /// <summary>
    /// Finds an input by name, tolerating the naming differences between exports and falling
    /// back to position when nothing matches.
    /// </summary>
    private static string InputNameContaining(InferenceSession session, string fragment, int fallbackIndex)
    {
        var names = session.InputMetadata.Keys.ToList();

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
            $"The decoder has no input matching '{fragment}'. Found: {string.Join(", ", names)}. " +
            "This export uses a different signature and needs its own binding.");
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _decoder.Dispose();
    }
}
