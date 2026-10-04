using LocalScribe.Core.Audio;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Transcription;
using Whisper.net;

namespace LocalScribe.WhisperCpp;

/// <summary>
/// Whisper through whisper.cpp, for machines whose accelerator ONNX Runtime cannot reach.
/// <para>
/// On Apple silicon the Core ML build runs the encoder on the Neural Engine and keeps the
/// decoder on CPU/Metal — the same split the ONNX transcriber enforces by policy, arrived at
/// by whisper.cpp for the same reason (per-step dispatch overhead outweighs the decode
/// compute). The pipeline sees the same <see cref="ITranscriber"/> contract either way, and
/// its stamps get the same distrust: the sawtooth drift and final-window inflation are
/// Whisper properties, not properties of the Windows build.
/// </para>
/// </summary>
public sealed class WhisperCppTranscriber : ITranscriber
{
    private readonly WhisperFactory _factory;
    private readonly int _threads;
    private WhisperProcessor? _processor;
    private SpeechTask _task = SpeechTask.Transcribe;
    private string? _language;
    private readonly bool _hearsWords;
    private readonly object _heardGate = new();
    private List<WordTimings.Word> _heard = [];

    private WhisperCppTranscriber(WhisperFactory factory, string modelName, int threads, bool hearsWords)
    {
        _factory = factory;
        _threads = threads;
        _hearsWords = hearsWords;

        // The loaded native library is named so a Core ML build that quietly fell back to the
        // CPU runtime is visible in every report. Silent fallback to the wrong processor is
        // the failure this project distrusts most, whatever the runtime.
        var runtime = Whisper.net.LibraryLoader.RuntimeOptions.LoadedLibrary?.ToString() ?? "no runtime loaded";
        Description = $"Whisper {modelName} via whisper.cpp ({runtime} runtime)";
    }

    public string Description { get; }

    public string? DetectedLanguage => _language;

    /// <summary>
    /// Opens a GGML model file. The Core ML encoder is picked up automatically when its
    /// <c>-encoder.mlmodelc</c> bundle sits beside the file under its published name — which
    /// is one more reason downloaded model files are never renamed.
    /// </summary>
    /// <param name="modelPath">Path to a ggml-*.bin file, keeping its published name.</param>
    /// <param name="plan">Supplies the CPU thread budget; the cap is a product requirement.</param>
    public static WhisperCppTranscriber Load(string modelPath, ExecutionPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(
                "GGML model file not found. whisper.cpp models keep their published names, "
                + "e.g. ggml-large-v3-turbo.bin.",
                modelPath);
        }

        var name = Path.GetFileNameWithoutExtension(modelPath)
            .Replace("ggml-", string.Empty, StringComparison.Ordinal);

        // The loader's story is otherwise one generic message. Opt-in, because at Debug it
        // narrates every dlopen, which is exactly what diagnosing a silent runtime fallback
        // needs and what every other run does not.
        if (Environment.GetEnvironmentVariable("LOCALSCRIBE_WHISPER_DEBUG") == "1")
        {
            Whisper.net.Logger.LogProvider.AddConsoleLogging(Whisper.net.Logger.WhisperLogLevel.Debug);
        }

        // Asked for explicitly, because the binding's default order chose the plain CPU/Metal
        // library on a machine where the Core ML one was present — measured via the runtime
        // name in Description, not guessed from speed. Falling back to Cpu stays in the list:
        // an encoder on Metal is slower than the Neural Engine, but no transcription at all
        // is not an acceptable way to report a Core ML load failure.
        if (OperatingSystem.IsMacOS())
        {
            Whisper.net.LibraryLoader.RuntimeOptions.RuntimeLibraryOrder =
            [
                Whisper.net.LibraryLoader.RuntimeLibrary.CoreML,
                Whisper.net.LibraryLoader.RuntimeLibrary.Cpu,
            ];
        }

        // Word times from the decoder's cross-attention need the alignment heads of the exact
        // model, which whisper.cpp publishes per size; a model with no preset transcribes as
        // before and leaves its words to the aligner.
        // Measured on the podcast at Balanced: 68-70 s to transcribe against 60 s without,
        // because the timed decode ends its segments differently and the chunker takes two
        // more windows; each window costs the same. The scan finishing a few seconds later is
        // the price of every line being clickable as it lands. LOCALSCRIBE_HEARD_WORDS=0 turns
        // it off so that cost can be measured again rather than argued. The DTW times arrive
        // without WithTokenTimestamps, which was tried and changed nothing but the work.
        var heads = Environment.GetEnvironmentVariable("LOCALSCRIBE_HEARD_WORDS") == "0"
            ? WhisperAlignmentHeadsPreset.None
            : HeadsFor(name);
        var options = heads == WhisperAlignmentHeadsPreset.None
            ? new WhisperFactoryOptions()
            : new WhisperFactoryOptions { UseDtwTimeStamps = true, HeadsPreset = heads };

        return new WhisperCppTranscriber(
            WhisperFactory.FromPath(modelPath, options),
            name,
            plan.CpuBudget.IntraOpThreads,
            heads != WhisperAlignmentHeadsPreset.None);
    }

    /// <summary>The alignment-heads preset for a model by its published name.</summary>
    internal static WhisperAlignmentHeadsPreset HeadsFor(string name) => name switch
    {
        "large-v3-turbo" or "large-v3-turbo-q5_0" or "large-v3-turbo-q8_0" => WhisperAlignmentHeadsPreset.LargeV3Turbo,
        "large-v3" or "large-v3-q5_0" => WhisperAlignmentHeadsPreset.LargeV3,
        "large-v2" or "large-v2-q5_0" => WhisperAlignmentHeadsPreset.LargeV2,
        "large-v1" or "large" => WhisperAlignmentHeadsPreset.LargeV1,
        "medium" or "medium-q5_0" => WhisperAlignmentHeadsPreset.Medium,
        "medium.en" or "medium.en-q5_0" => WhisperAlignmentHeadsPreset.MediumEn,
        "small" or "small-q5_1" => WhisperAlignmentHeadsPreset.Small,
        "small.en" or "small.en-q5_1" => WhisperAlignmentHeadsPreset.SmallEn,
        "base" or "base-q5_1" => WhisperAlignmentHeadsPreset.Base,
        "base.en" or "base.en-q5_1" => WhisperAlignmentHeadsPreset.BaseEn,
        "tiny" or "tiny-q5_1" => WhisperAlignmentHeadsPreset.Tiny,
        "tiny.en" or "tiny.en-q5_1" => WhisperAlignmentHeadsPreset.TinyEn,
        _ => WhisperAlignmentHeadsPreset.None,
    };

    /// <summary>
    /// How late Whisper's cross-attention places a word, against the aligner that
    /// --check-words verified on the audio: +0.12 to +0.18 s on the two reference recordings,
    /// constant across each with no drift (doctor --aligner-trial). Taken off so a highlight
    /// driven by these times lands on the word rather than just after it.
    /// </summary>
    private const double CrossAttentionLagSeconds = 0.15;

    public IReadOnlyList<WordTimings.Word> HeardWords
    {
        get
        {
            lock (_heardGate)
            {
                return _heard;
            }
        }
    }

    public void BeginRecording(SpeechTask task = SpeechTask.Transcribe)
    {
        _task = task;

        // The language is settled per recording, not per window: whisper answers the
        // language question differently on different passes over the same audio, and a
        // language held across recordings turns transcription of the next file into
        // translation. Dropping the processor forces the next chunk to re-detect.
        _language = null;
        _processor?.Dispose();
        _processor = null;

        lock (_heardGate)
        {
            _heard = [];
        }
    }

    public async Task<IReadOnlyList<TranscriptSegment>> TranscribeChunkAsync(
        AudioChunk chunk,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(chunk);

        // A window that is almost entirely padding makes Whisper hallucinate; the ONNX
        // transcriber declines it and this one keeps the same rule.
        if (chunk.IsMostlyPadding)
        {
            return [];
        }

        // whisper.cpp pads to the encoder's 30 seconds itself, so it gets the real audio
        // only. Handing it our zero-padding as content invites text placed over silence.
        var contentSamples = Math.Min(
            chunk.Samples.Length,
            (int)(chunk.ContentSeconds * PcmAudio.WhisperSampleRate));

        var samples = chunk.Samples.AsMemory(0, contentSamples).ToArray();

        var processor = _processor ??= BuildProcessor();
        var segments = new List<TranscriptSegment>();
        var heard = new List<WordTimings.Word>();

        await foreach (var segment in processor
            .ProcessAsync(samples, cancellationToken)
            .ConfigureAwait(false))
        {
            var text = segment.Text.Trim();

            if (text.Length == 0)
            {
                continue;
            }

            if (_language is null && !string.IsNullOrEmpty(segment.Language))
            {
                _language = segment.Language;
            }

            var startInWindow = segment.Start.TotalSeconds;
            var endInWindow = segment.End.TotalSeconds;

            // Whisper's guesswork gate expects an average token log-probability, which is
            // what the ONNX decoder reports. whisper.cpp reports the mean probability
            // instead, so it is mapped through a log to land on the same scale; the floor
            // only keeps a zero from becoming -infinity.
            var confidence = Math.Log(Math.Max(segment.Probability, 1e-10));

            text = TranscriptQuality.SoundsLikeGuesswork(text, confidence)
                ? TranscriptQuality.Unintelligible
                : TranscriptQuality.TrimLoopedTail(text);

            segments.Add(new TranscriptSegment(
                text,
                chunk.StartSeconds + startInWindow,
                chunk.StartSeconds + Math.Min(endInWindow, chunk.ContentSeconds),
                confidence));

            if (_hearsWords)
            {
                HearWords(segment, chunk, heard);
            }
        }

        if (heard.Count > 0)
        {
            // Replaced rather than appended in place: readers hold the previous list, and a
            // list never changes after it is handed out.
            lock (_heardGate)
            {
                _heard = [.. _heard, .. heard];
            }
        }

        return segments;
    }

    /// <summary>
    /// Joins a segment's BPE tokens into words, each stamped where its first token was heard.
    /// </summary>
    private static void HearWords(SegmentData segment, AudioChunk chunk, List<WordTimings.Word> heard)
    {
        var building = string.Empty;
        var start = 0.0;

        void Flush()
        {
            if (building.Trim().Length > 0)
            {
                heard.Add(new WordTimings.Word(building.Trim(), start, start));
            }

            building = string.Empty;
        }

        foreach (var token in segment.Tokens)
        {
            var text = token.Text ?? string.Empty;

            // Special tokens ([_BEG_], [_TT_123], <|en|>) carry no word; a token whisper.cpp
            // could not place reports a negative time.
            if (text.StartsWith("[_", StringComparison.Ordinal)
                || text.StartsWith("<|", StringComparison.Ordinal)
                || token.DtwTimestamp < 0)
            {
                continue;
            }

            // A leading space starts a word; BPE pieces without one continue it.
            if (text.StartsWith(' ') && building.Length > 0)
            {
                Flush();
            }

            if (building.Length == 0)
            {
                var inWindow = (token.DtwTimestamp / 100.0) - CrossAttentionLagSeconds;

                // Padding past the window's real audio is not speech, whatever was heard there.
                if (inWindow > chunk.ContentSeconds)
                {
                    continue;
                }

                start = chunk.StartSeconds + Math.Max(0, inWindow);
            }

            building += text;
        }

        Flush();
    }

    private WhisperProcessor BuildProcessor()
    {
        // Probabilities are opt-in: without WithProbabilities the binding reports zero for
        // every segment, and the guesswork gate downstream reads zero as certainty of noise
        // and blanks the entire transcript. That is exactly what it did.
        var builder = _factory.CreateBuilder().WithThreads(_threads).WithProbabilities();

        builder = _language is null
            ? builder.WithLanguageDetection()
            : builder.WithLanguage(_language);

        if (_task == SpeechTask.TranslateToEnglish)
        {
            builder = builder.WithTranslate();
        }

        return builder.Build();
    }

    public void Dispose()
    {
        _processor?.Dispose();
        _factory.Dispose();
    }
}
