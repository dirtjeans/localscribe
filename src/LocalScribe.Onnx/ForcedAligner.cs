using LocalScribe.Core.Alignment;
using LocalScribe.Core.Audio;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Transcription;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalScribe.Onnx;

/// <summary>
/// Times the words of a segment by aligning them against the audio.
/// <para>
/// Whisper times whole segments and its exported graphs emit no cross-attention, so the usual
/// way of recovering word times from Whisper is unavailable and they have been estimated from
/// loudness — good to about half a second. This measures them instead, to about the length of
/// one twenty-millisecond frame.
/// </para>
/// <para>
/// The model is a multilingual CTC recogniser, used here for the far easier job of alignment:
/// the words are already known, so all that is wanted is which frames go with which letters.
/// One model covers a thousand languages because it works in a romanised alphabet, which is why
/// <see cref="AlignmentAlphabet"/> folds every word down to plain Latin letters first.
/// </para>
/// <para>
/// Optional, like the speaker models. A machine without it transcribes exactly as before and
/// falls back to estimating.
/// </para>
/// </summary>
public sealed class ForcedAligner : IDisposable
{
    private readonly InferenceSession _session;
    private readonly AlignmentAlphabet _alphabet;
    private readonly string _input;

    private ForcedAligner(InferenceSession session, AlignmentAlphabet alphabet)
    {
        _session = session;
        _alphabet = alphabet;
        _input = session.InputMetadata.Keys.First();
    }

    /// <summary>What the aligner needs on disk, or null when it is not installed.</summary>
    public static string? Find(string modelRoot)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelRoot);

        var directory = Path.Combine(modelRoot, "alignment");

        return File.Exists(Path.Combine(directory, "vocab.json")) && ModelIn(directory) is not null
            ? directory
            : null;
    }

    /// <summary>Loads the aligner from a directory, or throws saying what is missing.</summary>
    public static ForcedAligner Load(string directory, ExecutionPlan? plan = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);

        var model = ModelIn(directory)
            ?? throw new FileNotFoundException($"No alignment model under {directory}.");

        var vocabulary = Path.Combine(directory, "vocab.json");
        if (!File.Exists(vocabulary))
        {
            throw new FileNotFoundException($"No alignment vocabulary at {vocabulary}.");
        }

        var options = new SessionOptions
        {
            // The same restraint the rest of the app runs under. Alignment is a second pass over
            // the audio and there is no version of this worth making the machine feel busy for.
            IntraOpNumThreads = plan?.CpuBudget.IntraOpThreads ?? 4,
            InterOpNumThreads = plan?.CpuBudget.InterOpThreads ?? 1,
        };

        return new ForcedAligner(new InferenceSession(model, options), AlignmentAlphabet.Load(vocabulary));
    }

    /// <summary>
    /// Times every word of a segment, or returns null when the segment cannot be aligned.
    /// <para>
    /// Null rather than a guess: the caller has a perfectly good estimate to fall back on, and a
    /// bad alignment presented as a measurement is worse than an honest approximation.
    /// </para>
    /// </summary>
    public IReadOnlyList<WordTimings.Word>? Align(
        PcmAudio audio,
        TranscriptSegment segment,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(segment);

        var words = Words(segment.Text);
        if (words.Count == 0)
        {
            return null;
        }

        var (tokens, spellings) = _alphabet.Spell([.. words.Select(w => w.Text)]);
        if (tokens.Count == 0 || spellings.Count == 0)
        {
            return null;
        }

        var first = Math.Clamp((int)(segment.StartSeconds * audio.SampleRate), 0, audio.Samples.Length);
        var last = Math.Clamp((int)(segment.EndSeconds * audio.SampleRate), first, audio.Samples.Length);

        if (last - first < ShortestAlignableSamples)
        {
            return null;
        }

        cancellationToken.ThrowIfCancellationRequested();

        var samples = Normalise(audio.Samples.AsSpan(first, last - first));

        using var outputs = _session.Run(
        [
            NamedOnnxValue.CreateFromTensor(_input, new DenseTensor<float>(samples, [1, samples.Length])),
        ]);

        var logits = outputs.First().AsTensor<float>();
        var frames = logits.Dimensions[1];
        var alphabet = logits.Dimensions[2];

        if (frames <= 0 || alphabet != _alphabet.Size)
        {
            return null;
        }

        var scores = LogSoftmax(logits.ToArray(), frames, alphabet);
        var placed = CtcForcedAlignment.Align(scores, frames, alphabet, tokens, _alphabet.Blank);

        if (placed is null)
        {
            return null;
        }

        // A frame is a fixed stride of audio, so a frame index is a time.
        var perFrame = (last - first) / (double)frames / audio.SampleRate;

        var spelled = spellings.ToDictionary(spelling => spelling.Index);
        var timed = new List<WordTimings.Word>(words.Count);
        var previous = segment.StartSeconds;

        // Every word gets an entry, including the ones with no letters in them. A lone dash or
        // full stop cannot be aligned — there is no sound to align it to — but leaving it out
        // makes the returned list shorter than the segment's own words, and a caller pairing the
        // two positionally would then give every word after it its neighbour's time. On the
        // debate recording exactly one segment ended in a stray full stop, and that one segment
        // silently fell back to the estimate.
        for (var i = 0; i < words.Count; i++)
        {
            if (!spelled.TryGetValue(i, out var spelling))
            {
                timed.Add(new WordTimings.Word(words[i].Text, previous, previous)
                {
                    Offset = words[i].Offset,
                });

                continue;
            }

            var letters = placed.Skip(spelling.First).Take(spelling.Count).ToList();
            if (letters.Count == 0)
            {
                timed.Add(new WordTimings.Word(words[i].Text, previous, previous) { Offset = words[i].Offset });
                continue;
            }

            var from = segment.StartSeconds + (letters[0].FirstFrame * perFrame);
            var to = segment.StartSeconds + ((letters[^1].LastFrame + 1) * perFrame);

            timed.Add(new WordTimings.Word(spelling.Word, from, Math.Max(to, from))
            {
                Offset = words[i].Offset,
            });

            previous = Math.Max(to, from);
        }

        return timed.Count == 0 ? null : timed;
    }

    /// <summary>
    /// Zero mean and unit variance, which is what the model's own feature extractor does. It is
    /// declared in the model's preprocessor config and is not optional: the network was trained
    /// on normalised audio and a quiet recording fed in raw simply reads as silence.
    /// </summary>
    private static float[] Normalise(ReadOnlySpan<float> samples)
    {
        var mean = 0.0;
        foreach (var sample in samples)
        {
            mean += sample;
        }

        mean /= samples.Length;

        var variance = 0.0;
        foreach (var sample in samples)
        {
            variance += (sample - mean) * (sample - mean);
        }

        var deviation = Math.Sqrt((variance / samples.Length) + 1e-7);
        var scaled = new float[samples.Length];

        for (var i = 0; i < samples.Length; i++)
        {
            scaled[i] = (float)((samples[i] - mean) / deviation);
        }

        return scaled;
    }

    /// <summary>
    /// Turns the model's raw scores into log probabilities, one frame at a time.
    /// <para>
    /// Subtracting the largest score first is not a nicety: without it the exponentials overflow
    /// and every frame comes back as nothing at all.
    /// </para>
    /// </summary>
    private static float[] LogSoftmax(float[] logits, int frames, int alphabet)
    {
        var scores = new float[logits.Length];

        for (var t = 0; t < frames; t++)
        {
            var row = t * alphabet;

            var largest = float.NegativeInfinity;
            for (var k = 0; k < alphabet; k++)
            {
                if (logits[row + k] > largest)
                {
                    largest = logits[row + k];
                }
            }

            var total = 0.0;
            for (var k = 0; k < alphabet; k++)
            {
                total += Math.Exp(logits[row + k] - largest);
            }

            var offset = largest + Math.Log(total);
            for (var k = 0; k < alphabet; k++)
            {
                scores[row + k] = (float)(logits[row + k] - offset);
            }
        }

        return scores;
    }

    /// <summary>Words with where they start in the segment's text, for highlighting.</summary>
    private static List<WordTimings.Word> Words(string text)
    {
        var words = new List<WordTimings.Word>();
        var at = 0;

        while (at < text.Length)
        {
            while (at < text.Length && char.IsWhiteSpace(text[at]))
            {
                at++;
            }

            if (at >= text.Length)
            {
                break;
            }

            var from = at;
            while (at < text.Length && !char.IsWhiteSpace(text[at]))
            {
                at++;
            }

            words.Add(new WordTimings.Word(text[from..at], 0, 0) { Offset = from });
        }

        return words;
    }

    private static string? ModelIn(string directory)
    {
        if (!Directory.Exists(directory))
        {
            return null;
        }

        // Whichever export is present. The half-precision one is preferred: the quantised builds
        // use ConvInteger, which ONNX Runtime has no ARM64 implementation for, so a machine that
        // downloaded one would fail at load rather than run slowly.
        foreach (var name in new[] { "model_fp16.onnx", "model.onnx", "model_fp32.onnx" })
        {
            var path = Path.Combine(directory, name);
            if (File.Exists(path))
            {
                return path;
            }
        }

        return null;
    }

    /// <summary>Below about a fifth of a second there are too few frames to place anything.</summary>
    private const int ShortestAlignableSamples = 16000 / 5;

    public void Dispose() => _session.Dispose();
}
