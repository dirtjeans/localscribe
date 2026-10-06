using System.Diagnostics;
using LocalScribe.Core.Archive;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Pipeline;
using LocalScribe.Core.Transcription;
using LocalScribe.WhisperCpp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Timed = LocalScribe.Doctor.AlignerTrialCommand.Timed;

namespace LocalScribe.Doctor;

/// <summary>
/// Implements <c>--asr-trial &lt;file.scrb&gt;</c>: transcribes a saved recording with NVIDIA's
/// Nemotron 3.5 ASR (streaming, 4-bit ONNX, in models/nemotron-asr) and with the app's Whisper,
/// and reports speed, how far the two transcripts agree, and how well Nemotron's own word times
/// match the MMS fp16 aligner.
/// <para>
/// A trial, not an engine: it answers whether replacing Whisper is worth building. The decoding
/// follows onnxruntime-genai's reference implementation (NemotronStreamingProcessor and
/// NemotronSpeechState) so that a bad number is the model's and not a misreading of it, with one
/// change that cannot alter the output: the prediction network depends only on the last token
/// emitted, so its answer is kept across blank frames instead of recomputed on every one.
/// </para>
/// </summary>
internal static class AsrTrialCommand
{
    public static int Run(string archivePath, string modelRoot, ExecutionPlan plan)
    {
        var contents = TranscriptArchive.Load(archivePath);
        var audio = contents.Audio;
        var seconds = audio.DurationSeconds;

        Console.WriteLine();
        Console.WriteLine($"Transcriber trial — {Path.GetFileName(archivePath)}, {seconds:F0} s, "
            + $"{plan.CpuBudget.IntraOpThreads} threads");
        Console.WriteLine();

        // Nemotron.
        var watch = Stopwatch.StartNew();
        using var nemotron = NemotronAsr.Load(Path.Combine(modelRoot, "nemotron-asr"), plan);
        var nemotronLoad = watch.Elapsed.TotalSeconds;

        watch.Restart();
        var (nemotronText, nemotronWords) = nemotron.Transcribe(audio.Samples);
        var nemotronSeconds = watch.Elapsed.TotalSeconds;

        Console.WriteLine($"  {"Nemotron 3.5 ASR (int4)",-28} {nemotronSeconds,6:F1} s  "
            + $"{seconds / nemotronSeconds,5:F1}x real time   (load {nemotronLoad:F1} s)");
        var (featureTime, encoderTime, decodeTime) = nemotron.LastTimings;
        Console.WriteLine($"  {"",-28} features {featureTime:F1} s, encoder {encoderTime:F1} s, decoding {decodeTime:F1} s");

        // Whisper, exactly as the app runs it.
        var ggml = Directory.EnumerateFiles(Path.Combine(modelRoot, "whisper-cpp"), "ggml-*.bin").First();
        watch.Restart();
        using var whisper = WhisperCppTranscriber.Load(ggml, plan);
        var whisperLoad = watch.Elapsed.TotalSeconds;

        watch.Restart();
        var whisperTranscript = new TranscriptionPipeline(whisper)
            .TranscribeAsync(audio).GetAwaiter().GetResult();
        var whisperSeconds = watch.Elapsed.TotalSeconds;
        var whisperText = whisperTranscript.FullText;

        Console.WriteLine($"  {"Whisper turbo (app path)",-28} {whisperSeconds,6:F1} s  "
            + $"{seconds / whisperSeconds,5:F1}x real time   (load {whisperLoad:F1} s)");
        Console.WriteLine();

        // Agreement. Neither is ground truth; the disagreements are printed so a person can
        // listen and say which one heard right.
        var a = Words(whisperText);
        var b = Words(nemotronText);
        var (edits, subs, ins, del, diffs) = Compare(a, b);

        Console.WriteLine($"  Words: Whisper {a.Count}, Nemotron {b.Count}. They differ on {edits} "
            + $"({100.0 * edits / Math.Max(1, a.Count):F1}% of Whisper's words): "
            + $"{subs} substituted, {ins} only in Nemotron, {del} only in Whisper.");
        Console.WriteLine();
        Console.WriteLine("  Where they disagree (Whisper | Nemotron), longest first:");

        foreach (var (w, n) in diffs.OrderByDescending(d => d.Whisper.Length + d.Nemotron.Length).Take(15))
        {
            Console.WriteLine($"    {Clip(w),-40} | {Clip(n)}");
        }

        Console.WriteLine();

        // Word timing against the aligner, as --aligner-trial grades candidates.
        var reference = AlignerTrialCommand.RunMms(
            Path.Combine(modelRoot, "alignment"), audio, contents.Segments, plan, out _, "model_fp16.onnx");

        if (reference is not null)
        {
            AlignerTrialCommand.Report("Nemotron word times", nemotronSeconds, nemotronWords, reference);

            var heard = whisper.HeardWords
                .Select(w => new Timed(AlignerTrialCommand.Fold(w.Text), w.StartSeconds))
                .Where(w => w.Word.Length > 0)
                .ToList();
            AlignerTrialCommand.Report("Whisper heard words", whisperSeconds, heard, reference);
        }

        File.WriteAllText(Path.ChangeExtension(archivePath, ".nemotron.txt"), nemotronText);
        File.WriteAllText(Path.ChangeExtension(archivePath, ".whisper.txt"), whisperText);
        Console.WriteLine();
        Console.WriteLine($"  Full texts written beside the archive (.nemotron.txt, .whisper.txt).");
        return 0;
    }

    private static string Clip(string text) => text.Length <= 40 ? text : text[..37] + "...";

    private static List<string> Words(string text) =>
        [.. text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            .Select(AlignerTrialCommand.Fold)
            .Where(w => w.Length > 0)];

    /// <summary>Word-level edit distance, with the differing stretches for reading.</summary>
    private static (int Edits, int Subs, int Ins, int Del, List<(string Whisper, string Nemotron)> Diffs)
        Compare(List<string> a, List<string> b)
    {
        var d = new int[a.Count + 1, b.Count + 1];

        for (var i = 0; i <= a.Count; i++)
        {
            d[i, 0] = i;
        }

        for (var j = 0; j <= b.Count; j++)
        {
            d[0, j] = j;
        }

        for (var i = 1; i <= a.Count; i++)
        {
            for (var j = 1; j <= b.Count; j++)
            {
                d[i, j] = Math.Min(
                    d[i - 1, j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1),
                    Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1));
            }
        }

        int x = a.Count, y = b.Count, subs = 0, ins = 0, del = 0;
        var diffs = new List<(string, string)>();
        var wa = new List<string>();
        var wb = new List<string>();

        void Close()
        {
            if (wa.Count + wb.Count > 0)
            {
                wa.Reverse();
                wb.Reverse();
                diffs.Add((string.Join(' ', wa), string.Join(' ', wb)));
                wa.Clear();
                wb.Clear();
            }
        }

        while (x > 0 || y > 0)
        {
            if (x > 0 && y > 0 && a[x - 1] == b[y - 1] && d[x, y] == d[x - 1, y - 1])
            {
                Close();
                x--;
                y--;
            }
            else if (x > 0 && y > 0 && d[x, y] == d[x - 1, y - 1] + 1)
            {
                subs++;
                wa.Add(a[--x]);
                wb.Add(b[--y]);
            }
            else if (x > 0 && d[x, y] == d[x - 1, y] + 1)
            {
                del++;
                wa.Add(a[--x]);
            }
            else
            {
                ins++;
                wb.Add(b[--y]);
            }
        }

        Close();
        return (d[a.Count, b.Count], subs, ins, del, diffs);
    }
}

/// <summary>
/// Nemotron 3.5 ASR streaming, decoded greedily chunk by chunk with the encoder's caches carried
/// across, as onnxruntime-genai does it.
/// </summary>
internal sealed class NemotronAsr : IDisposable
{
    private const int SampleRate = 16_000;
    private const int ChunkSamples = 8960;
    private const int Mels = 128;
    private const int FftSize = 512;
    private const int Hop = 160;
    private const int Window = 400;
    private const float Preemphasis = 0.97f;
    private const float LogEpsilon = 5.96046448e-08f;
    private const int PreEncodeCache = 9;
    private const int Layers = 24;
    private const int LeftContext = 56;
    private const int ConvContext = 8;
    private const int Hidden = 1024;
    private const int LstmLayers = 2;
    private const int LstmSize = 640;
    private const int Blank = 13087;
    private const int MaxSymbolsPerStep = 10;
    private const double SecondsPerFrame = 0.08;

    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly InferenceSession _joint;
    private readonly string[] _vocabulary;
    private readonly float[][] _filters;
    private readonly float[] _window;

    private NemotronAsr(InferenceSession encoder, InferenceSession decoder, InferenceSession joint, string[] vocabulary)
    {
        _encoder = encoder;
        _decoder = decoder;
        _joint = joint;
        _vocabulary = vocabulary;
        _filters = SlaneyFilterbank();

        // Symmetric, as NeMo's streaming extractor uses.
        _window = [.. Enumerable.Range(0, Window).Select(n => (float)(0.5 - (0.5 * Math.Cos(2 * Math.PI * n / (Window - 1)))))];
    }

    public static NemotronAsr Load(string directory, ExecutionPlan plan)
    {
        SessionOptions Options() => new()
        {
            IntraOpNumThreads = plan.CpuBudget.IntraOpThreads,
            InterOpNumThreads = plan.CpuBudget.InterOpThreads,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };

        return new NemotronAsr(
            new InferenceSession(Path.Combine(directory, "encoder.onnx"), Options()),
            new InferenceSession(Path.Combine(directory, "decoder.onnx"), Options()),
            new InferenceSession(Path.Combine(directory, "joint.onnx"), Options()),
            File.ReadAllLines(Path.Combine(directory, "vocab.txt")));
    }

    /// <summary>Where the last transcription's time went: features, encoder, decoding.</summary>
    public (double Features, double Encoder, double Decode) LastTimings { get; private set; }

    public (string Text, List<Timed> Words) Transcribe(float[] samples)
    {
        var featureWatch = new Stopwatch();
        var encoderWatch = new Stopwatch();
        var decodeWatch = new Stopwatch();

        var melCache = new float[PreEncodeCache * Mels];
        var cacheChannel = new DenseTensor<float>([1, Layers, LeftContext, Hidden]);
        var cacheTime = new DenseTensor<float>([1, Layers, Hidden, ConvContext]);
        var cacheChannelLength = new DenseTensor<long>([1]);
        var language = new DenseTensor<long>(new long[] { 0 }, [1]);

        var h = new DenseTensor<float>([LstmLayers, 1, LstmSize]);
        var c = new DenseTensor<float>([LstmLayers, 1, LstmSize]);
        long lastToken = Blank;
        float[]? prediction = null;
        DenseTensor<float>? hNext = null;
        DenseTensor<float>? cNext = null;

        var overlap = new float[FftSize / 2];
        var lastSample = 0f;
        var framesBefore = 0;

        var text = new System.Text.StringBuilder();
        var words = new List<Timed>();
        var building = string.Empty;
        var buildingStart = 0.0;

        void FlushWord()
        {
            var folded = AlignerTrialCommand.Fold(building);
            if (folded.Length > 0)
            {
                words.Add(new Timed(folded, buildingStart));
            }

            building = string.Empty;
        }

        for (var from = 0; from < samples.Length; from += ChunkSamples)
        {
            var chunk = new float[ChunkSamples];
            Array.Copy(samples, from, chunk, 0, Math.Min(ChunkSamples, samples.Length - from));

            featureWatch.Start();
            var (mel, frames) = Mel(chunk, overlap, ref lastSample);
            featureWatch.Stop();
            var total = PreEncodeCache + frames;
            var signal = new DenseTensor<float>([1, total, Mels]);
            var span = signal.Buffer.Span;
            melCache.CopyTo(span);

            for (var t = 0; t < frames; t++)
            {
                for (var m = 0; m < Mels; m++)
                {
                    span[((PreEncodeCache + t) * Mels) + m] = mel[(m * frames) + t];
                }
            }

            // The newest frames become the next chunk's left context.
            for (var t = 0; t < PreEncodeCache; t++)
            {
                for (var m = 0; m < Mels; m++)
                {
                    melCache[(t * Mels) + m] = mel[(m * frames) + (frames - PreEncodeCache + t)];
                }
            }

            encoderWatch.Start();
            using var encoded = _encoder.Run(
            [
                NamedOnnxValue.CreateFromTensor("audio_signal", signal),
                NamedOnnxValue.CreateFromTensor("length", new DenseTensor<long>(new long[] { total }, [1])),
                NamedOnnxValue.CreateFromTensor("cache_last_channel", cacheChannel),
                NamedOnnxValue.CreateFromTensor("cache_last_time", cacheTime),
                NamedOnnxValue.CreateFromTensor("cache_last_channel_len", cacheChannelLength),
                NamedOnnxValue.CreateFromTensor("lang_id", language),
            ]);

            var outputs = encoded.ToDictionary(v => v.Name);
            var encoderOut = outputs["outputs"].AsTensor<float>();
            var encodedLength = (int)outputs["encoded_lengths"].AsTensor<long>().GetValue(0);
            cacheChannel = Copy(outputs["cache_last_channel_next"].AsTensor<float>());
            cacheTime = Copy(outputs["cache_last_time_next"].AsTensor<float>());
            cacheChannelLength = new DenseTensor<long>(
                new[] { outputs["cache_last_channel_len_next"].AsTensor<long>().GetValue(0) }, [1]);

            var steps = Math.Min(encoderOut.Dimensions[1], encodedLength);
            encoderWatch.Stop();
            decodeWatch.Start();

            for (var t = 0; t < steps; t++)
            {
                var frame = new DenseTensor<float>([1, 1, Hidden]);
                for (var k = 0; k < Hidden; k++)
                {
                    frame.Buffer.Span[k] = encoderOut[0, t, k];
                }

                var symbols = 0;

                while (true)
                {
                    if (prediction is null)
                    {
                        using var predicted = _decoder.Run(
                        [
                            NamedOnnxValue.CreateFromTensor("targets", new DenseTensor<long>(new[] { lastToken }, [1, 1])),
                            NamedOnnxValue.CreateFromTensor("h_in", h),
                            NamedOnnxValue.CreateFromTensor("c_in", c),
                        ]);
                        var p = predicted.ToDictionary(v => v.Name);
                        prediction = [.. p["decoder_output"].AsTensor<float>()];
                        hNext = Copy(p["h_out"].AsTensor<float>());
                        cNext = Copy(p["c_out"].AsTensor<float>());
                    }

                    using var joined = _joint.Run(
                    [
                        NamedOnnxValue.CreateFromTensor("encoder_output", frame),
                        NamedOnnxValue.CreateFromTensor("decoder_output", new DenseTensor<float>(prediction, [1, 1, prediction.Length])),
                    ]);

                    var logits = joined.First().AsTensor<float>();
                    var best = 0;
                    var bestValue = float.NegativeInfinity;
                    var i = 0;

                    foreach (var value in logits)
                    {
                        if (value > bestValue)
                        {
                            bestValue = value;
                            best = i;
                        }

                        i++;
                    }

                    if (best == Blank)
                    {
                        break;
                    }

                    // Emitted: the prediction network moves on to this token.
                    lastToken = best;
                    h = hNext!;
                    c = cNext!;
                    prediction = null;

                    var piece = _vocabulary[best];

                    if (!(piece.StartsWith('<') && piece.EndsWith('>')))
                    {
                        var starts = piece.StartsWith('▁');
                        var visible = piece.Replace('▁', ' ');

                        if (starts)
                        {
                            FlushWord();
                            buildingStart = (framesBefore + t) * SecondsPerFrame;
                        }
                        else if (building.Length == 0)
                        {
                            buildingStart = (framesBefore + t) * SecondsPerFrame;
                        }

                        building += visible;
                        text.Append(visible);
                    }

                    if (++symbols >= MaxSymbolsPerStep)
                    {
                        break;
                    }
                }
            }

            framesBefore += steps;
            decodeWatch.Stop();
        }

        LastTimings = (featureWatch.Elapsed.TotalSeconds, encoderWatch.Elapsed.TotalSeconds, decodeWatch.Elapsed.TotalSeconds);

        FlushWord();
        return (text.ToString().Trim(), words);
    }

    private static DenseTensor<float> Copy(Tensor<float> source) =>
        new DenseTensor<float>(source.ToArray(), source.Dimensions.ToArray());

    /// <summary>
    /// One chunk of NeMo's streaming log-mel: pre-emphasis carried across chunks, half an FFT of
    /// the previous chunk prepended in place of centre padding, frequency-major out.
    /// </summary>
    private (float[] Mel, int Frames) Mel(float[] chunk, float[] overlap, ref float lastSample)
    {
        var pre = new float[chunk.Length];
        pre[0] = chunk[0] - (Preemphasis * lastSample);
        for (var i = 1; i < chunk.Length; i++)
        {
            pre[i] = chunk[i] - (Preemphasis * chunk[i - 1]);
        }

        lastSample = chunk[^1];

        var pad = FftSize / 2;
        var offset = (FftSize - Window) / 2;
        var padded = new float[pad + chunk.Length + offset];
        overlap.CopyTo(padded, 0);
        pre.CopyTo(padded, pad);
        Array.Copy(pre, pre.Length - pad, overlap, 0, pad);

        var frames = ((padded.Length - offset - Window) / Hop) + 1;
        var mel = new float[Mels * frames];
        var re = new double[FftSize];
        var im = new double[FftSize];
        var power = new double[(FftSize / 2) + 1];

        for (var t = 0; t < frames; t++)
        {
            Array.Clear(re);
            Array.Clear(im);
            var start = (t * Hop) + offset;

            for (var n = 0; n < Window; n++)
            {
                re[n] = padded[start + n] * _window[n];
            }

            Fft(re, im);

            for (var k = 0; k < power.Length; k++)
            {
                power[k] = (re[k] * re[k]) + (im[k] * im[k]);
            }

            for (var m = 0; m < Mels; m++)
            {
                var sum = 0.0;
                var filter = _filters[m];
                for (var k = 0; k < power.Length; k++)
                {
                    sum += filter[k] * power[k];
                }

                mel[(m * frames) + t] = (float)Math.Log(sum + LogEpsilon);
            }
        }

        return (mel, frames);
    }

    /// <summary>Slaney-scale, area-normalised triangles, as librosa and NeMo build them.</summary>
    private static float[][] SlaneyFilterbank()
    {
        static double HzToMel(double hz) => hz < 1000 ? hz / (200.0 / 3) : 15 + (Math.Log(hz / 1000) / (Math.Log(6.4) / 27));
        static double MelToHz(double mel) => mel < 15 ? mel * (200.0 / 3) : 1000 * Math.Exp((mel - 15) * (Math.Log(6.4) / 27));

        var bins = (FftSize / 2) + 1;
        var low = HzToMel(0);
        var high = HzToMel(SampleRate / 2.0);
        var centres = Enumerable.Range(0, Mels + 2).Select(i => MelToHz(low + ((high - low) * i / (Mels + 1)))).ToArray();
        var filters = new float[Mels][];

        for (var m = 0; m < Mels; m++)
        {
            filters[m] = new float[bins];
            var norm = 2.0 / (centres[m + 2] - centres[m] + 1e-10);

            for (var k = 0; k < bins; k++)
            {
                var f = (double)k * SampleRate / FftSize;
                var lower = (f - centres[m]) / (centres[m + 1] - centres[m] + 1e-10);
                var upper = (centres[m + 2] - f) / (centres[m + 2] - centres[m + 1] + 1e-10);
                filters[m][k] = (float)(Math.Max(0, Math.Min(lower, upper)) * norm);
            }
        }

        return filters;
    }

    /// <summary>In-place radix-2 FFT; the size here is always 512.</summary>
    private static void Fft(double[] re, double[] im)
    {
        var n = re.Length;

        for (int i = 1, j = 0; i < n; i++)
        {
            var bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
            {
                j ^= bit;
            }

            j ^= bit;

            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        for (var length = 2; length <= n; length <<= 1)
        {
            var angle = -2 * Math.PI / length;
            var wr = Math.Cos(angle);
            var wi = Math.Sin(angle);

            for (var i = 0; i < n; i += length)
            {
                double cr = 1, ci = 0;

                for (var k = 0; k < length / 2; k++)
                {
                    var ur = re[i + k];
                    var ui = im[i + k];
                    var vr = (re[i + k + (length / 2)] * cr) - (im[i + k + (length / 2)] * ci);
                    var vi = (re[i + k + (length / 2)] * ci) + (im[i + k + (length / 2)] * cr);
                    re[i + k] = ur + vr;
                    im[i + k] = ui + vi;
                    re[i + k + (length / 2)] = ur - vr;
                    im[i + k + (length / 2)] = ui - vi;
                    var next = (cr * wr) - (ci * wi);
                    ci = (cr * wi) + (ci * wr);
                    cr = next;
                }
            }
        }
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _decoder.Dispose();
        _joint.Dispose();
    }
}
