using System.Diagnostics;
using LocalScribe.Core.Archive;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Transcription;
using LocalScribe.Onnx;
using Whisper.net;
using Whisper.net.LibraryLoader;

namespace LocalScribe.Doctor;

/// <summary>
/// Implements <c>--aligner-trial &lt;file.scrb&gt;</c>: times candidate word aligners on a saved
/// transcript and grades each one's word times against the reference aligner, word by word.
/// <para>
/// The reference is the MMS fp16 aligner, which --check-words has graded against the audio
/// (drift +0.00 in every fifth on both reference recordings). Agreement with it is therefore
/// a proxy for accuracy that a candidate cannot fake: it never sees the reference's answers.
/// Candidates: the 4-bit MMS builds in models/alignment-q4 and models/alignment-q4f16, and
/// Whisper's own cross-attention (DTW) token timestamps, which would remove the scan entirely.
/// </para>
/// </summary>
internal static class AlignerTrialCommand
{
    private sealed record Timed(string Word, double Start);

    public static int Run(string archivePath, string modelRoot, ExecutionPlan plan)
    {
        var contents = TranscriptArchive.Load(archivePath);
        var audio = contents.Audio;
        var segments = contents.Segments;

        Console.WriteLine();
        Console.WriteLine($"Aligner trial — {Path.GetFileName(archivePath)}, "
            + $"{audio.DurationSeconds:F0} s, {segments.Count} segments");
        Console.WriteLine();

        var reference = RunMms(Path.Combine(modelRoot, "alignment"), audio, segments, plan, out var refSeconds, "model_fp16.onnx");

        if (reference is null)
        {
            Console.Error.WriteLine("The reference aligner produced no words.");
            return 1;
        }

        Console.WriteLine($"  {"MMS fp16 (reference)",-28} {refSeconds,6:F1} s   {reference.Count} words");

        foreach (var variant in new[] { "q4", "q4f16" })
        {
            var directory = Path.Combine(modelRoot, $"alignment-{variant}");

            if (!Directory.Exists(directory))
            {
                continue;
            }

            try
            {
                var words = RunMms(directory, audio, segments, plan, out var seconds, $"model_{variant}.onnx");
                Report($"MMS {variant}", seconds, words, reference);
            }
            catch (Exception exception)
            {
                Console.WriteLine($"  {"MMS " + variant,-28} failed: {exception.GetType().Name}: {exception.Message}");
            }
        }

        var ggml = Directory.Exists(Path.Combine(modelRoot, "whisper-cpp"))
            ? Directory.EnumerateFiles(Path.Combine(modelRoot, "whisper-cpp"), "ggml-*.bin").FirstOrDefault()
            : null;

        if (ggml is not null)
        {
            try
            {
                var words = RunWhisperDtw(ggml, audio.Samples, plan, out var seconds);
                Report("Whisper DTW (transcribes too)", seconds, words, reference);
            }
            catch (Exception exception)
            {
                Console.WriteLine($"  {"Whisper DTW",-28} failed: {exception.GetType().Name}: {exception.Message}");
            }
        }

        Console.WriteLine();
        Console.WriteLine("  Offsets are each word's start against the reference's. Drift is the median");
        Console.WriteLine("  signed offset in each fifth of the recording; a candidate that slides shows it there.");
        return 0;
    }

    private static List<Timed>? RunMms(
        string directory, Core.Audio.PcmAudio audio, IReadOnlyList<TranscriptSegment> segments,
        ExecutionPlan plan, out double seconds, string modelFileName)
    {
        var watch = Stopwatch.StartNew();
        using var aligner = ForcedAligner.Load(directory, plan, modelFileName);
        var scores = aligner.Scan(audio) ?? throw new InvalidOperationException("scan returned nothing");
        var placed = aligner.AlignAll(scores, segments);
        seconds = watch.Elapsed.TotalSeconds;

        return [.. placed
            .Where(words => words is not null)
            .SelectMany(words => words!)
            .Where(word => word.EndSeconds > word.StartSeconds)
            .Select(word => new Timed(Fold(word.Text), word.StartSeconds))
            .Where(word => word.Word.Length > 0)];
    }

    private static List<Timed> RunWhisperDtw(string ggml, float[] samples, ExecutionPlan plan, out double seconds)
    {
        RuntimeOptions.RuntimeLibraryOrder = [RuntimeLibrary.CoreML, RuntimeLibrary.Cpu];

        var watch = Stopwatch.StartNew();

        using var factory = WhisperFactory.FromPath(ggml, new WhisperFactoryOptions
        {
            UseDtwTimeStamps = true,
            HeadsPreset = WhisperAlignmentHeadsPreset.LargeV3Turbo,
        });

        using var processor = factory.CreateBuilder()
            .WithThreads(plan.CpuBudget.IntraOpThreads)
            .WithLanguage("en")
            .WithTokenTimestamps()
            .Build();

        var words = new List<Timed>();
        var building = string.Empty;
        var start = 0.0;

        void Flush()
        {
            var folded = Fold(building);

            if (folded.Length > 0)
            {
                words.Add(new Timed(folded, start));
            }

            building = string.Empty;
        }

        // Thirty-second windows, as the app transcribes: handed a whole recording at once,
        // whisper.cpp stopped at 55 s of a 116 s file. DTW times are relative to each call's
        // audio, so each window's offset is added back.
        const int window = 30 * 16_000;

        for (var from = 0; from < samples.Length; from += window)
        {
        var offset = from / 16_000.0;
        var piece = samples.AsSpan(from, Math.Min(window, samples.Length - from)).ToArray();

        foreach (var segment in processor.ProcessAsync(piece).ToBlockingEnumerable())
        {
            foreach (var token in segment.Tokens)
            {
                var text = token.Text ?? string.Empty;

                // Special tokens ([_BEG_], [_TT_123], <|en|>) carry no word.
                if (text.StartsWith("[_", StringComparison.Ordinal) || text.StartsWith("<|", StringComparison.Ordinal))
                {
                    continue;
                }

                // A leading space starts a new word; whisper's BPE pieces otherwise continue one.
                if (text.StartsWith(' ') && building.Length > 0)
                {
                    Flush();
                }

                if (building.Length == 0)
                {
                    start = offset + (token.DtwTimestamp / 100.0);
                }

                building += text;
            }

            Flush();
        }
        }

        seconds = watch.Elapsed.TotalSeconds;

        if (Environment.GetEnvironmentVariable("LOCALSCRIBE_DTW_DUMP") is { Length: > 0 } dump)
        {
            File.WriteAllLines(dump, words.Select(w => $"{w.Start:F2}\t{w.Word}"));
        }

        return words;
    }

    private static void Report(string name, double seconds, List<Timed>? words, List<Timed> reference)
    {
        if (words is null || words.Count == 0)
        {
            Console.WriteLine($"  {name,-28} {seconds,6:F1} s   no words");
            return;
        }

        var pairs = Match(reference, words);
        var offsets = pairs.Select(p => words[p.Candidate].Start - reference[p.Reference].Start).ToList();
        var absolute = offsets.Select(Math.Abs).OrderBy(x => x).ToList();

        double Share(double limit) => 100.0 * absolute.Count(x => x <= limit) / Math.Max(1, absolute.Count);

        Console.WriteLine($"  {name,-28} {seconds,6:F1} s   matched {pairs.Count} of {reference.Count}; "
            + $"median {Percentile(absolute, 0.5):F2} s, p90 {Percentile(absolute, 0.9):F2} s; "
            + $"within 0.1 s {Share(0.1):F0}%, 0.25 s {Share(0.25):F0}%, 0.5 s {Share(0.5):F0}%");

        var duration = reference[^1].Start;
        var fifths = Enumerable.Range(0, 5).Select(f =>
        {
            var inFifth = pairs
                .Select((p, i) => (Time: reference[p.Reference].Start, Offset: offsets[i]))
                .Where(x => x.Time >= duration * f / 5 && x.Time < duration * (f + 1) / 5)
                .Select(x => x.Offset)
                .OrderBy(x => x)
                .ToList();
            return inFifth.Count == 0 ? "   n/a" : $"{Percentile(inFifth, 0.5),+6:+0.00;-0.00}";
        });

        Console.WriteLine($"  {"",-28}          drift by fifth: {string.Join("  ", fifths)}");

        // A constant lag can be subtracted; scatter cannot. Centred on its own median, the
        // spread is what a corrected candidate would actually deliver.
        var bias = Percentile(offsets.OrderBy(o => o).ToList(), 0.5);
        var centred = offsets.Select(o => Math.Abs(o - bias)).OrderBy(o => o).ToList();
        double CentredShare(double limit) => 100.0 * centred.Count(x => x <= limit) / Math.Max(1, centred.Count);

        Console.WriteLine($"  {"",-28}          with its {bias:+0.00;-0.00} s lag removed: median {Percentile(centred, 0.5):F2} s, "
            + $"p90 {Percentile(centred, 0.9):F2} s; within 0.1 s {CentredShare(0.1):F0}%, 0.25 s {CentredShare(0.25):F0}%");
    }

    /// <summary>Longest common subsequence over folded words: pairs the two readings in order.</summary>
    private static List<(int Reference, int Candidate)> Match(List<Timed> a, List<Timed> b)
    {
        var table = new int[a.Count + 1, b.Count + 1];

        for (var i = a.Count - 1; i >= 0; i--)
        {
            for (var j = b.Count - 1; j >= 0; j--)
            {
                table[i, j] = a[i].Word == b[j].Word
                    ? table[i + 1, j + 1] + 1
                    : Math.Max(table[i + 1, j], table[i, j + 1]);
            }
        }

        var pairs = new List<(int, int)>();
        int x = 0, y = 0;

        while (x < a.Count && y < b.Count)
        {
            if (a[x].Word == b[y].Word)
            {
                pairs.Add((x++, y++));
            }
            else if (table[x + 1, y] >= table[x, y + 1])
            {
                x++;
            }
            else
            {
                y++;
            }
        }

        return pairs;
    }

    private static double Percentile(List<double> sorted, double p) =>
        sorted.Count == 0 ? 0 : sorted[Math.Min(sorted.Count - 1, (int)(p * sorted.Count))];

    private static string Fold(string text) =>
        new([.. text.Where(char.IsLetterOrDigit).Select(char.ToLowerInvariant)]);
}
