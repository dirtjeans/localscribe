using LocalScribe.Core.Archive;
using LocalScribe.Core.Diarization;
using LocalScribe.Onnx;

namespace LocalScribe.Doctor;

/// <summary>
/// Compares speaker embedding models on a transcript whose speakers are already known.
/// <para>
/// Published error rates are measured on VoxCeleb, which is celebrity interviews recorded
/// reasonably well. They rank models usefully and they do not answer the question that matters
/// here, which is whether a model tells two voices apart on the recording in front of you. A
/// saved transcript already carries the answer: the audio, and a label on every segment saying
/// who said it.
/// </para>
/// <para>
/// The measure is the one that exposed the missing cepstral mean normalisation: how far apart
/// two spans of the same voice sit, against how far apart two different voices sit. A model is
/// better here when that gap is wider, not when its own paper says so — and the overlap between
/// the two distributions matters more than either average, because that is what a single
/// threshold has to separate.
/// </para>
/// </summary>
public static class SpeakerModelsCommand
{
    public static int Run(string archivePath, string candidateRoot)
    {
        if (!File.Exists(archivePath))
        {
            Console.Error.WriteLine($"No such file: {archivePath}");
            return 1;
        }

        if (!Directory.Exists(candidateRoot))
        {
            Console.Error.WriteLine($"No such directory: {candidateRoot}");
            return 1;
        }

        Heading("Speaker models");
        Console.WriteLine($"  Archive    {archivePath}");

        TranscriptArchive.Contents contents;

        using (var file = File.OpenRead(archivePath))
        {
            contents = TranscriptArchive.Read(file);
        }

        // Only labelled segments long enough to embed. A half-second of speech gives an embedding
        // dominated by whatever phoneme happened to be in it.
        var spans = contents.Segments
            .Where(s => s.Speaker is { Length: > 0 })
            .Where(s => s.EndSeconds - s.StartSeconds >= ShortestSpanSeconds)
            .ToList();

        var speakers = spans.Select(s => s.Speaker!).Distinct().Count();

        Console.WriteLine(
            $"  Spans      {spans.Count} of {contents.Segments.Count} segments, {speakers} speakers");

        if (spans.Count < 4 || speakers < 2)
        {
            Console.Error.WriteLine("  Not enough labelled speech to compare anything.");
            return 1;
        }

        var candidates = Directory.GetDirectories(candidateRoot)
            .Where(d => File.Exists(Path.Combine(d, "embedding.onnx")))
            .OrderBy(d => d, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (candidates.Count == 0)
        {
            Console.Error.WriteLine($"  No model directories under {candidateRoot}.");
            return 1;
        }

        Heading("Results");
        Console.WriteLine(
            $"  {"model",-28} {"same",7} {"different",10} {"gap",7} {"confusable",11} {"threshold",10}");

        foreach (var candidate in candidates)
        {
            Measure(candidate, contents, spans);
        }

        Console.WriteLine();
        Console.WriteLine("  same       average distance between two spans of one voice (lower is better)");
        Console.WriteLine("  different  average distance between two voices (higher is better)");
        Console.WriteLine("  confusable share of pairs a single threshold must get wrong (lower is better)");
        Console.WriteLine("  threshold  the distance that separates them best on this recording");

        return 0;
    }

    private const double ShortestSpanSeconds = 1.5;

    private static void Measure(
        string directory,
        TranscriptArchive.Contents contents,
        IReadOnlyList<Core.Transcription.TranscriptSegment> spans)
    {
        var name = Path.GetFileName(directory);

        try
        {
            using var diarizer = SpeakerDiarizer.Load(directory);

            var voices = new List<(string Speaker, float[] Embedding)>();

            foreach (var span in spans)
            {
                if (diarizer.EmbedSpan(contents.Audio, span.StartSeconds, span.EndSeconds) is { } embedding)
                {
                    voices.Add((span.Speaker!, Unit(embedding)));
                }
            }

            if (voices.Count < 4)
            {
                Console.WriteLine($"  {name,-28} too few spans could be embedded");
                return;
            }

            var same = new List<double>();
            var different = new List<double>();

            for (var i = 0; i < voices.Count; i++)
            {
                for (var j = i + 1; j < voices.Count; j++)
                {
                    var distance = SpeakerClustering.CosineDistance(voices[i].Embedding, voices[j].Embedding);

                    (voices[i].Speaker == voices[j].Speaker ? same : different).Add(distance);
                }
            }

            if (same.Count == 0 || different.Count == 0)
            {
                Console.WriteLine($"  {name,-28} nothing to compare");
                return;
            }

            var (threshold, confusable) = BestThreshold(same, different);

            Console.WriteLine(
                $"  {name,-28} {same.Average(),7:F3} {different.Average(),10:F3} "
                + $"{different.Average() - same.Average(),7:F3} {confusable,10:P1} {threshold,10:F2}");
        }
        catch (Exception exception)
        {
            Console.WriteLine($"  {name,-28} failed: {exception.Message}");
        }
    }

    /// <summary>
    /// Scales an embedding to unit length.
    /// <para>
    /// Not optional, and not a detail. CosineDistance is one minus the dot product, which is a
    /// cosine only when both sides are unit vectors; on raw embeddings it returns whatever the
    /// magnitudes happen to multiply out to. The first run of this printed distances of minus
    /// five thousand, and the models differ enough in scale that comparing them without this
    /// would have ranked them by how large their outputs are.
    /// </para>
    /// </summary>
    private static float[] Unit(float[] embedding)
    {
        var sum = 0.0;

        foreach (var value in embedding)
        {
            sum += value * value;
        }

        var length = Math.Sqrt(sum);

        if (length <= 0)
        {
            return embedding;
        }

        var scaled = new float[embedding.Length];

        for (var i = 0; i < embedding.Length; i++)
        {
            scaled[i] = (float)(embedding[i] / length);
        }

        return scaled;
    }

    /// <summary>
    /// The distance that separates the two sets best, and the share of pairs it still gets wrong.
    /// <para>
    /// Reported rather than the averages alone because the averages can flatter a model whose
    /// distributions overlap badly. What a threshold has to do is separate them, and this is how
    /// well the best possible one manages.
    /// </para>
    /// </summary>
    private static (double Threshold, double Confusable) BestThreshold(
        List<double> same,
        List<double> different)
    {
        var best = (Threshold: 0.5, Wrong: double.MaxValue);

        for (var threshold = 0.05; threshold <= 1.5; threshold += 0.01)
        {
            // A same-voice pair above the threshold is split in two; a different-voice pair below
            // it is merged into one. Both are the same kind of mistake to a reader.
            var wrong = (same.Count(d => d > threshold) / (double)same.Count)
                + (different.Count(d => d <= threshold) / (double)different.Count);

            if (wrong < best.Wrong)
            {
                best = (threshold, wrong);
            }
        }

        return (best.Threshold, best.Wrong / 2);
    }

    private static void Heading(string title)
    {
        Console.WriteLine();
        Console.WriteLine(title);
        Console.WriteLine(new string('-', title.Length));
    }
}
