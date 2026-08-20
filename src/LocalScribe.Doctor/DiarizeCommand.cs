using System.Diagnostics;
using System.Globalization;
using LocalScribe.Core.Diarization;
using LocalScribe.Onnx;

namespace LocalScribe.Doctor;

/// <summary>
/// Implements <c>--diarize</c>: runs the speaker models over a file and prints who spoke when.
/// <para>
/// Separate from transcription on purpose. Diarization has its own failure modes — splitting one
/// speaker in two, merging two into one, drifting boundaries — and none of them are visible when
/// the output is already tangled up with the words. Turns and times alone can be checked against
/// a recording by ear.
/// </para>
/// </summary>
internal static class DiarizeCommand
{
    public static int Run(string audioPath, string modelDirectory, string? speakers, string? threshold)
    {
        if (!File.Exists(audioPath))
        {
            Console.Error.WriteLine($"No such file: {audioPath}");
            return 1;
        }

        Heading("Diarize");
        Console.WriteLine($"  Audio      {audioPath}");
        Console.WriteLine($"  Models     {modelDirectory}");

        var audio = WavReader.Read(audioPath);
        audio.EnsureWhisperFormat();

        Console.WriteLine($"  Duration   {audio.DurationSeconds:F1}s");

        var maxSpeakers = int.TryParse(speakers, out var parsedSpeakers) ? parsedSpeakers : (int?)null;
        var distance = double.TryParse(threshold, CultureInfo.InvariantCulture, out var parsedThreshold)
            ? parsedThreshold
            : SpeakerClustering.DefaultThreshold;

        Console.WriteLine($"  Threshold  {distance:F2}{(maxSpeakers is { } n ? $", at most {n} speakers" : string.Empty)}");
        Console.WriteLine();

        IReadOnlyList<SpeakerTurn> turns;
        var watch = Stopwatch.StartNew();

        try
        {
            using var diarizer = SpeakerDiarizer.Load(modelDirectory);
            turns = diarizer.Diarize(audio, distance, maxSpeakers);
        }
        catch (FileNotFoundException exception)
        {
            Console.Error.WriteLine($"  {exception.Message}");
            return 1;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"  Diarization failed: {exception.GetType().Name}: {exception.Message}");
            return 1;
        }

        watch.Stop();

        Heading("Turns");

        if (turns.Count == 0)
        {
            Console.WriteLine("  (no speech found)");
            return 1;
        }

        foreach (var turn in turns)
        {
            Console.WriteLine(
                $"  {turn.StartSeconds,7:F2} - {turn.EndSeconds,7:F2}  ({turn.DurationSeconds,5:F2}s)  {turn.Label}");
        }

        var distinct = turns.Select(t => t.Speaker).Distinct().Count();
        var speech = turns.Sum(t => t.DurationSeconds);

        Heading("Summary");
        Console.WriteLine($"  Speakers   {distinct}");
        Console.WriteLine($"  Turns      {turns.Count}");
        Console.WriteLine($"  Speech     {speech:F1}s of {audio.DurationSeconds:F1}s");
        Console.WriteLine($"  Took       {watch.Elapsed.TotalSeconds:F1}s "
            + $"({audio.DurationSeconds / Math.Max(0.001, watch.Elapsed.TotalSeconds):F1}x real time)");

        return 0;
    }

    /// <summary>
    /// Embeds known spans and prints the distances between them.
    /// <para>
    /// The one measurement that separates the two ways this pipeline fails. If a speaker's own
    /// spans are closer to each other than to the other speaker's, the features and the
    /// embedding model are right and any mistake is in segmentation or clustering. If they are
    /// not, nothing downstream can be fixed, because the vectors do not carry identity.
    /// </para>
    /// </summary>
    public static int Matrix(string audioPath, string modelDirectory, string spans)
    {
        var audio = WavReader.Read(audioPath);
        audio.EnsureWhisperFormat();

        var parsed = spans.Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(part => part.Split(':'))
            .Select(p => (
                Start: double.Parse(p[0], CultureInfo.InvariantCulture),
                End: double.Parse(p[1], CultureInfo.InvariantCulture)))
            .ToList();

        using var diarizer = SpeakerDiarizer.Load(modelDirectory);

        var vectors = parsed
            .Select(span => diarizer.EmbedSpan(audio, span.Start, span.End))
            .ToList();

        Heading("Distance between spans");
        Console.Write("        ");
        for (var i = 0; i < parsed.Count; i++)
        {
            Console.Write($"{i,7}");
        }

        Console.WriteLine();

        for (var i = 0; i < vectors.Count; i++)
        {
            Console.Write($"  {i,2} {parsed[i].Start,4:F1}");

            for (var j = 0; j < vectors.Count; j++)
            {
                var d = vectors[i] is null || vectors[j] is null
                    ? double.NaN
                    : SpeakerClustering.CosineDistance(Unit(vectors[i]!), Unit(vectors[j]!));

                Console.Write($"{d,7:F3}");
            }

            Console.WriteLine();
        }

        return 0;
    }

    private static float[] Unit(float[] v)
    {
        var sum = v.Sum(x => (double)x * x);
        var magnitude = Math.Sqrt(sum);

        return magnitude < 1e-12 ? v : v.Select(x => (float)(x / magnitude)).ToArray();
    }

    private static void Heading(string title)
    {
        Console.WriteLine();
        Console.WriteLine(title);
        Console.WriteLine(new string('-', title.Length));
    }
}
