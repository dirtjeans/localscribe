using LocalScribe.Core.Audio;
using System.Diagnostics;
using System.Globalization;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Hardware;
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
    /// <summary>
    /// Runs the models once, then clusters the same embeddings at every threshold in turn.
    /// <para>
    /// The threshold decides whether two people are told apart, and the right value is a
    /// property of the recording rather than of the code — how alike the voices are, how much
    /// room noise there is, how long anyone talks uninterrupted. It has been calibrated twice on
    /// a sample of synthesised speech, which is cleaner and more consistent than people, and
    /// been wrong about real recordings both times.
    /// </para>
    /// <para>
    /// So this shows the answer rather than assuming it: what the speaker count does across the
    /// range, and how the distances between stretches of speech are actually distributed. Two
    /// speakers who are genuinely distinguishable leave a gap in that distribution, and the
    /// threshold belongs in the gap.
    /// </para>
    /// </summary>
    public static int Sweep(string audioPath, string modelDirectory)
    {
        if (!File.Exists(audioPath))
        {
            Console.Error.WriteLine($"No such file: {audioPath}");
            return 1;
        }

        Heading("Threshold sweep");
        Console.WriteLine($"  Audio      {audioPath}");

        var audio = WavReader.Read(audioPath);
        audio.EnsureWhisperFormat();
        Console.WriteLine($"  Duration   {audio.DurationSeconds:F1}s");

        IReadOnlyList<SpeakerDiarizer.Voice> voices;
        var watch = Stopwatch.StartNew();

        try
        {
            using var diarizer = SpeakerDiarizer.Load(
            modelDirectory,
            OperatingSystem.IsMacOS() ? AcceleratorPlanner.Plan(DeviceProbe.Probe()) : null);
            voices = diarizer.Describe(audio);
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"  Failed: {exception.GetType().Name}: {exception.Message}");
            return 1;
        }

        if (voices.Count < 2)
        {
            Console.Error.WriteLine($"  Only {voices.Count} stretch(es) of speech found — nothing to compare.");
            return 1;
        }

        Console.WriteLine($"  Speech     {voices.Count} stretches, "
            + $"{voices.Sum(v => v.DurationSeconds):F1}s, median {Median([.. voices.Select(v => v.DurationSeconds)]):F1}s");
        Console.WriteLine($"  Measured   in {watch.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine();

        var points = voices.Select(v => Unit(v.Embedding)).ToList();

        var distances = new List<double>();
        for (var i = 0; i < points.Count; i++)
        {
            for (var j = i + 1; j < points.Count; j++)
            {
                distances.Add(SpeakerClustering.CosineDistance(points[i], points[j]));
            }
        }

        distances.Sort();

        Heading("How far apart the stretches of speech are");
        foreach (var percentile in new[] { 5, 10, 25, 50, 75, 90, 95 })
        {
            Console.WriteLine($"  {percentile,3}%      {distances[(int)((percentile / 100.0) * (distances.Count - 1))]:F3}");
        }

        Console.WriteLine();
        Console.WriteLine("  A recording with two distinguishable speakers has two humps here: the");
        Console.WriteLine("  low one is the same person twice, the high one is two people. If the");
        Console.WriteLine("  numbers climb smoothly with no gap, the voices are not separable and no");
        Console.WriteLine("  threshold will help.");
        Console.WriteLine();

        Heading("Speakers found, by threshold");
        Console.WriteLine("  threshold   speakers   largest share of speech");

        for (var threshold = 0.10; threshold <= 0.75001; threshold += 0.05)
        {
            // The diarizer's own clustering, not the plain kind: short stretches get attached to
            // whichever neighbour they sound like rather than forming speakers of their own, and
            // that changes the count. Sweeping the plain kind reported three speakers where the
            // app finds two.
            var labels = SpeakerDiarizer.Assign(voices, threshold);
            var found = labels.Distinct().Count();

            var biggest = labels
                .Select((label, i) => (label, seconds: voices[i].DurationSeconds))
                .GroupBy(x => x.label)
                .Max(g => g.Sum(x => x.seconds));

            var share = biggest / voices.Sum(v => v.DurationSeconds);
            var marker = Math.Abs(threshold - SpeakerClustering.DefaultThreshold) < 0.025 ? "  <- current" : string.Empty;

            Console.WriteLine($"  {threshold,6:F2}      {found,5}      {share,6:P0}{marker}");
        }

        Console.WriteLine();
        Console.WriteLine("  Pick the threshold where the count matches the number of people in the");
        Console.WriteLine("  room and stays there across a few rows. A count that changes at every");
        Console.WriteLine("  row means the voices are not well separated in this recording.");
        Console.WriteLine();

        return 0;
    }

    private static double Median(double[] values)
    {
        Array.Sort(values);
        return values.Length == 0 ? 0 : values[values.Length / 2];
    }

    public static int Run(
        string audioPath,
        string modelDirectory,
        string? speakers,
        string? threshold,
        bool tracking = false)
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
        Console.WriteLine($"  Method     {(tracking ? "following speakers between windows" : "comparing voices")}");
        Console.WriteLine();

        IReadOnlyList<SpeakerTurn> turns;
        var watch = Stopwatch.StartNew();

        // Read off the diarizer before it goes out of scope, since the counts live on it.
        var found = 0;
        var kept = 0;
        var nested = 0;
        IReadOnlyList<(double Start, double End)> contested = [];

        try
        {
            using var diarizer = SpeakerDiarizer.Load(
            modelDirectory,
            OperatingSystem.IsMacOS() ? AcceleratorPlanner.Plan(DeviceProbe.Probe()) : null);

            // Two ways to turn the segmentation model's per-window numbering into people:
            // compare what the voices sound like, or follow them through the audio consecutive
            // windows share. The second holds up on recordings too poor for the first.
            turns = tracking
                ? diarizer.DiarizeByTracking(audio, maxSpeakers, maxSpeakers, distance)
                : diarizer.Diarize(audio, distance, maxSpeakers);

            found = diarizer.LastSpansFound;
            kept = diarizer.LastSpansKept;
            nested = diarizer.LastNestedDropped;
            contested = diarizer.LastOverlaps;
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

        Heading("Speech spans");
        Console.WriteLine($"  Found        {found} across the overlapping windows");
        Console.WriteLine($"  Kept         {kept} after near-duplicates were dropped");
        Console.WriteLine($"  Nested       {nested} dropped though far shorter than what covered them");

        // Where the model heard two voices at once. This is what the app's crosstalk badge is
        // built from, so a recording full of interruptions should show plenty here — and one
        // that shows none has found a bug, not a polite conversation.
        Heading("Crosstalk");

        if (contested.Count == 0)
        {
            Console.WriteLine("  (none heard)");
        }
        else
        {
            foreach (var (start, end) in contested.Where(c => c.End - c.Start >= 0.5))
            {
                Console.WriteLine($"  {start,7:F2} - {end,7:F2}  ({end - start,5:F2}s)");
            }

            Console.WriteLine(
                $"  {contested.Count} stretch(es), {contested.Sum(c => c.End - c.Start):F1}s in all "
                + "(listed where at least half a second)");
        }

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

        using var diarizer = SpeakerDiarizer.Load(
            modelDirectory,
            OperatingSystem.IsMacOS() ? AcceleratorPlanner.Plan(DeviceProbe.Probe()) : null);

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
