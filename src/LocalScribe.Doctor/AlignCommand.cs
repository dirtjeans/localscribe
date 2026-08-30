using LocalScribe.Core.Audio;
using System.Diagnostics;
using LocalScribe.Core.Hardware;
using LocalScribe.Onnx;

namespace LocalScribe.Doctor;

/// <summary>
/// Scans a recording with the alignment model and reports what the grid came out as.
/// <para>
/// The frame arithmetic in <see cref="ForcedAligner.Scan"/> is the one part of word timing that
/// cannot be tested without the model: how much audio a frame covers, whether the windows tile
/// the recording without gap or overlap, and whether a frame index means the moment it claims.
/// Getting that wrong moves every word by a fixed amount, which is the sort of thing a reader
/// notices and a test suite does not.
/// </para>
/// <para>
/// Decoding a stretch back to letters is the check that matters. If the letters read like what
/// was said at those timestamps, the grid is right and so is its mapping onto the clock.
/// </para>
/// </summary>
public static class AlignCommand
{
    public static int Run(string audioPath, string modelDirectory, ExecutionPlan plan, string? window)
    {
        if (!File.Exists(audioPath))
        {
            Console.Error.WriteLine($"No such file: {audioPath}");
            return 1;
        }

        if (ForcedAligner.Find(modelDirectory) is not { } directory)
        {
            Console.Error.WriteLine(
                $"No alignment model under {modelDirectory}. Word times will be estimated.");
            return 1;
        }

        Heading("Align");
        Console.WriteLine($"  Audio      {audioPath}");
        Console.WriteLine($"  Model      {directory}");

        var audio = WavReader.Read(audioPath);
        Console.WriteLine($"  Length     {audio.DurationSeconds:F1}s at {audio.SampleRate} Hz");

        using var aligner = ForcedAligner.Load(directory, plan);

        var clock = Stopwatch.StartNew();
        var scores = aligner.Scan(audio, new Progress<double>(Report));
        clock.Stop();

        Console.WriteLine();

        if (scores is null)
        {
            Console.Error.WriteLine("  The recording could not be scanned.");
            return 1;
        }

        Heading("Grid");
        Console.WriteLine($"  Frames     {scores.Frames}");
        Console.WriteLine($"  Per frame  {scores.FrameSeconds * 1000:F2} ms");
        Console.WriteLine($"  Alphabet   {scores.Alphabet} tokens");
        Console.WriteLine($"  Covers     {scores.SecondsAt(scores.Frames):F1}s of {audio.DurationSeconds:F1}s");
        Console.WriteLine($"  Held       {scores.Frames * (long)scores.Alphabet * 4 / 1024.0 / 1024:F1} MB");
        Console.WriteLine($"  Took       {clock.Elapsed.TotalSeconds:F1}s");

        // A gap between the recording's length and what the grid covers means windows are not
        // tiling it, which is the failure that would put every later word early.
        var missing = audio.DurationSeconds - scores.SecondsAt(scores.Frames);
        if (missing > 1)
        {
            Console.WriteLine($"  Warning    {missing:F1}s of the recording is not in the grid.");
        }

        Heading("Read back");

        foreach (var (from, to) in Windows(window, audio.DurationSeconds))
        {
            Console.WriteLine($"  {from,7:F1} – {to,-7:F1} {aligner.Read(scores, from, to)}");
        }

        return 0;
    }

    /// <summary>The stretches to decode: whatever was asked for, or a spread through the file.</summary>
    private static IEnumerable<(double From, double To)> Windows(string? asked, double duration)
    {
        if (asked is not null
            && asked.Split('-', 2) is [var start, var end]
            && double.TryParse(start, out var from)
            && double.TryParse(end, out var to))
        {
            yield return (from, to);
            yield break;
        }

        foreach (var share in new[] { 0.1, 0.3, 0.5, 0.7, 0.9 })
        {
            var at = duration * share;
            yield return (at, Math.Min(duration, at + 6));
        }
    }

    private static void Report(double fraction) =>
        Console.Write($"\r  Scanning   {(int)(fraction * 100)}%   ");

    private static void Heading(string title)
    {
        Console.WriteLine();
        Console.WriteLine(title);
        Console.WriteLine(new string('-', title.Length));
    }
}
