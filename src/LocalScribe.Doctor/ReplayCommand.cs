using System.Globalization;
using LocalScribe.Core.Alignment;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Transcription;
using LocalScribe.Onnx;

namespace LocalScribe.Doctor;

/// <summary>
/// Re-runs the aligner on the exact input an app run dumped, against the same audio.
/// <para>
/// The app writes its aligner input — the transcriber's raw stamps — to a file precisely
/// because the checker could not reproduce them: an archive's stored bounds have been through
/// an alignment already, and every divergence between "the checker is clean" and "the app is
/// wrong" has come down to inputs the checker never saw. This closes that gap: the same
/// segments, the same stamps, the same audio, outside the app, where the result can be
/// compared line by line against a fresh scan of the raw samples.
/// </para>
/// </summary>
public static class ReplayCommand
{
    public static int Run(string inputPath, string audioPath, string modelDirectory, ExecutionPlan plan)
    {
        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"No such file: {inputPath}");
            return 1;
        }

        if (!File.Exists(audioPath))
        {
            Console.Error.WriteLine($"No such file: {audioPath}");
            return 1;
        }

        if (ForcedAligner.Find(modelDirectory) is not { } directory)
        {
            Console.Error.WriteLine($"No alignment model under {modelDirectory}.");
            return 1;
        }

        var segments = new List<TranscriptSegment>();

        foreach (var line in File.ReadLines(inputPath))
        {
            if (line.Split('\t', 3) is [var from, var to, var text]
                && double.TryParse(from, CultureInfo.InvariantCulture, out var start)
                && double.TryParse(to, CultureInfo.InvariantCulture, out var end))
            {
                segments.Add(new TranscriptSegment(text, start, end));
            }
        }

        Console.WriteLine($"  Input      {segments.Count} segments from {inputPath}");

        var audio = WavReader.Read(audioPath);
        Console.WriteLine($"  Audio      {audio.DurationSeconds:F1}s at {audio.SampleRate} Hz");

        using var aligner = ForcedAligner.Load(directory, plan);

        var scores = aligner.Scan(audio, new Progress<double>(f =>
            Console.Write($"\r  Scanning   {(int)(f * 100)}%   ")));
        Console.WriteLine();

        if (scores is null)
        {
            Console.Error.WriteLine("  The recording could not be scanned.");
            return 1;
        }

        var all = aligner.AlignAll(scores, segments, CancellationToken.None);

        for (var i = 0; i < segments.Count; i++)
        {
            var stamped = FormattableString.Invariant(
                $"{segments[i].StartSeconds,7:F2}-{segments[i].EndSeconds,-7:F2}");

            var sounded = all[i]?.Where(w => w.EndSeconds > w.StartSeconds).ToList();

            var placed = sounded is { Count: > 0 }
                ? FormattableString.Invariant(
                    $"{sounded[0].StartSeconds,7:F2}-{sounded[^1].EndSeconds,-7:F2}")
                : "      unplaced      ";

            var head = segments[i].Text.Length <= 44
                ? segments[i].Text
                : segments[i].Text[..41] + "…";

            Console.WriteLine($"  stamped {stamped}  placed {placed}  \"{head}\"");
        }

        return 0;
    }
}
