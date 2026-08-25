using LocalScribe.Core.Diarization;

namespace LocalScribe.Doctor;

/// <summary>
/// Switches how the app decides who spoke when.
/// <para>
/// Two methods, neither better everywhere. Following speakers between overlapping windows uses
/// the segmentation model's own opinion that two people are talking at once as a constraint, and
/// on a phone recording of an argument that was the difference between three speakers and
/// nineteen. Clustering stretches of speech by voice is the older path, and on a studio podcast
/// with five voices it found all five and held that answer across four thresholds, where
/// tracking found three and 22 turns in seven minutes.
/// </para>
/// <para>
/// Long uninterrupted runs are what separates them: they give tracking almost no overlaps to
/// constrain anything with, and overlaps are what it depends on. Nothing here can yet look at a
/// recording and say which kind it is, so this is a setting and not a decision.
/// </para>
/// </summary>
public static class DiarizerCommand
{
    public static int Run(string modelRoot, string? requested)
    {
        var directory = Path.Combine(modelRoot, "diarization");

        if (!Directory.Exists(directory))
        {
            Console.Error.WriteLine($"No speaker models under {directory}. Run --fetch-models first.");
            return 1;
        }

        var active = DiarizationChoice.Read(directory);

        if (string.IsNullOrWhiteSpace(requested))
        {
            Report(active);
            return 0;
        }

        var chosen = DiarizationChoice.Parse(requested);

        // Parse falls back rather than failing, which is right when reading a file and wrong when
        // reading a command: someone who typed a name deserves to know it was not one.
        if (!DiarizationChoice.Name(chosen).Equals(requested.Trim(), StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine($"Unknown method '{requested}'. Known: tracking, voices.");
            return 1;
        }

        DiarizationChoice.Write(directory, chosen);

        Console.WriteLine();
        Console.WriteLine($"Now using {DiarizationChoice.Name(chosen)}.");
        Console.WriteLine();
        Console.WriteLine("  Transcribe a recording again to see the difference, or press the speakers");
        Console.WriteLine("  button to work them out again without transcribing.");

        if (chosen == DiarizationMethod.Voices)
        {
            Console.WriteLine();
            Console.WriteLine("  Crosstalk is not marked on this path: overlapping speech is something the");
            Console.WriteLine("  other one notices while following speakers between windows.");
        }

        return 0;
    }

    private static void Report(DiarizationMethod active)
    {
        Console.WriteLine();
        Console.WriteLine("How speakers are worked out");
        Console.WriteLine("---------------------------");

        foreach (var (name, note) in new[]
        {
            ("tracking", "Follows speakers between overlapping windows. Better on crosstalk and argument."),
            ("voices", "Clusters stretches of speech by voice. Better on long uninterrupted runs."),
        })
        {
            var state = name.Equals(DiarizationChoice.Name(active), StringComparison.Ordinal)
                ? "in use"
                : "available";

            Console.WriteLine($"  {name,-10} {state,-11} {note}");
        }

        Console.WriteLine();
        Console.WriteLine("  Switch with --diarizer <name>. Neither is better everywhere, which is why");
        Console.WriteLine("  this is a setting: --diarize <wav> --sweep says which suits a recording.");
    }
}
