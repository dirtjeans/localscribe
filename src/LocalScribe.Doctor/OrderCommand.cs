using LocalScribe.Core.Audio;

namespace LocalScribe.Doctor;

/// <summary>
/// Implements <c>--order &lt;file&gt; &lt;file&gt; …</c>: what each recording says about itself,
/// and the order the app would propose for transcribing them as one. The instrument for
/// <see cref="RecordingFacts"/> on real files, whose recorders the tests can only imitate.
/// </summary>
internal static class OrderCommand
{
    public static int Run(IReadOnlyList<string> paths)
    {
        if (paths.Count == 0)
        {
            Console.Error.WriteLine("--order needs at least one file.");
            return 1;
        }

        var facts = paths.Select(RecordingFacts.Read).ToList();

        Console.WriteLine();
        foreach (var file in facts)
        {
            Console.WriteLine($"  {file.Name}");
            Console.WriteLine($"      recorded {file.RecordedAt?.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss") ?? "—"}, "
                + $"part {file.Track?.ToString() ?? "—"}, "
                + $"length {(file.DurationSeconds is { } s ? LocalScribe.Core.Pipeline.RunTimes.Clock(s) : "—")}, "
                + $"saved {file.LastSaved?.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss") ?? "—"}");
        }

        var proposal = RecordingOrder.Decide(facts);

        Console.WriteLine();
        Console.WriteLine($"  {proposal.Reason}");

        if (proposal.StampsMarkEnds)
        {
            Console.WriteLine("  (the stamps are read as when each recording stopped; starts are those less each length)");
        }

        for (var i = 0; i < proposal.Ordered.Count; i++)
        {
            var gap = proposal.GapsBefore[i] is { } g ? $"   (gap before: {g:F0} s)" : string.Empty;
            Console.WriteLine($"  {i + 1}. {proposal.Ordered[i].Name}{gap}");
        }

        foreach (var doubt in proposal.Doubts)
        {
            Console.WriteLine($"  ! {doubt}");
        }

        Console.WriteLine($"  Together: {RecordingOrder.CombinedName([.. proposal.Ordered.Select(f => f.Name)])}");
        return 0;
    }
}
