using System.Globalization;
using System.Text;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Alignment;

/// <summary>
/// Reports what aligning did to a transcript's segment boundaries.
/// <para>
/// Measuring this from a saved archive cannot answer the question, because an archive's segments
/// have already been aligned: the input is crowded before the experiment starts, so the result
/// understates how much crowding a run creates and cannot separate the two. The only place the
/// question can be asked properly is inside the run, where the segments going in still carry the
/// transcriber's own times and do not overlap.
/// </para>
/// <para>
/// Drift says nothing about any of this. It was +0.00 in every fifth of two recordings whose
/// transcripts were visibly wrong, because every word can sit at the right moment while the
/// segments holding them lie on top of each other — and everything that walks a transcript in
/// order then breaks: which paragraph is being spoken, which word inside it, and what order two
/// of them go in.
/// </para>
/// </summary>
public static class AlignmentCrowding
{
    /// <summary>Ignore overlaps below this; they are rounding, not stacking.</summary>
    public const double Noticeable = 0.05;

    /// <param name="Segments">How many segments were aligned.</param>
    /// <param name="OverlappedBefore">Boundaries already overlapping when they went in.</param>
    /// <param name="OverlappedAfter">Boundaries overlapping once aligned.</param>
    /// <param name="Swallowed">Segments ending up wholly inside the one before them.</param>
    /// <param name="Worst">The largest overlap, in seconds.</param>
    /// <param name="WorstText">What the worst overlap was on.</param>
    /// <param name="Moved">The furthest a segment's start moved, most first.</param>
    public sealed record Report(
        int Segments,
        int OverlappedBefore,
        int OverlappedAfter,
        int Swallowed,
        double Worst,
        string WorstText,
        IReadOnlyList<(double By, double From, double To, string Text)> Moved);

    /// <summary>Compares the segments going in with the ones coming out.</summary>
    public static Report Describe(
        IReadOnlyList<TranscriptSegment> before,
        IReadOnlyList<TimedSegment> after)
    {
        ArgumentNullException.ThrowIfNull(before);
        ArgumentNullException.ThrowIfNull(after);

        var placed = after.Select(t => t.Segment).ToList();

        var moved = new List<(double By, double From, double To, string Text)>();

        // Only where the same segment can be recognised on both sides. Aligning divides nothing,
        // but the caller may have, and a moved-by figure across two different segments is noise.
        for (var i = 0; i < Math.Min(before.Count, placed.Count); i++)
        {
            if (!before[i].Text.Equals(placed[i].Text, StringComparison.Ordinal))
            {
                continue;
            }

            var by = placed[i].StartSeconds - before[i].StartSeconds;

            if (Math.Abs(by) >= Noticeable)
            {
                moved.Add((by, before[i].StartSeconds, placed[i].StartSeconds, before[i].Text));
            }
        }

        var worst = (Overlap: 0.0, Text: string.Empty);
        var swallowed = 0;

        for (var i = 1; i < placed.Count; i++)
        {
            var overlap = placed[i - 1].EndSeconds - placed[i].StartSeconds;

            if (overlap <= Noticeable)
            {
                continue;
            }

            if (placed[i].EndSeconds <= placed[i - 1].EndSeconds)
            {
                swallowed++;
            }

            if (overlap > worst.Overlap)
            {
                worst = (overlap, placed[i].Text);
            }
        }

        return new Report(
            placed.Count,
            Overlapping(before),
            Overlapping(placed),
            swallowed,
            worst.Overlap,
            worst.Text,
            [.. moved.OrderByDescending(m => Math.Abs(m.By)).Take(12)]);
    }

    private static int Overlapping(IReadOnlyList<TranscriptSegment> segments)
    {
        var count = 0;

        for (var i = 1; i < segments.Count; i++)
        {
            if (segments[i - 1].EndSeconds - segments[i].StartSeconds > Noticeable)
            {
                count++;
            }
        }

        return count;
    }

    /// <summary>The report as text, for writing somewhere a person can read it.</summary>
    public static string Format(Report report, string source)
    {
        ArgumentNullException.ThrowIfNull(report);

        var text = new StringBuilder();
        var culture = CultureInfo.InvariantCulture;

        text.AppendLine(culture, $"Alignment crowding — {source}");
        text.AppendLine(new string('-', 40));
        text.AppendLine(culture, $"Segments            {report.Segments}");
        text.AppendLine(culture, $"Overlapping before  {report.OverlappedBefore}");
        text.AppendLine(culture, $"Overlapping after   {report.OverlappedAfter}");
        text.AppendLine(culture, $"Swallowed whole     {report.Swallowed}");

        if (report.Worst > 0)
        {
            text.AppendLine(culture, $"Worst overlap       {report.Worst:F2}s on \"{Short(report.WorstText)}\"");
        }

        if (report.Moved.Count > 0)
        {
            text.AppendLine();
            text.AppendLine("Segments that moved furthest");

            foreach (var (by, from, to, what) in report.Moved)
            {
                text.AppendLine(culture, $"  {by,+7:+0.00;-0.00}s  {from,7:F2} -> {to,-7:F2} \"{Short(what)}\"");
            }
        }

        return text.ToString();
    }

    private static string Short(string text) =>
        text.Length <= 52 ? text : string.Concat(text.AsSpan(0, 49), "…");
}
