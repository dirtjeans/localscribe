using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Diarization;

/// <summary>
/// Marks the stretches where two people were provably talking at once.
/// <para>
/// The evidence is the measured word times, not the diarizer. When crosstalk is transcribed at
/// all, both streams end up in the text, and the global alignment puts each word on the sound
/// it came from — so two different speakers' words sounding at the same moments is the aligner
/// testifying that the moments were contested. The diarizer's tuning is not consulted and not
/// touched; whatever labels it chose, the mark says those labels were earned under the worst
/// conditions the recording has.
/// </para>
/// <para>
/// The mark exists for the reader, not the pipeline. A name on a line implies more certainty
/// than two-voices-at-once supports, and a reader told "people talking over each other" forgives
/// the transcript for being unsure in exactly the places where being sure is not possible.
/// </para>
/// </summary>
public static class CrosstalkMarks
{
    /// <summary>
    /// How much simultaneous speech makes crosstalk worth saying. Words at a clean turn
    /// boundary brush against each other for a couple of tenths of a second — the aligner
    /// works on twenty-millisecond frames and a handover is not a conversation. Three
    /// quarters of a second of two voices is somebody genuinely being talked over.
    /// </summary>
    public const double NoticeableSeconds = 0.75;

    /// <summary>
    /// How many segments apart two voices can be and still be compared. Crosstalk is local —
    /// an interjection lands in the middle of the sentence it interrupts, not a page later.
    /// </summary>
    public const int Reach = 3;

    /// <summary>Flags both sides of every contested stretch, changing nothing else.</summary>
    public static IReadOnlyList<TimedSegment> Apply(IReadOnlyList<TimedSegment> timed)
    {
        ArgumentNullException.ThrowIfNull(timed);

        var overlapped = new bool[timed.Count];
        var coverage = new List<(double From, double To)>[timed.Count];

        for (var i = 0; i < timed.Count; i++)
        {
            coverage[i] = Coverage(timed[i].Words);
        }

        for (var i = 0; i < timed.Count; i++)
        {
            for (var j = i + 1; j <= Math.Min(timed.Count - 1, i + Reach); j++)
            {
                if (timed[i].Segment.Speaker is not { Length: > 0 } first
                    || timed[j].Segment.Speaker is not { Length: > 0 } second
                    || first == second)
                {
                    continue;
                }

                if (Shared(coverage[i], coverage[j]) >= NoticeableSeconds)
                {
                    overlapped[i] = true;
                    overlapped[j] = true;
                }
            }
        }

        return [.. timed.Select((item, index) => overlapped[index]
            ? item with { Segment = item.Segment with { Overlapped = true } }
            : item)];
    }

    /// <summary>
    /// When this segment's voice was actually sounding: its word spans, merged. The envelope
    /// from first word to last would count the silences a speaker leaves while the other talks,
    /// and turn taking turns into crosstalk.
    /// </summary>
    private static List<(double From, double To)> Coverage(IReadOnlyList<WordTimings.Word> words)
    {
        var merged = new List<(double From, double To)>();

        foreach (var word in words
            .Where(w => w.EndSeconds > w.StartSeconds)
            .OrderBy(w => w.StartSeconds))
        {
            if (merged.Count > 0 && word.StartSeconds <= merged[^1].To)
            {
                merged[^1] = (merged[^1].From, Math.Max(merged[^1].To, word.EndSeconds));
            }
            else
            {
                merged.Add((word.StartSeconds, word.EndSeconds));
            }
        }

        return merged;
    }

    /// <summary>Seconds during which both coverages are sounding at once.</summary>
    private static double Shared(
        List<(double From, double To)> first, List<(double From, double To)> second)
    {
        var total = 0.0;
        var i = 0;
        var j = 0;

        while (i < first.Count && j < second.Count)
        {
            total += Math.Max(
                0, Math.Min(first[i].To, second[j].To) - Math.Max(first[i].From, second[j].From));

            if (first[i].To < second[j].To)
            {
                i++;
            }
            else
            {
                j++;
            }
        }

        return total;
    }
}
