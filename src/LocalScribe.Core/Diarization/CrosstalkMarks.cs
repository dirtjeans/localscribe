using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Diarization;

/// <summary>
/// Marks the lines spoken while somebody else was talking.
/// <para>
/// The evidence is the segmentation model's own overlap classes — the powerset decoder reports
/// frames on which two local speakers are active at once, and those moments arrive here as
/// contested time spans. Nothing else can testify to this: the transcriber usually writes down
/// only the louder stream, so the text reads clean, and the global alignment is a single
/// monotone path that cannot place two words on the same instant even when both streams were
/// transcribed. The first version of this mark used word-time collisions as evidence and could
/// therefore never fire; a debate full of audible crosstalk came back unmarked.
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
    /// How much of a line must fall on contested moments before the line is marked. A clean
    /// handover grazes the overlap classes for tenths of a second; three quarters of a second
    /// of two voices is somebody genuinely being talked over.
    /// </summary>
    public const double NoticeableSeconds = 0.75;

    /// <summary>Flags every line that spent noticeable time contested, changing nothing else.</summary>
    /// <param name="timed">The attributed pieces, words and all.</param>
    /// <param name="contested">Where two voices sounded at once, from the segmentation model.</param>
    public static IReadOnlyList<TimedSegment> Apply(
        IReadOnlyList<TimedSegment> timed,
        IReadOnlyList<(double Start, double End)> contested)
    {
        ArgumentNullException.ThrowIfNull(timed);
        ArgumentNullException.ThrowIfNull(contested);

        if (contested.Count == 0)
        {
            return timed;
        }

        var windows = contested.OrderBy(c => c.Start).ToList();

        return [.. timed.Select(item =>
            Shared(Coverage(item), windows) >= NoticeableSeconds
                ? item with { Segment = item.Segment with { Overlapped = true } }
                : item)];
    }

    /// <summary>
    /// When this line's voice was actually sounding: its measured word spans, merged — or its
    /// stated bounds when no words were measured, which overstates a little and is the honest
    /// fallback for a transcript that was never aligned.
    /// </summary>
    private static List<(double From, double To)> Coverage(TimedSegment item)
    {
        var sounded = item.Words
            .Where(w => w.EndSeconds > w.StartSeconds)
            .OrderBy(w => w.StartSeconds)
            .ToList();

        if (sounded.Count == 0)
        {
            return item.Segment.EndSeconds > item.Segment.StartSeconds
                ? [(item.Segment.StartSeconds, item.Segment.EndSeconds)]
                : [];
        }

        var merged = new List<(double From, double To)>();

        foreach (var word in sounded)
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

    /// <summary>Seconds during which the line's voice and a contested span coincide.</summary>
    private static double Shared(
        List<(double From, double To)> voice, List<(double Start, double End)> contested)
    {
        var total = 0.0;
        var i = 0;
        var j = 0;

        while (i < voice.Count && j < contested.Count)
        {
            total += Math.Max(
                0, Math.Min(voice[i].To, contested[j].End) - Math.Max(voice[i].From, contested[j].Start));

            if (voice[i].To < contested[j].End)
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
