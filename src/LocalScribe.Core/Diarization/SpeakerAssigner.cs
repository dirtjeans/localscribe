using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Diarization;

/// <summary>
/// Attaches speaker labels to transcript segments by matching them up in time.
/// <para>
/// Transcription and diarization run independently and produce boundaries that do not line up,
/// so this reconciles them. It is a deliberately conservative merge: a segment takes the speaker
/// it overlaps most, and says so when that majority is thin. Nothing is invented for a segment
/// that overlaps no turn at all.
/// </para>
/// </summary>
public sealed class SpeakerAssigner
{
    private readonly DiarizationOptions _options;

    public SpeakerAssigner(DiarizationOptions? options = null)
    {
        _options = options ?? DiarizationOptions.Default;
    }

    /// <summary>
    /// Returns the segments with speakers attached. Input is not modified.
    /// </summary>
    public IReadOnlyList<TranscriptSegment> Assign(
        IReadOnlyList<TranscriptSegment> segments,
        IReadOnlyList<SpeakerTurn> turns)
    {
        ArgumentNullException.ThrowIfNull(segments);
        ArgumentNullException.ThrowIfNull(turns);

        if (turns.Count == 0)
        {
            // No diarization result is different from "one speaker throughout". Labelling
            // everything Speaker 1 would be a claim we cannot support.
            return segments;
        }

        return [.. segments.Select(segment => AssignOne(segment, turns))];
    }

    private TranscriptSegment AssignOne(TranscriptSegment segment, IReadOnlyList<SpeakerTurn> turns)
    {
        string? best = null;
        var bestOverlap = 0.0;

        foreach (var turn in turns)
        {
            var overlap = turn.OverlapWith(segment.StartSeconds, segment.EndSeconds);

            // Strictly greater keeps the earliest turn on an exact tie, so a segment split
            // evenly between two speakers resolves the same way every run.
            if (overlap > bestOverlap)
            {
                bestOverlap = overlap;
                best = turn.Speaker;
            }
        }

        if (best is null)
        {
            return segment;
        }

        // A speaker is only chosen when overlap is positive, and positive overlap requires a
        // positive duration, so this division is safe. Capped because a turn extending past
        // the segment on both sides would otherwise exceed 1.
        var fraction = Math.Min(1.0, bestOverlap / segment.DurationSeconds);

        return segment with { Speaker = best, SpeakerOverlapFraction = fraction };
    }

    /// <summary>
    /// Merges adjacent turns from the same speaker separated by only a short gap.
    /// <para>
    /// Segmentation models emit a new turn after any pause, so an uninterrupted speaker arrives
    /// as a stream of fragments. Left alone those produce a transcript that changes speaker
    /// every few seconds between the same two people.
    /// </para>
    /// </summary>
    public IReadOnlyList<SpeakerTurn> Consolidate(IReadOnlyList<SpeakerTurn> turns)
    {
        ArgumentNullException.ThrowIfNull(turns);

        if (turns.Count == 0)
        {
            return turns;
        }

        var ordered = turns.OrderBy(t => t.StartSeconds).ThenBy(t => t.EndSeconds).ToList();
        var merged = new List<SpeakerTurn>();
        var current = ordered[0];

        foreach (var next in ordered.Skip(1))
        {
            var gap = next.StartSeconds - current.EndSeconds;

            if (next.Speaker == current.Speaker && gap <= _options.MinimumGapSeconds)
            {
                current = current with { EndSeconds = Math.Max(current.EndSeconds, next.EndSeconds) };
                continue;
            }

            merged.Add(current);
            current = next;
        }

        merged.Add(current);

        // Dropping short turns after merging rather than before: a fragment that is too brief on
        // its own may be part of a longer stretch by the same speaker once joined up.
        return [.. merged.Where(t => t.DurationSeconds >= _options.MinimumTurnSeconds)];
    }

    /// <summary>
    /// Renders a transcript as dialogue, starting a new block whenever the speaker changes.
    /// Segments with no speaker are attributed to whoever was last speaking, since an
    /// unattributed gap mid-sentence is almost always a boundary artefact rather than a third
    /// person.
    /// </summary>
    public static string FormatAsDialogue(IReadOnlyList<TranscriptSegment> segments)
    {
        ArgumentNullException.ThrowIfNull(segments);

        var builder = new System.Text.StringBuilder();
        string? currentSpeaker = null;
        var lineHasText = false;

        foreach (var segment in segments)
        {
            var text = segment.Text.Trim();
            if (text.Length == 0)
            {
                continue;
            }

            var speaker = segment.Speaker ?? currentSpeaker;

            if (speaker != currentSpeaker)
            {
                if (lineHasText)
                {
                    builder.AppendLine().AppendLine();
                }

                builder.Append(speaker ?? "Unknown").Append(": ");
                currentSpeaker = speaker;
                lineHasText = false;
            }
            else if (lineHasText)
            {
                builder.Append(' ');
            }

            builder.Append(text);
            lineHasText = true;
        }

        return builder.ToString();
    }
}
