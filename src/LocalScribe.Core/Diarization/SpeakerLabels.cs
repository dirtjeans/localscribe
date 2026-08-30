using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Diarization;

/// <summary>
/// Renumbers automatic speaker labels so the first voice heard is "Speaker 1".
/// <para>
/// The clustering diarizer numbers speakers by cluster, and cluster order has nothing to do
/// with speaking order — so a transcript could open with "Speaker 2", which reads as a bug
/// even though the separation is right. The labels are presentation, not tuning: renumbering
/// touches no boundary and no assignment, only what the people are called.
/// </para>
/// <para>
/// Only untouched automatic labels ("Speaker N") are renumbered, and only when every label
/// still is one — after a user has renamed anybody, the remaining numbers are theirs.
/// </para>
/// </summary>
public static class SpeakerLabels
{
    public static IReadOnlyList<TranscriptSegment> RenumberByAppearance(
        IReadOnlyList<TranscriptSegment> segments)
    {
        ArgumentNullException.ThrowIfNull(segments);

        var mapping = new Dictionary<string, string>(StringComparer.Ordinal);
        var next = 1;
        var changed = false;

        foreach (var segment in segments)
        {
            if (segment.Speaker is not { Length: > 0 } label || mapping.ContainsKey(label))
            {
                continue;
            }

            if (!IsAutomatic(label))
            {
                return segments;
            }

            var renumbered = $"Speaker {next++}";
            mapping[label] = renumbered;
            changed |= renumbered != label;
        }

        if (!changed)
        {
            return segments;
        }

        return [.. segments.Select(segment =>
            segment.Speaker is { } label && mapping.TryGetValue(label, out var renumbered)
                ? segment with { Speaker = renumbered }
                : segment)];
    }

    private static bool IsAutomatic(string label) =>
        label.StartsWith("Speaker ", StringComparison.Ordinal)
        && int.TryParse(label["Speaker ".Length..], out _);
}
