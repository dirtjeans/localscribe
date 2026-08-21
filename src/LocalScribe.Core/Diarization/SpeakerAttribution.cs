using System.Text.RegularExpressions;
using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Diarization;

/// <summary>
/// Joins speaker turns to transcript segments.
/// <para>
/// The two are found independently and do not line up. Whisper segments on its own sense of
/// where a thought ends, and over a quick exchange it will happily return "Did you get the
/// report? Yes, this morning. And the numbers? They look fine." as one segment — four turns and
/// two people inside a single timestamped span. Attributing whole segments then hands all of it
/// to whoever spoke most, and a perfectly diarized recording still reads as one person talking
/// to themselves.
/// </para>
/// <para>
/// So a segment covering more than one turn is split. Sentences are the only boundary available
/// — Whisper timestamps segments, not words — and they are a good one, because a speaker change
/// almost always happens at the end of a sentence. Each sentence is placed in time by its share
/// of the segment's characters, which assumes an even speaking rate; that is wrong in detail and
/// close enough to pick the right turn.
/// </para>
/// </summary>
public static class SpeakerAttribution
{
    /// <summary>Ends of sentences, kept with the sentence they close.</summary>
    private static readonly Regex SentenceEnd = new(
        @"(?<=[.!?])\s+", RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>Labels segments by speaker, splitting any that span more than one turn.</summary>
    public static IReadOnlyList<TranscriptSegment> Apply(
        IReadOnlyList<TranscriptSegment> segments,
        IReadOnlyList<SpeakerTurn> turns)
    {
        ArgumentNullException.ThrowIfNull(segments);
        ArgumentNullException.ThrowIfNull(turns);

        if (turns.Count == 0)
        {
            return segments;
        }

        var result = new List<TranscriptSegment>(segments.Count);

        foreach (var segment in segments)
        {
            result.AddRange(Split(segment, turns));
        }

        return result;
    }

    private static IEnumerable<TranscriptSegment> Split(
        TranscriptSegment segment,
        IReadOnlyList<SpeakerTurn> turns)
    {
        var overlapping = turns
            .Where(t => t.OverlapWith(segment.StartSeconds, segment.EndSeconds) > 0)
            .OrderBy(t => t.StartSeconds)
            .ToList();

        if (overlapping.Count == 0)
        {
            return [segment];
        }

        if (overlapping.Count == 1)
        {
            return [segment with { Speaker = overlapping[0].Label }];
        }

        var sentences = SentenceEnd.Split(segment.Text.Trim())
            .Where(s => s.Trim().Length > 0)
            .ToList();

        // Nothing to cut along. Better one span attributed to whoever spoke most of it than a
        // sentence chopped in half at an arbitrary character.
        if (sentences.Count < 2)
        {
            return [segment with { Speaker = Dominant(overlapping, segment).Label }];
        }

        return Group(segment, sentences, overlapping);
    }

    /// <summary>
    /// Places each sentence in time, labels it, then rejoins consecutive sentences that came out
    /// with the same speaker so the transcript does not fragment into one line per sentence.
    /// </summary>
    private static List<TranscriptSegment> Group(
        TranscriptSegment segment,
        List<string> sentences,
        List<SpeakerTurn> turns)
    {
        var totalCharacters = sentences.Sum(s => s.Length);
        var duration = segment.EndSeconds - segment.StartSeconds;

        var pieces = new List<TranscriptSegment>();
        var consumed = 0;

        foreach (var sentence in sentences)
        {
            var start = segment.StartSeconds + (duration * consumed / totalCharacters);
            consumed += sentence.Length;
            var end = segment.StartSeconds + (duration * consumed / totalCharacters);

            var speaker = turns
                .OrderByDescending(t => t.OverlapWith(start, end))
                .First()
                .Label;

            if (pieces.Count > 0 && pieces[^1].Speaker == speaker)
            {
                pieces[^1] = pieces[^1] with
                {
                    Text = $"{pieces[^1].Text} {sentence.Trim()}",
                    EndSeconds = end,
                };

                continue;
            }

            pieces.Add(segment with
            {
                Text = sentence.Trim(),
                StartSeconds = start,
                EndSeconds = end,
                Speaker = speaker,
            });
        }

        return pieces;
    }

    private static SpeakerTurn Dominant(List<SpeakerTurn> turns, TranscriptSegment segment) =>
        turns.OrderByDescending(t => t.OverlapWith(segment.StartSeconds, segment.EndSeconds)).First();
}
