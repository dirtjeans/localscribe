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
    /// <summary>
    /// Nudges each turn boundary onto the nearby gap between two segments.
    /// <para>
    /// The two models disagree about when the speaker changed, and near a real change the
    /// transcriber is usually the better witness: it heard a sentence finish and another begin,
    /// while the diarizer is working from a voice that fades in and out over a fraction of a
    /// second. On the debate recording the diarizer put a change at 16.6s and Whisper ended a
    /// segment at 16.0s, where "Not in the least." begins — so the segment straddled the
    /// boundary, was assigned by which side held more of it, and the interruption was credited
    /// to the person being interrupted.
    /// </para>
    /// <para>
    /// Moving the boundary the last half second fixes it and costs nothing: a boundary that
    /// lands in a gap between segments splits no segment at all, which is the outcome to want
    /// wherever it is available. Boundaries with no segment gap nearby are left exactly where
    /// the diarizer put them.
    /// </para>
    /// </summary>
    internal static IReadOnlyList<SpeakerTurn> SnapToSegmentBoundaries(
        IReadOnlyList<SpeakerTurn> turns,
        IReadOnlyList<TranscriptSegment> segments)
    {
        if (turns.Count < 2 || segments.Count < 2)
        {
            return turns;
        }

        // Where one segment gives way to the next. The transcriber's opinion of where a thought
        // ended, which is the only competing evidence there is.
        var gaps = new List<double>(segments.Count);

        for (var i = 1; i < segments.Count; i++)
        {
            gaps.Add(segments[i].StartSeconds);
        }

        gaps.Sort();

        var moved = turns.ToArray();

        for (var i = 1; i < moved.Length; i++)
        {
            var boundary = moved[i].StartSeconds;

            if (BestStartFor(gaps, boundary) is not { } nearest)
            {
                continue;
            }

            // Both sides move together, or the turns would overlap or leave a hole.
            if (nearest <= moved[i - 1].StartSeconds || nearest >= moved[i].EndSeconds)
            {
                continue;
            }

            moved[i - 1] = moved[i - 1] with { EndSeconds = nearest };
            moved[i] = moved[i] with { StartSeconds = nearest };
        }

        return moved;
    }

    /// <summary>
    /// The segment start a turn boundary should move to, or null when none is close enough.
    /// <para>
    /// Looks backwards much further than forwards, because the disagreement is not symmetric.
    /// The segmentation model decides a voice has changed from a window of audio around the
    /// moment, so it notices a change slightly after it happens; the transcriber marks the
    /// instant a new utterance begins. Taking the nearest boundary in either direction gets this
    /// exactly wrong on an interruption — the far end of the interrupting sentence is nearer
    /// than its start, so the whole interruption is handed to the person being interrupted,
    /// which is what the debate recording did with "Not in the least."
    /// </para>
    /// </summary>
    private static double? BestStartFor(List<double> starts, double boundary)
    {
        double? best = null;

        foreach (var start in starts)
        {
            if (start < boundary - SnapWithinSeconds || start > boundary + SnapForwardSeconds)
            {
                continue;
            }

            // The latest one that qualifies: nearest to the boundary among those it may move to.
            if (best is null || start > best)
            {
                best = start;
            }
        }

        return best;
    }

    /// <summary>
    /// How far a turn boundary may be moved to meet a segment boundary. Long enough to cover the
    /// disagreement between two models watching the same moment, short enough that a boundary
    /// never crosses a whole utterance to reach one.
    /// </summary>
    private const double SnapWithinSeconds = 0.9;

    /// <summary>
    /// How far forward a boundary may move. Barely at all: a segment start after the diarizer's
    /// boundary means the change was noticed before the words began, which is the direction the
    /// smoothing does not produce.
    /// </summary>
    private const double SnapForwardSeconds = 0.2;

    /// <summary>Ends of sentences, kept with the sentence they close.</summary>
    private static readonly Regex SentenceEnd = new(
        @"(?<=[.!?])\s+", RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>
    /// Refuses a speaker change that falls in the middle of a sentence.
    /// <para>
    /// On the debate recording one sentence came out as three people taking turns: "You accept
    /// conscience as a guide," to the first, "and conscience is one of the defining
    /// characteristics" to the second, "of God in the Old Testament." back to the first. Whisper
    /// returned those as three segments, so nothing split them badly — each was labelled whole,
    /// and the labels themselves alternated.
    /// </para>
    /// <para>
    /// Grammar settles it. A sentence that has not ended, continued by a fragment that starts in
    /// lower case, is one person still talking: the second piece is not a reply, it is the rest
    /// of the clause. People do interrupt mid-sentence, but what they say then is a new thought
    /// starting with a capital, not a subordinate clause completing somebody else's.
    /// </para>
    /// <para>
    /// This is also what keeps a genuine quick exchange intact, which a minimum turn length would
    /// not. Two lines down from the sentence above, "I think you're being intellectually
    /// disingenuous." is answered by "In what way?" — a turn of about a second, and the shortest
    /// kind of exchange there is. The full stop and the capital say it is real, and it survives
    /// untouched.
    /// </para>
    /// </summary>
    public static IReadOnlyList<TranscriptSegment> KeepSentencesWhole(
        IReadOnlyList<TranscriptSegment> segments)
    {
        ArgumentNullException.ThrowIfNull(segments);

        if (segments.Count < 2)
        {
            return segments;
        }

        var kept = new List<TranscriptSegment>(segments);

        for (var i = 1; i < kept.Count; i++)
        {
            var previous = kept[i - 1];
            var current = kept[i];

            if (previous.Speaker is null
                || current.Speaker is null
                || previous.Speaker == current.Speaker)
            {
                continue;
            }

            // A pause long enough to lose the thread. Read across a silence, "and…" is as likely
            // to be somebody starting their own sentence with it.
            if (current.StartSeconds - previous.EndSeconds > LongestPause)
            {
                continue;
            }

            if (EndsSentence(previous.Text) || !ContinuesSentence(current.Text))
            {
                continue;
            }

            // Taken from the piece before rather than voted on, so a run of fragments resolves to
            // whoever began the sentence: the third piece then reads the second as already
            // corrected and follows it.
            kept[i] = current with { Speaker = previous.Speaker };
        }

        return kept;
    }

    /// <summary>True when this text finishes what it was saying.</summary>
    private static bool EndsSentence(string text)
    {
        for (var i = text.Length - 1; i >= 0; i--)
        {
            var character = text[i];

            // Trailing quotes and brackets close the sentence rather than being it.
            if (char.IsWhiteSpace(character) || character is '"' or '\'' or ')' or ']' or '”' or '’')
            {
                continue;
            }

            return character is '.' or '!' or '?' or '…' or ':' or ';';
        }

        // Nothing there to be continued.
        return true;
    }

    /// <summary>True when this text reads as the rest of a sentence rather than the start of one.</summary>
    private static bool ContinuesSentence(string text)
    {
        foreach (var character in text)
        {
            if (char.IsLetter(character))
            {
                return char.IsLower(character);
            }

            if (char.IsDigit(character))
            {
                return false;
            }
        }

        return false;
    }

    /// <summary>
    /// How long a gap may be before the two pieces stop reading as one sentence. Generous enough
    /// to cover a breath, short enough that it is not reaching across a turn.
    /// </summary>
    private const double LongestPause = 2;

    /// <summary>
    /// Marks the segments that fall inside a stretch of crosstalk.
    /// <para>
    /// Half of a segment overlapping is enough. A word or two of someone cutting in does not
    /// make the sentence unreliable, but a line spoken mostly over somebody else is one the
    /// transcriber heard through another voice, and a reader deserves to know that before
    /// quoting it.
    /// </para>
    /// </summary>
    public static IReadOnlyList<TranscriptSegment> MarkOverlaps(
        IReadOnlyList<TranscriptSegment> segments,
        IReadOnlyList<(double Start, double End)> overlaps)
    {
        ArgumentNullException.ThrowIfNull(segments);
        ArgumentNullException.ThrowIfNull(overlaps);

        if (overlaps.Count == 0)
        {
            return segments;
        }

        return [.. segments.Select(segment =>
        {
            var length = segment.EndSeconds - segment.StartSeconds;
            if (length <= 0)
            {
                return segment;
            }

            var shared = overlaps.Sum(o =>
                Math.Max(0, Math.Min(o.End, segment.EndSeconds) - Math.Max(o.Start, segment.StartSeconds)));

            return shared >= length * MostlyOverlapped ? segment with { Overlapped = true } : segment;
        })];
    }

    /// <summary>How much of a segment must be crosstalk before the segment is called crosstalk.</summary>
    private const double MostlyOverlapped = 0.5;

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

        var aligned = SnapToSegmentBoundaries(turns, segments);
        var result = new List<TranscriptSegment>(segments.Count);

        foreach (var segment in segments)
        {
            result.AddRange(Split(segment, aligned));
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
