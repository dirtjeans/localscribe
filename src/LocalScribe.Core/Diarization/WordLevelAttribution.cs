using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Diarization;

/// <summary>A segment and the measured time of every word in it.</summary>
public sealed record TimedSegment(TranscriptSegment Segment, IReadOnlyList<WordTimings.Word> Words);

/// <summary>
/// Divides segments between speakers at the word the voice actually changed on.
/// <para>
/// The transcriber and the diarizer disagree about where anything begins, and the transcriber
/// wins by default because its segments are what the reader sees. On the debate recording that
/// cost most of the diarization: of 31 speaker changes found, 26 fell inside a segment rather
/// than at its edge. One nine-second segment held two changes and one eleven-second segment held
/// three, and each was handed whole to whichever voice held most of it.
/// </para>
/// <para>
/// Sentences were the only boundary available when this was first written — Whisper times
/// segments, not words — so a segment could only be cut where a sentence ended, and one that
/// held no sentence end could not be cut at all. Words are measured now, so the cut can go where
/// the voice changed. Swapping the segmentation model and swapping the embedding model both
/// changed nothing measurable, because neither was what was losing the turns.
/// </para>
/// </summary>
public static class WordLevelAttribution
{
    /// <summary>
    /// A turn shorter than this is not cut on.
    /// <para>
    /// The window vote produces flickers a fifth of a second long at the boundary between two
    /// real turns. Cutting on those would divide a sentence into three around a speaker who was
    /// never there.
    /// </para>
    /// </summary>
    public const double ShortestTurnSeconds = 0.5;

    /// <summary>Labels each segment, dividing any that spans more than one speaker.</summary>
    public static IReadOnlyList<TimedSegment> Apply(
        IReadOnlyList<TimedSegment> timed,
        IReadOnlyList<SpeakerTurn> turns)
    {
        ArgumentNullException.ThrowIfNull(timed);
        ArgumentNullException.ThrowIfNull(turns);

        var worth = turns.Where(t => t.EndSeconds - t.StartSeconds >= ShortestTurnSeconds).ToList();

        if (worth.Count == 0)
        {
            return timed;
        }

        var result = new List<TimedSegment>(timed.Count);

        foreach (var item in timed)
        {
            result.AddRange(Divide(item, worth));
        }

        return result;
    }

    private static IEnumerable<TimedSegment> Divide(TimedSegment item, List<SpeakerTurn> turns)
    {
        var words = item.Words;

        if (words.Count == 0)
        {
            return [item with { Segment = item.Segment with { Speaker = Loudest(item.Segment, turns) } }];
        }

        var labels = Label(words, turns);

        if (labels.Distinct().Count() <= 1)
        {
            return [item with { Segment = item.Segment with { Speaker = labels[0] } }];
        }

        return Pieces(item, labels);
    }

    /// <summary>Which speaker each word belongs to, with gaps filled from its neighbours.</summary>
    private static string?[] Label(IReadOnlyList<WordTimings.Word> words, List<SpeakerTurn> turns)
    {
        var labels = new string?[words.Count];

        for (var i = 0; i < words.Count; i++)
        {
            // A word with no duration is punctuation the aligner could not place. It has no
            // sound of its own to attribute, so it takes whatever its neighbours say.
            if (words[i].EndSeconds <= words[i].StartSeconds)
            {
                continue;
            }

            labels[i] = turns
                .OrderByDescending(t => t.OverlapWith(words[i].StartSeconds, words[i].EndSeconds))
                .First()
                .Label;
        }

        // Forwards then backwards, so a word before the first placed one is covered too.
        for (var i = 1; i < labels.Length; i++)
        {
            labels[i] ??= labels[i - 1];
        }

        for (var i = labels.Length - 2; i >= 0; i--)
        {
            labels[i] ??= labels[i + 1];
        }

        Steady(labels);

        return labels;
    }

    /// <summary>
    /// Removes a single word attributed against both its neighbours.
    /// <para>
    /// One word is not a turn. It is the boundary of a real turn landing a word early or late,
    /// and cutting on it produces a one-word interjection nobody said.
    /// </para>
    /// </summary>
    private static void Steady(string?[] labels)
    {
        for (var i = 1; i < labels.Length - 1; i++)
        {
            if (labels[i] != labels[i - 1] && labels[i - 1] == labels[i + 1])
            {
                labels[i] = labels[i - 1];
            }
        }
    }

    /// <summary>Cuts the segment where the label changes, keeping the original text exactly.</summary>
    private static IEnumerable<TimedSegment> Pieces(TimedSegment item, string?[] labels)
    {
        var pieces = new List<TimedSegment>();
        var start = 0;

        for (var i = 1; i <= labels.Length; i++)
        {
            if (i < labels.Length && labels[i] == labels[start])
            {
                continue;
            }

            if (Piece(item, labels[start], start, i - 1) is { } piece)
            {
                pieces.Add(piece);
            }

            start = i;
        }

        return pieces.Count > 0 ? pieces : [item];
    }

    private static TimedSegment? Piece(TimedSegment item, string? speaker, int first, int last)
    {
        var words = item.Words;
        var from = words[first].Offset;
        var to = words[last].Offset + words[last].Text.Length;

        // Sliced out of the original rather than rebuilt from the words, so the punctuation and
        // spacing the reader sees are the ones cleanup produced.
        if (from < 0 || to > item.Segment.Text.Length || to <= from)
        {
            return null;
        }

        var text = item.Segment.Text[from..to].Trim();

        if (text.Length == 0)
        {
            return null;
        }

        // Offsets are into this piece's text now, not its parent's.
        var moved = new List<WordTimings.Word>(last - first + 1);

        for (var i = first; i <= last; i++)
        {
            moved.Add(new WordTimings.Word(words[i].Text, words[i].StartSeconds, words[i].EndSeconds)
            {
                Offset = words[i].Offset - from,
            });
        }

        var begins = words[first].StartSeconds;
        var ends = Math.Max(words[last].EndSeconds, begins);

        return new TimedSegment(
            item.Segment with
            {
                Text = text,
                StartSeconds = begins,
                EndSeconds = ends,
                Speaker = speaker,
            },
            moved);
    }

    private static string? Loudest(TranscriptSegment segment, List<SpeakerTurn> turns) =>
        turns
            .OrderByDescending(t => t.OverlapWith(segment.StartSeconds, segment.EndSeconds))
            .FirstOrDefault()
            ?.Label;
}
