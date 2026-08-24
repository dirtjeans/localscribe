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

    /// <summary>
    /// How much of a word's span to believe when deciding whose it is.
    /// <para>
    /// Long enough for any word anybody says. A measured span longer than this is a word that
    /// has swallowed what came before it, and the end is the part that is really the word.
    /// </para>
    /// </summary>
    public const double LongestWordSeconds = 1.5;

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

        // In the order they were said. Segments are aligned one at a time and may move, so two
        // that were adjacent need not come back adjacent — and a transcript whose timestamps go
        // backwards reads as though lines have gone missing, because the eye stops following it.
        result.Sort((left, right) => left.Segment.StartSeconds.CompareTo(right.Segment.StartSeconds));

        // Last, and across segments rather than inside one. Where a sentence was left open, the
        // words that finish it belong to whoever opened it — and the split is usually at a
        // segment boundary, which is exactly where the rule inside a segment cannot see.
        return UnfinishedSentences.Apply(result);
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

        var pieces = Pieces(item, labels);

        // Every word of the parent has to end up in exactly one piece. Cutting is done by slicing
        // text between offsets, so a piece that comes out empty or a boundary that does not reach
        // the end takes words out of the transcript entirely — and a reader has no way to tell
        // that from the recogniser never having heard them. Where the arithmetic does not add up,
        // the segment is left whole and merely labelled: worse attribution, but nothing lost.
        return pieces.Sum(piece => piece.Words.Count) == words.Count
            ? pieces
            : [item with { Segment = item.Segment with { Speaker = labels[0] } }];
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

            // Judged on the end of the word rather than all of it. A word is not five seconds
            // long: where one is, it has absorbed the silence or the speech in front of it,
            // and asking which turn covers the most of that hands the word to whoever was
            // talking before it started.
            var from = Math.Max(words[i].StartSeconds, words[i].EndSeconds - LongestWordSeconds);

            labels[i] = turns
                .OrderByDescending(t => t.OverlapWith(from, words[i].EndSeconds))
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
        KeepTails(labels, words);

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

        // The ends have only one neighbour to be wrong against, so the rule above cannot see
        // them. A segment opening with one word in another voice is the same fault: "All" went
        // to one speaker and "right, so that is interesting…" to the other.
        if (labels.Length > 1 && labels[0] != labels[1])
        {
            labels[0] = labels[1];
        }

    }

    /// <summary>
    /// Hands back the end of a sentence to whoever began it.
    /// <para>
    /// A new speaker does not finish the last speaker's sentence for them. It happens — people
    /// complete each other's thoughts — but far less often than a turn boundary lands a word or
    /// two early, and the cost of the two mistakes is not equal: a sentence split across two
    /// names reads as a transcription error, while a missed interjection reads as nothing at all.
    /// </para>
    /// <para>
    /// Only a short tail that finishes the sentence it is the end of. A long stretch beginning
    /// mid-clause is much more likely to be a real turn; so is a short one that does not close
    /// the sentence, since a sentence still running afterwards was never being finished.
    /// </para>
    /// </summary>
    private static void KeepTails(string?[] labels, IReadOnlyList<WordTimings.Word> words)
    {
        var start = LastRun(labels);

        if (start <= 0 || labels.Length - start > TailWords)
        {
            return;
        }

        // Two conditions, and both matter. The sentence has to have still been running when the
        // speaker supposedly changed, and this has to be the end of it rather than merely a
        // short unpunctuated run — otherwise every segment whose speaker changes near the end
        // with no full stop nearby gets merged, which is most of them.
        if (!Unfinished(words[start - 1].Text) || Unfinished(words[^1].Text))
        {
            return;
        }

        for (var i = start; i < labels.Length; i++)
        {
            labels[i] = labels[start - 1];
        }
    }

    /// <summary>Where the final run of one speaker begins.</summary>
    private static int LastRun(string?[] labels)
    {
        var start = labels.Length - 1;

        while (start > 0 && labels[start - 1] == labels[start])
        {
            start--;
        }

        return start;
    }

    /// <summary>True when this word leaves its sentence open.</summary>
    private static bool Unfinished(string word)
    {
        for (var i = word.Length - 1; i >= 0; i--)
        {
            if (char.IsLetterOrDigit(word[i]))
            {
                return true;
            }

            if (word[i] is '.' or '!' or '?' or '…' or ':' or ';')
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// How many words a sentence's tail may be before it stops looking like a tail.
    /// </summary>
    private const int TailWords = 4;

    /// <summary>Cuts the segment where the label changes, keeping the original text exactly.</summary>
    private static List<TimedSegment> Pieces(TimedSegment item, string?[] labels)
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

        return pieces;
    }

    private static TimedSegment? Piece(TimedSegment item, string? speaker, int first, int last)
    {
        var words = item.Words;
        var from = words[first].Offset;

        // Up to where the next piece begins, so the pieces tile the text and nothing between two
        // words can fall down the gap. Measuring the last word instead — offset plus the length
        // of its text — trusts that the word's text is exactly what sits at that offset, and one
        // character of disagreement silently truncates the piece.
        var to = last + 1 < words.Count
            ? words[last + 1].Offset
            : item.Segment.Text.Length;

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
