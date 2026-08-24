using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Diarization;

/// <summary>
/// Gives the words that finish a sentence back to whoever started it, across segments.
/// <para>
/// A new speaker does not finish the last speaker's sentence for them. People do complete each
/// other's thoughts, but far less often than a turn boundary lands a word or two out — and the
/// two mistakes do not cost the same. One sentence under two names reads as the transcript being
/// broken; a missed one-word interjection reads as nothing at all.
/// </para>
/// <para>
/// The rule inside a segment is not enough, because the split usually is not inside one. Two
/// examples from the same recording, both across a segment boundary and neither caught by it:
/// "…you're kind of expanding the meaning of" followed by "God. No, I'm not…", where only the
/// first word belongs to the previous speaker; and "…one of the defining" followed by
/// "characteristics", where all of it does.
/// </para>
/// <para>
/// The second is why closing the sentence cannot be required. "characteristics" does not end
/// anything — the sentence runs on past it — and it is still plainly the previous speaker's
/// word. What both have in common is only that a sentence was left open and the next words
/// finish the thought.
/// </para>
/// </summary>
public static class UnfinishedSentences
{
    /// <summary>
    /// How many words may be handed back when they close the sentence. Long enough for the end
    /// of a clause, short enough that a real turn beginning mid-sentence is left where it is.
    /// </summary>
    public const int MostWords = 4;

    /// <summary>
    /// How many may be handed back when they do not.
    /// <para>
    /// One, and the difference is not fussiness. Words that close the sentence they were handed
    /// are self-evidently its end — "God." after "…the meaning of" can be nothing else. Words
    /// that leave it open are only a fragment if there are very few of them: a lone word that
    /// neither finishes a sentence nor fills a segment is not a turn, while three such words
    /// might be somebody actually saying something.
    /// </para>
    /// </summary>
    public const int MostWordsLeavingItOpen = 1;

    /// <summary>A pause long enough that the second speaker was starting, not finishing.</summary>
    public const double LongestPauseSeconds = 1.5;

    /// <summary>Moves trailing fragments back, leaving everything else alone.</summary>
    public static IReadOnlyList<TimedSegment> Apply(IReadOnlyList<TimedSegment> timed)
    {
        ArgumentNullException.ThrowIfNull(timed);

        if (timed.Count < 2)
        {
            return timed;
        }

        var result = new List<TimedSegment>(timed);

        for (var i = 1; i < result.Count; i++)
        {
            var before = result[i - 1];
            var after = result[i];

            if (!Worth(before, after))
            {
                continue;
            }

            // Forward first. A segment ending with a finished sentence and then a short dangling
            // run is a sentence being started, not one being finished, and the run belongs to
            // whoever says the rest of it: "…my opening statement. Then how" / "do we specify
            // what we're arguing about?"
            if (Opening(before.Words) is { } opening
                && Continues(after.Words[0].Text)
                && MoveForward(before, after, opening) is { } handed)
            {
                result[i - 1] = handed.Before;
                result[i] = handed.After;
                continue;
            }

            var head = Head(after.Words);
            var closes = head <= after.Words.Count && !Unfinished(after.Words[head - 1].Text);

            if (head == 0 || head > (closes ? MostWords : MostWordsLeavingItOpen))
            {
                continue;
            }

            var moved = Move(before, after, head);

            if (moved is not { } pair)
            {
                continue;
            }

            result[i - 1] = pair.Before;

            if (pair.After is { } rest)
            {
                result[i] = rest;
            }
            else
            {
                // The whole of it went back, so there is nothing left to be a segment.
                result.RemoveAt(i);
                i--;
            }
        }

        return result;
    }

    /// <summary>True when the second segment might be finishing the first one's sentence.</summary>
    private static bool Worth(TimedSegment before, TimedSegment after) =>
        before.Words.Count > 0
        && after.Words.Count > 0
        && before.Segment.Speaker is { Length: > 0 }
        && after.Segment.Speaker is { Length: > 0 }
        && before.Segment.Speaker != after.Segment.Speaker
        && after.Segment.StartSeconds - before.Segment.EndSeconds <= LongestPauseSeconds
        && Unfinished(before.Words[^1].Text);

    /// <summary>
    /// How many words of the second segment finish the sentence: up to and including the first
    /// one that ends it, or all of them when none does.
    /// </summary>
    private static int Head(IReadOnlyList<WordTimings.Word> words)
    {
        for (var i = 0; i < words.Count; i++)
        {
            if (!Unfinished(words[i].Text))
            {
                return i + 1;
            }
        }

        return words.Count;
    }

    /// <summary>Rebuilds both segments with the first <paramref name="head"/> words moved back.</summary>
    private static (TimedSegment Before, TimedSegment? After)? Move(
        TimedSegment before,
        TimedSegment after,
        int head)
    {
        var text = after.Segment.Text;
        var cut = head < after.Words.Count ? after.Words[head].Offset : text.Length;

        if (cut > text.Length)
        {
            return null;
        }

        var taken = text[..cut].Trim();
        var left = text[cut..].Trim();

        if (taken.Length == 0)
        {
            return null;
        }

        var joined = $"{before.Segment.Text} {taken}";
        var shift = before.Segment.Text.Length + 1;

        var words = new List<WordTimings.Word>(before.Words.Count + head);
        words.AddRange(before.Words);

        for (var i = 0; i < head; i++)
        {
            words.Add(new WordTimings.Word(
                after.Words[i].Text, after.Words[i].StartSeconds, after.Words[i].EndSeconds)
            {
                Offset = after.Words[i].Offset + shift,
            });
        }

        var grown = new TimedSegment(
            before.Segment with
            {
                Text = joined,
                EndSeconds = Math.Max(before.Segment.EndSeconds, after.Words[head - 1].EndSeconds),
            },
            words);

        if (left.Length == 0 || head >= after.Words.Count)
        {
            return (grown, null);
        }

        var rest = new List<WordTimings.Word>(after.Words.Count - head);

        for (var i = head; i < after.Words.Count; i++)
        {
            rest.Add(new WordTimings.Word(
                after.Words[i].Text, after.Words[i].StartSeconds, after.Words[i].EndSeconds)
            {
                Offset = after.Words[i].Offset - cut,
            });
        }

        var remainder = new TimedSegment(
            after.Segment with
            {
                Text = left,
                StartSeconds = after.Words[head].StartSeconds,
            },
            rest);

        return (grown, remainder);
    }

    /// <summary>
    /// How many words at the end of a segment are the start of a new sentence, or null when it
    /// does not end with one.
    /// <para>
    /// A run after the last full stop, with a completed sentence in front of it. Without that
    /// completed sentence the whole segment is one unfinished thought and its end is its own —
    /// which is the case the backward rule handles, and the two must not both fire.
    /// </para>
    /// </summary>
    private static int? Opening(IReadOnlyList<WordTimings.Word> words)
    {
        var ended = -1;

        for (var i = words.Count - 1; i >= 0; i--)
        {
            if (!Unfinished(words[i].Text))
            {
                ended = i;
                break;
            }
        }

        var trailing = words.Count - 1 - ended;

        return ended >= 0 && trailing is > 0 && trailing <= MostWords ? words.Count - trailing : null;
    }

    /// <summary>True when this word reads as the middle of a sentence rather than the start.</summary>
    private static bool Continues(string word)
    {
        foreach (var character in word)
        {
            if (char.IsLetter(character))
            {
                return char.IsLower(character);
            }
        }

        return false;
    }

    /// <summary>Rebuilds both segments with the trailing words moved on to the second.</summary>
    private static (TimedSegment Before, TimedSegment After)? MoveForward(
        TimedSegment before,
        TimedSegment after,
        int from)
    {
        var cut = before.Words[from].Offset;

        if (cut > before.Segment.Text.Length)
        {
            return null;
        }

        var kept = before.Segment.Text[..cut].Trim();
        var given = before.Segment.Text[cut..].Trim();

        if (kept.Length == 0 || given.Length == 0)
        {
            return null;
        }

        var joined = $"{given} {after.Segment.Text}";
        var shift = given.Length + 1;

        var words = new List<WordTimings.Word>(after.Words.Count + before.Words.Count - from);

        for (var i = from; i < before.Words.Count; i++)
        {
            words.Add(new WordTimings.Word(
                before.Words[i].Text, before.Words[i].StartSeconds, before.Words[i].EndSeconds)
            {
                Offset = before.Words[i].Offset - cut,
            });
        }

        foreach (var word in after.Words)
        {
            words.Add(new WordTimings.Word(word.Text, word.StartSeconds, word.EndSeconds)
            {
                Offset = word.Offset + shift,
            });
        }

        var shrunk = new TimedSegment(
            before.Segment with { Text = kept, EndSeconds = before.Words[from - 1].EndSeconds },
            [.. before.Words.Take(from)]);

        var grown = new TimedSegment(
            after.Segment with { Text = joined, StartSeconds = before.Words[from].StartSeconds },
            words);

        return (shrunk, grown);
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
}
