namespace LocalScribe.Core.Transcription;

/// <summary>
/// Joins the per-window results back into one transcript.
/// <para>
/// Because the chunker overlaps its windows, the same words are transcribed twice at every
/// boundary. Dropping duplicates by timestamp alone does not work: the two passes rarely agree
/// on exact timings, and Whisper often shifts a word by a few hundred milliseconds between
/// windows. So this matches on the words themselves and uses timing only to decide where to
/// look for a match.
/// </para>
/// </summary>
public sealed class TranscriptStitcher
{
    private readonly double _boundaryToleranceSeconds;

    /// <param name="boundaryToleranceSeconds">
    /// How far either side of a chunk boundary a duplicate might have drifted. Should be at
    /// least as large as the chunker's overlap.
    /// </param>
    public TranscriptStitcher(double boundaryToleranceSeconds = 2.5)
    {
        _boundaryToleranceSeconds = boundaryToleranceSeconds;
    }

    /// <summary>
    /// Merges segments from overlapping windows into one ordered, duplicate-free transcript.
    /// Segments are expected in chunk order; within a chunk they must already be time-ordered.
    /// </summary>
    public IReadOnlyList<TranscriptSegment> Stitch(IEnumerable<IReadOnlyList<TranscriptSegment>> chunkResults)
    {
        ArgumentNullException.ThrowIfNull(chunkResults);

        var merged = new List<TranscriptSegment>();

        foreach (var chunk in chunkResults)
        {
            foreach (var segment in chunk)
            {
                if (segment.LooksHallucinated || string.IsNullOrWhiteSpace(segment.Text))
                {
                    continue;
                }

                if (IsDuplicateOfRecent(merged, segment))
                {
                    continue;
                }

                // Not a whole duplicate, but its opening words may still repeat the tail of what
                // came before. That is the ordinary shape of an overlap between two windows.
                //
                // Only at a seam, though. Trimming is a repair to a join, and a join is close in
                // time; a phrase genuinely said again ten minutes later is not an artefact and
                // must survive intact.
                var trimmed = merged.Count > 0 && AbutsPrevious(merged[^1], segment)
                    ? segment with { Text = TrimLeadingOverlap(merged[^1].Text, segment.Text) }
                    : segment;

                // What is left begins where the words it repeated ended. Kept where it was, the
                // stamp still claimed those words' seconds — and before them: on the debate a
                // window trimmed of "That's why I'm trying to define it." still started half a
                // second before that sentence did.
                if (trimmed.Text != segment.Text && merged[^1].EndSeconds > trimmed.StartSeconds)
                {
                    trimmed = trimmed with
                    {
                        StartSeconds = merged[^1].EndSeconds,
                        EndSeconds = Math.Max(trimmed.EndSeconds, merged[^1].EndSeconds),
                    };
                }

                // And a repeat wholly inside this segment, which is the other way the same words
                // arrive twice: not two windows transcribing one stretch of audio, but the
                // decoder looping within a single window. Every check above compares the end of
                // one segment against the start of the next, so a sentence emitted twice in the
                // middle of one passed all of them.
                trimmed = trimmed with { Text = RepeatedPhrase.Trim(trimmed.Text) };

                if (trimmed.Text.Trim().Length == 0)
                {
                    continue;
                }

                merged.Add(trimmed);
            }
        }

        // In the order they were written — window by window, and within each in the decoder's
        // order — which is the order they were said. Not re-sorted by stamp: stamps are the
        // weaker evidence, and a sort let one wrong stamp carry a sentence thirty seconds out of
        // place on the debate. Stamps that disagree are pushed into line instead.
        return NoTwoAtOnce(merged);
    }

    /// <summary>
    /// Repairs one window's stamps so they agree with the order the decoder wrote its segments
    /// in, which is the order they were said.
    /// <para>
    /// Whisper sometimes stamps a segment where it cannot be: past the end of the audio the
    /// window held — the timestamps describing its own padding, which is why where the next
    /// window starts already ignores them — or later than the segment it wrote next. The words
    /// are real; only the stamp is wrong. But the stitch sorts by stamp, and on the debate the
    /// window covering 82–112 s wrote "That's why I'm trying to define it." first, stamped it
    /// 112.3 s, and the finished transcript printed it thirty seconds late, after the window's
    /// own next sentence.
    /// </para>
    /// <para>
    /// So a segment that runs past the window is cut at its end, or, begun past it, moved to end
    /// there with the length it was given. Then, last to first, a segment stamped later than the
    /// one after it is moved to start where that one does, keeping its length; the stitch keeps
    /// equal stamps in decoder order and pushes what follows clear of it. Where exactly the
    /// words fall is the aligner's job. What the stamps must get right is the order.
    /// </para>
    /// </summary>
    /// <param name="windowEndSeconds">Where the window's real audio ends, before any padding.</param>
    public static IReadOnlyList<TranscriptSegment> InDecoderOrder(
        IReadOnlyList<TranscriptSegment> segments,
        double windowStartSeconds,
        double windowEndSeconds)
    {
        ArgumentNullException.ThrowIfNull(segments);

        var repaired = segments.Select(segment =>
        {
            if (segment.EndSeconds <= windowEndSeconds)
            {
                return segment;
            }

            // Begun inside the window and running past it: only the end was invented.
            if (segment.StartSeconds < windowEndSeconds)
            {
                return segment with { EndSeconds = windowEndSeconds };
            }

            var length = Math.Max(0, segment.EndSeconds - segment.StartSeconds);
            var start = Math.Max(windowStartSeconds, Math.Min(segment.StartSeconds, windowEndSeconds - length));

            return segment with { StartSeconds = start, EndSeconds = windowEndSeconds };
        }).ToList();

        for (var i = repaired.Count - 2; i >= 0; i--)
        {
            var next = repaired[i + 1].StartSeconds;

            if (repaired[i].StartSeconds > next)
            {
                var length = Math.Max(0, repaired[i].EndSeconds - repaired[i].StartSeconds);
                repaired[i] = repaired[i] with { StartSeconds = next, EndSeconds = next + length };
            }
        }

        return repaired;
    }

    /// <summary>
    /// Pushes each segment to begin no earlier than the one before it ended.
    /// <para>
    /// Trimming the repeated words at a seam leaves the segment's start time where it was, so a
    /// segment whose opening was cut still claims the seconds those words occupied — and two
    /// segments end up covering the same moment. On a two-minute recording one paragraph held
    /// "That's how it's defined in the Old Testament" running to 29.98s and "Elijah and in
    /// Jonah" starting at 27.98s, which is not a thing that can have happened.
    /// </para>
    /// <para>
    /// It matters downstream rather than here. A reader never sees these numbers, but everything
    /// that walks the transcript in time does: word timings inside a paragraph stop running
    /// forwards, so the marker following the audio jumps back and forth between two overlapping
    /// segments instead of moving along the line.
    /// </para>
    /// </summary>
    private static List<TranscriptSegment> NoTwoAtOnce(List<TranscriptSegment> segments)
    {
        for (var i = 1; i < segments.Count; i++)
        {
            var previous = segments[i - 1].EndSeconds;

            if (segments[i].StartSeconds >= previous)
            {
                continue;
            }

            // Carried forward, keeping the length it was given. A segment squeezed past its own
            // end would report a negative duration, and the next one round the loop is pushed
            // clear of this one in turn, so the whole run comes out ordered.
            var length = Math.Max(0, segments[i].EndSeconds - segments[i].StartSeconds);

            segments[i] = segments[i] with
            {
                StartSeconds = previous,
                EndSeconds = Math.Max(segments[i].EndSeconds, previous + length),
            };
        }

        return segments;
    }

    /// <summary>
    /// Looks back over the segments near this one and reports whether the same words already
    /// landed. Only recent segments are considered, so a genuinely repeated phrase later in the
    /// recording survives.
    /// </summary>
    private bool IsDuplicateOfRecent(List<TranscriptSegment> merged, TranscriptSegment candidate)
    {
        var normalisedCandidate = Normalise(candidate.Text);
        if (normalisedCandidate.Length == 0)
        {
            return false;
        }

        for (var i = merged.Count - 1; i >= 0; i--)
        {
            var existing = merged[i];

            // Segments are appended in roughly increasing time order, so once we are well
            // behind the candidate there is nothing left to compare against.
            if (candidate.StartSeconds - existing.EndSeconds > _boundaryToleranceSeconds)
            {
                break;
            }

            if (Normalise(existing.Text) == normalisedCandidate)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>True when two segments are close enough in time to share a window boundary.</summary>
    private bool AbutsPrevious(TranscriptSegment previous, TranscriptSegment candidate) =>
        candidate.StartSeconds - previous.EndSeconds <= _boundaryToleranceSeconds;

    /// <summary>
    /// Removes words at the start of <paramref name="candidate"/> that already appear at the end
    /// of <paramref name="previous"/>.
    /// <para>
    /// Whole-segment equality only catches the case where two passes divided the audio the same
    /// way. Usually they do not: one produces "If the stitching works correctly," and the next
    /// starts "works correctly, this sentence will appear once". Neither contains the other, so
    /// both survive, and the seam reads "works correctly, works correctly," — which is what a
    /// duplicate actually looks like in practice.
    /// </para>
    /// <para>
    /// Compared word by word with punctuation and casing ignored, because the two passes rarely
    /// agree on either. The longest overlap wins, so a repeated single word does not shadow a
    /// repeated phrase.
    /// </para>
    /// </summary>
    /// <param name="maxWords">
    /// How far back to look. Bounded because a long match is more likely to be genuine
    /// repetition in the speech than an artefact of the seam.
    /// </param>
    public static string TrimLeadingOverlap(string previous, string candidate, int maxWords = 40)
    {
        ArgumentNullException.ThrowIfNull(previous);
        ArgumentNullException.ThrowIfNull(candidate);

        var tail = previous.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var head = candidate.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

        if (tail.Length == 0 || head.Length == 0)
        {
            return candidate;
        }

        var limit = Math.Min(maxWords, Math.Min(tail.Length, head.Length));

        for (var length = limit; length >= 1; length--)
        {
            var matches = true;

            for (var i = 0; i < length; i++)
            {
                if (!WordsMatch(tail[tail.Length - length + i], head[i]))
                {
                    matches = false;
                    break;
                }
            }

            if (matches)
            {
                // A dash or comma left stranded at the front once its sentence has been trimmed
                // away is punctuation attaching to nothing, so it goes with the overlap.
                var rest = head.Skip(length).SkipWhile(word => NormaliseWord(word).Length == 0);

                return string.Join(" ", rest);
            }
        }

        return candidate;
    }

    private static bool WordsMatch(string left, string right) =>
        NormaliseWord(left) == NormaliseWord(right);

    /// <summary>One word, stripped to what the two passes are likely to agree on.</summary>
    private static string NormaliseWord(string word)
    {
        Span<char> buffer = word.Length <= 64 ? stackalloc char[word.Length] : new char[word.Length];
        var length = 0;

        foreach (var character in word)
        {
            if (char.IsLetterOrDigit(character))
            {
                buffer[length++] = char.ToLowerInvariant(character);
            }
        }

        return new string(buffer[..length]);
    }

    /// <summary>
    /// Strips punctuation, casing, and spacing so that "Right, so —" and "right so" compare
    /// equal. The two passes over an overlap often differ by exactly this much.
    /// </summary>
    internal static string Normalise(string text)
    {
        Span<char> buffer = text.Length <= 256 ? stackalloc char[text.Length] : new char[text.Length];
        var length = 0;
        var lastWasSpace = true;

        foreach (var character in text)
        {
            if (char.IsLetterOrDigit(character))
            {
                buffer[length++] = char.ToLowerInvariant(character);
                lastWasSpace = false;
            }
            else if (!lastWasSpace)
            {
                buffer[length++] = ' ';
                lastWasSpace = true;
            }
        }

        while (length > 0 && buffer[length - 1] == ' ')
        {
            length--;
        }

        return new string(buffer[..length]);
    }
}
