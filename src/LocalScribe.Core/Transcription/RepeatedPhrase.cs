namespace LocalScribe.Core.Transcription;

/// <summary>
/// Removes a phrase the transcriber said twice in a row when it was only said once.
/// <para>
/// The stitcher already repairs the seam between two windows, where the same words are
/// transcribed twice because the audio was fed in twice. This is the other way a repeat happens:
/// the decoder loops inside a single window and emits a sentence, then emits it again. Nothing
/// looked for that, because every existing check compares the end of one segment with the start
/// of the next and this occurs in the middle of one.
/// </para>
/// <para>
/// It costs more than a doubled sentence to read. Word timing has to place every word it is
/// given somewhere inside the segment's audio, so a phrase that was never spoken takes its
/// frames from the words that follow, and the marker runs steadily further behind the voice for
/// the rest of the segment. A reader notices the drift long before they notice the repeat.
/// </para>
/// <para>
/// Only long runs, because people do repeat themselves. "No, no, no" and "I know, I know" are
/// how speech actually sounds and must survive untouched; five words repeated verbatim and
/// back to back is a decoder that has come unstuck.
/// </para>
/// </summary>
public static class RepeatedPhrase
{
    /// <summary>
    /// How many words must repeat before it is treated as a fault rather than as speech.
    /// <para>
    /// Chosen to sit above the length of ordinary spoken repetition. Emphasis repeats a word or
    /// a short phrase; it does not repeat a clause word for word with the same punctuation.
    /// </para>
    /// </summary>
    public const int ShortestRepeat = 5;

    /// <summary>
    /// How long a repeated run may be. A bound rather than a judgement: it keeps the search
    /// proportional to the length of the transcript, and a decoder that loops repeats a clause,
    /// not a paragraph.
    /// </summary>
    public const int LongestRepeat = 40;

    /// <summary>The text with any immediately repeated run of words reduced to one copy.</summary>
    public static string Trim(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var tokens = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).ToList();
        var before = tokens.Count;

        while (TrimOnce(tokens, owners: null))
        {
        }

        return tokens.Count == before ? text : string.Join(" ", tokens);
    }

    /// <summary>
    /// The same repair, read across segment boundaries rather than inside one segment.
    /// <para>
    /// Necessary because the two copies rarely land in the same segment. The transcriber breaks
    /// at the end of a sentence, and a sentence emitted twice therefore tends to break exactly
    /// between the copies — so each segment on its own contains one perfectly ordinary sentence
    /// and only the join is wrong. Trimming segment by segment cannot see that, and did not.
    /// </para>
    /// <para>
    /// Reading the transcript as one stream of words and putting the words back where they came
    /// from catches it wherever the boundary falls, and catches a repeat introduced after
    /// stitching — by cleanup rewriting the text, for instance — which the stitcher is long
    /// finished by.
    /// </para>
    /// </summary>
    public static IReadOnlyList<TranscriptSegment> TrimAcross(IReadOnlyList<TranscriptSegment> segments)
    {
        ArgumentNullException.ThrowIfNull(segments);

        var tokens = new List<string>();
        var owners = new List<int>();

        for (var i = 0; i < segments.Count; i++)
        {
            foreach (var word in segments[i].Text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
            {
                tokens.Add(word);
                owners.Add(i);
            }
        }

        var before = tokens.Count;

        while (TrimOnce(tokens, owners))
        {
        }

        if (tokens.Count == before)
        {
            return segments;
        }

        var rebuilt = new List<string>[segments.Count];
        for (var i = 0; i < tokens.Count; i++)
        {
            (rebuilt[owners[i]] ??= []).Add(tokens[i]);
        }

        var kept = new List<TranscriptSegment>(segments.Count);

        for (var i = 0; i < segments.Count; i++)
        {
            // A segment that was nothing but the second copy has nothing left to say. Keeping it
            // as an empty line would leave a timestamp with no words against it.
            if (rebuilt[i] is not { Count: > 0 } words)
            {
                continue;
            }

            kept.Add(segments[i] with { Text = string.Join(" ", words) });
        }

        return kept;
    }

    /// <summary>Removes one repeated run, and reports whether it found one.</summary>
    private static bool TrimOnce(List<string> tokens, List<int>? owners)
    {
        // Adjacency is counted in words, not in tokens. A decoder that loops often leaves a dash
        // or a comma sitting between the two copies, and treating that as a word between them
        // would hide the repeat — which is exactly what it did the first time this was written.
        var bare = tokens.Select(Bare).ToList();
        var words = new List<int>(tokens.Count);

        for (var i = 0; i < bare.Count; i++)
        {
            if (bare[i].Length > 0)
            {
                words.Add(i);
            }
        }

        // Longest first, so a sentence repeated whole is removed as one piece rather than being
        // picked apart from the middle.
        for (var length = Math.Min(LongestRepeat, words.Count / 2); length >= ShortestRepeat; length--)
        {
            for (var at = 0; at + (length * 2) <= words.Count; at++)
            {
                if (!Repeats(bare, words, at, length))
                {
                    continue;
                }

                // The second copy goes, along with whatever punctuation was standing between the
                // two. The first is the copy that carries the capitalisation and punctuation
                // joining it to what came before, which is what a reader is following.
                var from = words[at + length - 1] + 1;
                var to = words[at + (length * 2) - 1];

                tokens.RemoveRange(from, to - from + 1);
                owners?.RemoveRange(from, to - from + 1);

                return true;
            }
        }

        return false;
    }

    /// <summary>True when the run of words at <paramref name="at"/> is said again immediately after.</summary>
    private static bool Repeats(List<string> bare, List<int> words, int at, int length)
    {
        for (var i = 0; i < length; i++)
        {
            if (bare[words[at + i]] != bare[words[at + length + i]])
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>One word, stripped to what two passes of a decoder would agree on.</summary>
    private static string Bare(string word)
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
}
