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

    /// <summary>The text with any immediately repeated run of words reduced to one copy.</summary>
    public static string Trim(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var tokens = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).ToList();

        // Repeated until nothing more changes, which handles a decoder that stuck three or four
        // times rather than twice.
        while (TrimOnce(tokens))
        {
        }

        return tokens.Count == 0
            ? text
            : string.Join(" ", tokens) is var joined && joined == text ? text : joined;
    }

    /// <summary>Removes one repeated run, and reports whether it found one.</summary>
    private static bool TrimOnce(List<string> tokens)
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
        for (var length = words.Count / 2; length >= ShortestRepeat; length--)
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
