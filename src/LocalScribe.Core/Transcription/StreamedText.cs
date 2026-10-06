namespace LocalScribe.Core.Transcription;

/// <summary>
/// What a file run shows of each window as it lands.
/// <para>
/// Windows overlap, so the words at every seam are transcribed twice, and the stitcher only
/// removes the second copy once the whole recording is in. Until then the reader saw every seam
/// doubled, and the progressive timing pass — which times exactly this text — had to fund each
/// repeated word with real audio, pushing its neighbours off their own. Trimmed here with the
/// stitcher's own rules, so the streamed text says what the finished transcript will.
/// </para>
/// </summary>
public static class StreamedText
{
    /// <summary>
    /// One window's text with its opening overlap and any looped phrase removed.
    /// </summary>
    /// <param name="previous">The last window's streamed text that had any words, or null.</param>
    /// <param name="window">This window's text as the transcriber wrote it.</param>
    public static string Trim(string? previous, string window)
    {
        ArgumentNullException.ThrowIfNull(window);

        // The stitcher's order: the seam first, then a loop inside what remains.
        var text = previous is { Length: > 0 }
            ? TranscriptStitcher.TrimLeadingOverlap(previous, window)
            : window;

        return RepeatedPhrase.Trim(text).Trim();
    }

    /// <summary>
    /// How the streamed text differs from the finished one, in words: those shown that the
    /// finished transcript does not have, and those it has that were never shown.
    /// <para>
    /// A diagnostic for the stage log. Each extra word is one a reader saw vanish when the run
    /// finished; matched as a longest common subsequence, ignoring case and punctuation.
    /// </para>
    /// </summary>
    public static (int Extra, int Missing) Compare(string streamed, string finished)
    {
        ArgumentNullException.ThrowIfNull(streamed);
        ArgumentNullException.ThrowIfNull(finished);

        var a = Words(streamed);
        var b = Words(finished);

        // Two rows rather than the whole table: a long recording is thousands of words a side.
        var previous = new int[b.Length + 1];
        var current = new int[b.Length + 1];

        for (var i = 1; i <= a.Length; i++)
        {
            for (var j = 1; j <= b.Length; j++)
            {
                current[j] = a[i - 1] == b[j - 1]
                    ? previous[j - 1] + 1
                    : Math.Max(previous[j], current[j - 1]);
            }

            (previous, current) = (current, previous);
        }

        var common = previous[b.Length];
        return (a.Length - common, b.Length - common);
    }

    private static string[] Words(string text) =>
        [.. text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            .Select(word => new string([.. word.Where(char.IsLetterOrDigit)]).ToLowerInvariant())
            .Where(word => word.Length > 0)];
}
