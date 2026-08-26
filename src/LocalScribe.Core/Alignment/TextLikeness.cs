namespace LocalScribe.Core.Alignment;

/// <summary>
/// How much a stretch of decoded audio resembles a piece of text.
/// <para>
/// The recogniser's read-back is greedy and misspells freely — "Elijah" comes back "elija" — so
/// resemblance is measured as the share of the text's letters that appear in the decode in
/// order, not as an exact match. Text placed over its own audio scores high through any amount
/// of misspelling; text placed over somebody else's sentence scores low, because the letters it
/// needs are simply not there in that order.
/// </para>
/// <para>
/// This is what lets a transcript line be asked to prove it was spoken. A span test cannot: a
/// line invented past the end of a recording, crammed onto whatever real audio is left, spreads
/// its words plausibly across somebody else's speech and looks placed. It does not read as
/// itself.
/// </para>
/// </summary>
public static class TextLikeness
{
    /// <summary>The share of <paramref name="text"/>'s letters found in order in <paramref name="heard"/>.</summary>
    public static double Share(string text, string heard)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(heard);

        var expected = Fold(text);

        if (expected.Length == 0 || heard.Length == 0)
        {
            return 0;
        }

        // Longest common subsequence: letters in the right order, with the decode's own
        // insertions and omissions between them.
        var previous = new int[heard.Length + 1];
        var current = new int[heard.Length + 1];

        for (var i = 1; i <= expected.Length; i++)
        {
            for (var j = 1; j <= heard.Length; j++)
            {
                current[j] = expected[i - 1] == heard[j - 1]
                    ? previous[j - 1] + 1
                    : Math.Max(previous[j], current[j - 1]);
            }

            (previous, current) = (current, previous);
            Array.Clear(current);
        }

        return previous[heard.Length] / (double)expected.Length;
    }

    /// <summary>Text as the recogniser's alphabet would carry it: letters only, lower case.</summary>
    public static string Fold(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        Span<char> buffer = text.Length <= 512 ? stackalloc char[text.Length] : new char[text.Length];
        var length = 0;

        foreach (var character in text)
        {
            if (char.IsLetter(character))
            {
                buffer[length++] = char.ToLowerInvariant(character);
            }
        }

        return new string(buffer[..length]);
    }
}
