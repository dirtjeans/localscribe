using System.Globalization;
using System.Text;
using System.Text.Json;

namespace LocalScribe.Core.Alignment;

/// <summary>
/// Turns transcript words into the letters the alignment model knows.
/// <para>
/// The multilingual aligner works on a romanised alphabet: thirty-one tokens, being a blank,
/// twenty-six Latin letters and an apostrophe. That is the whole point of it — one model for
/// every language, reached by writing every language in the same letters, so a Greek or Cyrillic
/// or accented word is folded down to the nearest plain Latin spelling before it is aligned.
/// </para>
/// <para>
/// The folding here is the cheap kind: strip the accents, lower the case, drop the punctuation.
/// It is exact for English and close enough for anything written in a Latin alphabet. A word in
/// a script it cannot fold has no letters left and is skipped, which costs that word its timing
/// and nothing else.
/// </para>
/// </summary>
public sealed class AlignmentAlphabet
{
    private readonly Dictionary<char, int> _tokens;

    private AlignmentAlphabet(Dictionary<char, int> tokens, int blank)
    {
        _tokens = tokens;
        Blank = blank;
        Size = tokens.Count == 0 ? 0 : Math.Max(blank, tokens.Values.Max()) + 1;
    }

    /// <summary>The token meaning "nothing in particular".</summary>
    public int Blank { get; }

    /// <summary>
    /// The letter a token stands for, or a space when it stands for nothing sayable.
    /// <para>
    /// Only reading a grid back needs this — aligning goes the other way. It is what lets a scan
    /// be checked by decoding it and seeing whether the result reads like the speech.
    /// </para>
    /// </summary>
    public char Letter(int token)
    {
        foreach (var (letter, index) in _tokens)
        {
            if (index == token)
            {
                return letter;
            }
        }

        return ' ';
    }

    /// <summary>How many tokens the model knows.</summary>
    public int Size { get; }

    /// <summary>Reads the alphabet from the model's vocab.json.</summary>
    public static AlignmentAlphabet Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        return Parse(File.ReadAllText(path));
    }

    /// <summary>The same, from the file's contents.</summary>
    public static AlignmentAlphabet Parse(string json)
    {
        ArgumentException.ThrowIfNullOrEmpty(json);

        var entries = JsonSerializer.Deserialize<Dictionary<string, int>>(json)
            ?? throw new InvalidDataException("The alignment vocabulary could not be read.");

        var tokens = new Dictionary<char, int>();
        int? named = null;
        int? padding = null;

        foreach (var (name, id) in entries)
        {
            // The special tokens are written in angle brackets; only the single characters are
            // letters we can spell with.
            if (name.Length == 1)
            {
                tokens[char.ToLowerInvariant(name[0])] = id;
            }
            else if (name.Equals("<blank>", StringComparison.OrdinalIgnoreCase))
            {
                named = id;
            }
            else if (name.Equals("<pad>", StringComparison.OrdinalIgnoreCase))
            {
                padding = id;
            }
        }

        // Whichever the model calls it. Both names appear across exports and this one declares
        // both, in which case the token actually named "blank" is the one meant. Deciding by
        // which id happened to be read first returned the padding token instead, which would
        // have aligned every transcript against the wrong silence.
        var blank = named ?? padding ?? 0;

        if (tokens.Count == 0)
        {
            throw new InvalidDataException("The alignment vocabulary has no letters in it.");
        }

        return new AlignmentAlphabet(tokens, blank);
    }

    /// <param name="Word">The word as it appears in the transcript.</param>
    /// <param name="Index">Its position in the list of words handed in.</param>
    /// <param name="First">Where its letters begin in the token sequence.</param>
    /// <param name="Count">How many tokens it spells to.</param>
    public sealed record Spelling(string Word, int Index, int First, int Count);

    /// <summary>
    /// Spells a list of words, reporting which tokens belong to which word.
    /// </summary>
    /// <returns>The tokens, and one spelling per word that could be spelled at all.</returns>
    public (IReadOnlyList<int> Tokens, IReadOnlyList<Spelling> Words) Spell(IReadOnlyList<string> words)
    {
        ArgumentNullException.ThrowIfNull(words);

        var tokens = new List<int>();
        var spellings = new List<Spelling>();

        for (var i = 0; i < words.Count; i++)
        {
            var first = tokens.Count;

            foreach (var letter in Fold(words[i]))
            {
                if (_tokens.TryGetValue(letter, out var id))
                {
                    tokens.Add(id);
                }
            }

            if (tokens.Count > first)
            {
                spellings.Add(new Spelling(words[i], i, first, tokens.Count - first));
            }
        }

        return (tokens, spellings);
    }

    /// <summary>
    /// A word reduced to plain lowercase Latin letters: accents removed, case dropped, anything
    /// that is not a letter or an apostrophe discarded.
    /// </summary>
    internal static string Fold(string word)
    {
        ArgumentNullException.ThrowIfNull(word);

        // Decomposing separates a letter from its accent, so the accent can simply be dropped.
        var decomposed = word.Normalize(NormalizationForm.FormD);
        var builder = new StringBuilder(decomposed.Length);

        foreach (var character in decomposed)
        {
            if (CharUnicodeInfo.GetUnicodeCategory(character) == UnicodeCategory.NonSpacingMark)
            {
                continue;
            }

            var lower = char.ToLowerInvariant(character);

            if (lower is >= 'a' and <= 'z' or '\'')
            {
                builder.Append(lower);
            }
            else if (lower is '’')
            {
                // A typographic apostrophe is still an apostrophe.
                builder.Append('\'');
            }
            else if (Ligatures.TryGetValue(lower, out var spelled))
            {
                builder.Append(spelled);
            }
        }

        return builder.ToString();
    }

    /// <summary>
    /// Letters that are not accented versions of anything, and so survive decomposition intact
    /// only to be discarded as unrecognised.
    /// <para>
    /// Stripping accents handles e-acute and u-umlaut and n-tilde, because those really are a
    /// letter plus a mark. It does nothing for the letters below, which are letters in their own
    /// right — so a Danish place name folded to a single "r", losing two thirds of the word and
    /// any hope of aligning it. These spellings are how the languages that use them are
    /// romanised.
    /// </para>
    /// </summary>
    private static readonly Dictionary<char, string> Ligatures = new()
    {
        ['æ'] = "ae",
        ['œ'] = "oe",
        ['ø'] = "o",
        ['ß'] = "ss",
        ['þ'] = "th",
        ['ð'] = "d",
        ['đ'] = "d",
        ['ł'] = "l",
        ['ħ'] = "h",
        ['ŋ'] = "ng",
    };
}
