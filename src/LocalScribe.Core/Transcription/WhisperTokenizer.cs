using System.Text;
using System.Text.Json;

namespace LocalScribe.Core.Transcription;

/// <summary>
/// Turns Whisper token ids back into text.
/// <para>
/// Only decoding is implemented, because that is all the pipeline needs: the prompt is a short
/// fixed list of special tokens, and everything after it comes back from the model. Skipping
/// the merge-rule half of BPE removes a great deal of code that would never run.
/// </para>
/// <para>
/// Special token ids are read from the vocabulary rather than hard-coded. They differ between
/// the English-only and multilingual models, and between exports, and a wrong constant here
/// produces a transcript full of stray <c>&lt;|notimestamps|&gt;</c> markers or silently eats
/// real words.
/// </para>
/// </summary>
public sealed class WhisperTokenizer
{
    private readonly IReadOnlyDictionary<int, string> _idToToken;
    private readonly Dictionary<char, byte> _unicodeToByte;

    private WhisperTokenizer(IReadOnlyDictionary<int, string> idToToken, SpecialTokens specialTokens)
    {
        _idToToken = idToToken;
        Special = specialTokens;
        _unicodeToByte = BuildByteDecoder();
    }

    /// <summary>The ids the decoder loop needs to build a prompt and know when to stop.</summary>
    public SpecialTokens Special { get; }

    /// <param name="StartOfTranscript">Opens every decode.</param>
    /// <param name="EndOfText">Ends a decode.</param>
    /// <param name="Transcribe">Selects transcription rather than translation.</param>
    /// <param name="NoTimestamps">Suppresses timestamp tokens.</param>
    /// <param name="NoSpeech">Marks a window the model believes contains no speech.</param>
    /// <param name="FirstTimestamp">
    /// Id of <c>&lt;|0.00|&gt;</c>. Timestamp ids run upward from here in 20 ms steps.
    /// </param>
    public sealed record SpecialTokens(
        int StartOfTranscript,
        int EndOfText,
        int Transcribe,
        int NoTimestamps,
        int NoSpeech,
        int FirstTimestamp)
    {
        /// <summary>True when the id is a timestamp marker rather than text.</summary>
        public bool IsTimestamp(int tokenId) => FirstTimestamp >= 0 && tokenId >= FirstTimestamp;

        /// <summary>True when the id carries no printable text.</summary>
        public bool IsSpecial(int tokenId) => tokenId >= StartOfTranscript;
    }

    /// <summary>Seconds represented by one step between consecutive timestamp tokens.</summary>
    public const double TimestampResolutionSeconds = 0.02;

    /// <summary>
    /// Loads a tokenizer from a Whisper model directory, which must contain <c>vocab.json</c>.
    /// Special tokens are taken from <c>added_tokens.json</c> when present and from the main
    /// vocabulary otherwise, since exports disagree about where they live.
    /// </summary>
    public static WhisperTokenizer LoadFromDirectory(string modelDirectory)
    {
        var vocabularyPath = Path.Combine(modelDirectory, "vocab.json");
        if (!File.Exists(vocabularyPath))
        {
            throw new FileNotFoundException(
                "vocab.json is required to decode model output. It ships alongside the Whisper " +
                "weights; the doctor tool downloads it with --fetch-models.",
                vocabularyPath);
        }

        var tokenToId = ReadTokenMap(vocabularyPath);

        var addedPath = Path.Combine(modelDirectory, "added_tokens.json");
        if (File.Exists(addedPath))
        {
            foreach (var (token, id) in ReadTokenMap(addedPath))
            {
                tokenToId[token] = id;
            }
        }

        return FromTokenMap(tokenToId);
    }

    /// <summary>Builds a tokenizer from an already-parsed token map. Exposed for testing.</summary>
    public static WhisperTokenizer FromTokenMap(IDictionary<string, int> tokenToId)
    {
        ArgumentNullException.ThrowIfNull(tokenToId);

        var idToToken = new Dictionary<int, string>(tokenToId.Count);
        foreach (var (token, id) in tokenToId)
        {
            idToToken[id] = token;
        }

        int Find(string name, bool required)
        {
            if (tokenToId.TryGetValue(name, out var id))
            {
                return id;
            }

            return required
                ? throw new InvalidOperationException(
                    $"The vocabulary is missing the required special token {name}. This usually means " +
                    "the file belongs to a plain GPT-2 model rather than a Whisper export.")
                : -1;
        }

        return new WhisperTokenizer(
            idToToken,
            new SpecialTokens(
                StartOfTranscript: Find("<|startoftranscript|>", required: true),
                EndOfText: Find("<|endoftext|>", required: true),
                Transcribe: Find("<|transcribe|>", required: true),
                NoTimestamps: Find("<|notimestamps|>", required: false),
                NoSpeech: Find("<|nospeech|>", required: false),
                FirstTimestamp: Find("<|0.00|>", required: false)));
    }

    /// <summary>
    /// Decodes token ids to text, dropping every special marker.
    /// </summary>
    public string Decode(IEnumerable<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);

        var builder = new StringBuilder();
        foreach (var id in tokenIds)
        {
            if (Special.IsSpecial(id))
            {
                continue;
            }

            if (_idToToken.TryGetValue(id, out var token))
            {
                builder.Append(token);
            }
        }

        return ByteDecode(builder.ToString());
    }

    /// <summary>Converts a timestamp token id to its position in seconds within the window.</summary>
    public double TimestampToSeconds(int tokenId) =>
        Special.IsTimestamp(tokenId)
            ? (tokenId - Special.FirstTimestamp) * TimestampResolutionSeconds
            : throw new ArgumentOutOfRangeException(nameof(tokenId), tokenId, "Not a timestamp token.");

    /// <summary>
    /// The prompt that opens a decode: start, language, task, and optionally a request for no
    /// timestamps. English-only models have no language token, so it is omitted when absent.
    /// </summary>
    public IReadOnlyList<int> BuildPrompt(bool withTimestamps = true, int languageToken = -1)
    {
        var prompt = new List<int> { Special.StartOfTranscript };

        if (languageToken >= 0)
        {
            prompt.Add(languageToken);
        }

        prompt.Add(Special.Transcribe);

        if (!withTimestamps && Special.NoTimestamps >= 0)
        {
            prompt.Add(Special.NoTimestamps);
        }

        return prompt;
    }

    private static Dictionary<string, int> ReadTokenMap(string path)
    {
        using var stream = File.OpenRead(path);
        return JsonSerializer.Deserialize<Dictionary<string, int>>(stream)
            ?? throw new InvalidOperationException($"{path} did not contain a token map.");
    }

    /// <summary>
    /// Reverses the byte-level encoding. Tokens are stored as printable stand-ins for raw bytes
    /// so that the vocabulary is valid text; undoing that mapping recovers the UTF-8 bytes.
    /// </summary>
    private string ByteDecode(string text)
    {
        var bytes = new List<byte>(text.Length);
        foreach (var character in text)
        {
            if (_unicodeToByte.TryGetValue(character, out var value))
            {
                bytes.Add(value);
            }
            else
            {
                // Anything outside the mapping is already literal text.
                bytes.AddRange(Encoding.UTF8.GetBytes(character.ToString()));
            }
        }

        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    /// <summary>
    /// Builds the byte-to-printable-character table used by GPT-2 style tokenizers, then
    /// inverts it. Printable ASCII and Latin-1 ranges map to themselves; everything else is
    /// displaced above U+0100 so no token contains a control character.
    /// </summary>
    private static Dictionary<char, byte> BuildByteDecoder()
    {
        var direct = new List<int>();
        for (var b = '!'; b <= '~'; b++)
        {
            direct.Add(b);
        }

        for (var b = '¡'; b <= '¬'; b++)
        {
            direct.Add(b);
        }

        for (var b = '®'; b <= 'ÿ'; b++)
        {
            direct.Add(b);
        }

        var mapping = new Dictionary<char, byte>();
        foreach (var value in direct)
        {
            mapping[(char)value] = (byte)value;
        }

        var next = 0;
        for (var b = 0; b < 256; b++)
        {
            if (direct.Contains(b))
            {
                continue;
            }

            mapping[(char)(256 + next)] = (byte)b;
            next++;
        }

        return mapping;
    }
}
