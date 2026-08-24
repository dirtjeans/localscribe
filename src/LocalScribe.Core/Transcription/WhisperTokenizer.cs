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

    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<byte, char> _byteToUnicode;
    private readonly int _longestToken;

    private WhisperTokenizer(IReadOnlyDictionary<int, string> idToToken, SpecialTokens specialTokens)
    {
        _idToToken = idToToken;
        Special = specialTokens;
        _unicodeToByte = BuildByteDecoder();
        Languages = FindLanguages(idToToken, specialTokens);

        _byteToUnicode = _unicodeToByte.ToDictionary(pair => pair.Value, pair => pair.Key);

        // Ordinary text tokens only. A prompt made of special markers would be nonsense, and
        // <|startoftranscript|> arriving inside one would end the prompt early.
        _tokenToId = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var (id, token) in idToToken)
        {
            if (id < specialTokens.StartOfTranscript && !_tokenToId.ContainsKey(token))
            {
                _tokenToId[token] = id;
            }
        }

        _longestToken = _tokenToId.Count == 0 ? 1 : _tokenToId.Keys.Max(t => t.Length);
    }

    /// <summary>
    /// Turns text into token ids, for use as a prompt.
    /// <para>
    /// Longest-match rather than true byte-pair encoding, which would need the merge table this
    /// export does not ship. The difference matters when encoding text the model must reproduce
    /// exactly; it does not matter for a prompt, which is only ever conditioning. Any valid
    /// sequence of tokens spelling the right characters does the job.
    /// </para>
    /// <para>
    /// Byte-level, like the vocabulary: text becomes UTF-8 bytes, each byte becomes the printable
    /// stand-in the vocabulary is written in, and the result is matched against it. That is what
    /// makes a leading space part of a word rather than a token of its own.
    /// </para>
    /// </summary>
    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var builder = new StringBuilder();
        foreach (var value in Encoding.UTF8.GetBytes(text))
        {
            builder.Append(_byteToUnicode.TryGetValue(value, out var character) ? character : (char)value);
        }

        var encoded = builder.ToString();
        var tokens = new List<int>();
        var at = 0;

        while (at < encoded.Length)
        {
            var length = Math.Min(_longestToken, encoded.Length - at);
            var matched = false;

            for (; length > 0; length--)
            {
                if (_tokenToId.TryGetValue(encoded.Substring(at, length), out var id))
                {
                    tokens.Add(id);
                    at += length;
                    matched = true;
                    break;
                }
            }

            // A character the vocabulary cannot spell at all. Skipping it keeps the rest of the
            // prompt usable, which is worth more than refusing the lot.
            if (!matched)
            {
                at++;
            }
        }

        return tokens;
    }

    /// <summary>
    /// Picks the language markers out of the vocabulary. They are the only special tokens whose
    /// body is a bare two or three letter code; everything else — transcribe, notimestamps,
    /// startofprev — is a word.
    /// </summary>
    private static Dictionary<int, string> FindLanguages(
        IReadOnlyDictionary<int, string> idToToken,
        SpecialTokens special)
    {
        var languages = new Dictionary<int, string>();

        foreach (var (id, token) in idToToken)
        {
            if (id <= special.StartOfTranscript || token.Length is < 6 or > 7)
            {
                continue;
            }

            if (!token.StartsWith("<|", StringComparison.Ordinal)
                || !token.EndsWith("|>", StringComparison.Ordinal))
            {
                continue;
            }

            var code = token[2..^2];
            if (code.All(char.IsAsciiLetterLower))
            {
                languages[id] = code;
            }
        }

        return languages;
    }

    /// <summary>The ids the decoder loop needs to build a prompt and know when to stop.</summary>
    public SpecialTokens Special { get; }

    /// <summary>
    /// Language tokens by id, e.g. 50259 to "en". Empty on an English-only model.
    /// <para>
    /// Read from the vocabulary rather than assumed to be a contiguous block after
    /// <c>&lt;|startoftranscript|&gt;</c>: that happens to be true of the published models, but
    /// it is a layout detail no export promises.
    /// </para>
    /// </summary>
    public IReadOnlyDictionary<int, string> Languages { get; private init; } =
        new Dictionary<int, string>();

    /// <summary>
    /// Reads the language off one step of decoder output.
    /// <para>
    /// Whisper predicts the language as the token immediately after
    /// <c>&lt;|startoftranscript|&gt;</c>, so a single decode step with nothing but that token
    /// scores every language at once. Restricting the search to language ids is what makes this
    /// a detection rather than an ordinary greedy step, which would usually just start
    /// transcribing.
    /// </para>
    /// </summary>
    /// <param name="logits">Scores for the whole vocabulary at the position after the start token.</param>
    /// <returns>The winning language id, or -1 when this model has no language tokens.</returns>
    public int DetectLanguage(ReadOnlySpan<float> logits)
    {
        var best = -1;
        var bestScore = float.NegativeInfinity;

        foreach (var (id, _) in Languages)
        {
            if (id < logits.Length && logits[id] > bestScore)
            {
                bestScore = logits[id];
                best = id;
            }
        }

        return best;
    }

    /// <summary>The code for a language id, e.g. "en", or null when it is not one.</summary>
    public string? LanguageCode(int tokenId) =>
        Languages.TryGetValue(tokenId, out var code) ? code : null;

    /// <param name="StartOfTranscript">Opens every decode.</param>
    /// <param name="EndOfText">Ends a decode.</param>
    /// <param name="Transcribe">Selects transcription rather than translation.</param>
    /// <param name="StartOfPrev">
    /// Id of <c>&lt;|startofprev|&gt;</c>, which opens a prompt of earlier or example text. -1
    /// when the export has no such token.
    /// </param>
    /// <param name="English">
    /// Id of <c>&lt;|en|&gt;</c>, or -1 on an English-only model, which has no language tokens
    /// because it needs none.
    /// </param>
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
        int FirstTimestamp,
        int English = -1,
        int StartOfPrev = -1,
        int Translate = -1)
    {
        /// <summary>True when this vocabulary carries language tokens and therefore needs one.</summary>
        public bool IsMultilingual => English >= 0;

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
    public static WhisperTokenizer LoadFromDirectory(string modelDirectory) =>
        LoadFromFile(Path.Combine(modelDirectory, "vocab.json"));

    /// <summary>
    /// Loads a tokenizer from a specific vocabulary file. Special tokens are taken from an
    /// <c>added_tokens.json</c> sitting beside it when present, since exports disagree about
    /// where those live.
    /// </summary>
    public static WhisperTokenizer LoadFromFile(string vocabularyPath)
    {
        var modelDirectory = Path.GetDirectoryName(vocabularyPath) ?? ".";

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

                // Not required: English-only exports have no translate task, having nothing to
                // translate from.
                Translate: Find("<|translate|>", required: false),
                NoTimestamps: Find("<|notimestamps|>", required: false),
                NoSpeech: Find("<|nospeech|>", required: false),
                FirstTimestamp: Find("<|0.00|>", required: false),
                English: Find("<|en|>", required: false),
                StartOfPrev: Find("<|startofprev|>", required: false)));
    }

    /// <summary>
    /// The token for a task, falling back to transcription when the export cannot translate.
    /// <para>
    /// Silently, because the alternative is refusing to transcribe a recording over a setting
    /// the user can change afterwards. What they get is the language they spoke, which is the
    /// safer of the two wrong answers.
    /// </para>
    /// </summary>
    public int TaskToken(SpeechTask task) =>
        task == SpeechTask.TranslateToEnglish && Special.Translate >= 0
            ? Special.Translate
            : Special.Transcribe;

    /// <summary>True when this export can render speech as English.</summary>
    public bool CanTranslate => Special.Translate >= 0;

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
    /// <summary>
    /// Builds the tokens that open a decode.
    /// <para>
    /// The order is fixed and the language slot is not optional on a multilingual model:
    /// <c>&lt;|startoftranscript|&gt; &lt;|en|&gt; &lt;|transcribe|&gt;</c>, then
    /// <c>&lt;|notimestamps|&gt;</c> if timestamps are not wanted. English-only models have no
    /// language tokens at all and correctly get none.
    /// </para>
    /// <para>
    /// Leaving the slot empty on a multilingual model does not make the model detect the
    /// language, it makes it guess, and it guesses again on every pass. The observed result was
    /// a live transcript that returned "Gracias." for the first two seconds of English speech,
    /// and that flipped between properly punctuated output and a bare lowercase run of words
    /// from one pass to the next. Both are the same fault: an unconditioned prompt leaves the
    /// model free to pick a different language and a different output convention each time.
    /// </para>
    /// <para>
    /// The answer is to detect the language once and then say so on every pass, not to hard-code
    /// one. See <see cref="DetectLanguage"/>.
    /// </para>
    /// </summary>
    /// <param name="task">
    /// Whether to write down what was said or to render it in English. Falls back to writing it
    /// down when the export has no translate task, which is what an English-only model is.
    /// </param>
    /// <param name="withTimestamps">False to suppress timestamp tokens.</param>
    /// <param name="languageToken">
    /// The detected language id. Falls back to English when detection has not run and the model
    /// is multilingual, because naming a language is what keeps the output stable — an empty
    /// slot is the one option that is always wrong.
    /// </param>
    /// <param name="priorTokens">
    /// Text to condition on, as tokens, or null for none. Emitted after
    /// <c>&lt;|startofprev|&gt;</c> and before the start marker, which is where Whisper expects
    /// a prompt and the only place it is read as context rather than as speech.
    /// </param>
    public IReadOnlyList<int> BuildPrompt(
        bool withTimestamps = true,
        int languageToken = -1,
        IReadOnlyList<int>? priorTokens = null,
        SpeechTask task = SpeechTask.Transcribe)
    {
        var prompt = new List<int>();

        if (priorTokens is { Count: > 0 } && Special.StartOfPrev >= 0)
        {
            prompt.Add(Special.StartOfPrev);
            prompt.AddRange(priorTokens);
        }

        prompt.Add(Special.StartOfTranscript);

        var language = languageToken >= 0 ? languageToken : Special.English;
        if (language >= 0)
        {
            prompt.Add(language);
        }

        prompt.Add(TaskToken(task));

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
