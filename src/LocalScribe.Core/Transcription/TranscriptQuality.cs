namespace LocalScribe.Core.Transcription;

/// <summary>
/// Tells a well-formed transcription from Whisper's degenerate one.
/// <para>
/// Whisper has two output conventions and picks between them per decode. The normal one is
/// sentence-cased and punctuated. The other is a bare lowercase run of words, and it appears
/// deterministically on particular windows — of two passes over the same speech, one second
/// apart, the earlier returned
/// </para>
/// <code>
/// Okay, I'm going to test the transcription ability one more time. … I hope so.
/// </code>
/// <para>and the later returned</para>
/// <code>
/// okay i'm going to test the transcription ability one more time … hope so
/// </code>
/// <para>
/// The second is not a different transcription, it is the same one delivered worse, and it drops
/// words too. A live session re-transcribes its window on every pass, so it usually holds a good
/// reading and a bad one of identical audio, and it should not keep the bad one merely because
/// it arrived last.
/// </para>
/// <para>
/// This is a guard against a model quirk, not a judgement about writing. It only ever prefers
/// one reading of the same audio over another; it never edits either.
/// </para>
/// </summary>
public static class TranscriptQuality
{
    /// <summary>Marks that end or divide a sentence. Apostrophes are not among them.</summary>
    private static readonly char[] SentencePunctuation = ['.', ',', '?', '!', ';', ':'];

    /// <summary>
    /// True when text shows neither sentence punctuation nor capitalisation — the signature of
    /// the degenerate convention.
    /// </summary>
    public static bool LooksUnformatted(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        if (text.Trim().Length == 0)
        {
            return false;
        }

        return text.IndexOfAny(SentencePunctuation) < 0 && !text.Any(char.IsUpper);
    }

    /// <summary>
    /// True when <paramref name="candidate"/> should replace <paramref name="current"/>: same
    /// speech, but one reading is formatted and the other is not.
    /// </summary>
    /// <param name="current">The reading a later pass produced.</param>
    /// <param name="candidate">A reading an earlier pass produced.</param>
    public static bool PreferCandidate(string current, string candidate)
    {
        ArgumentNullException.ThrowIfNull(current);
        ArgumentNullException.ThrowIfNull(candidate);

        if (!LooksUnformatted(current) || LooksUnformatted(candidate))
        {
            return false;
        }

        // Only when the earlier reading says substantially as much. A formatted fragment is not
        // an improvement on a complete transcription, however badly the latter is presented, and
        // losing words is a worse failure than losing commas.
        return WordCount(candidate) >= WordCount(current) * 0.9;
    }

    private static int WordCount(string text) =>
        text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length;

    /// <summary>
    /// True when two readings are of the same speech rather than two different things.
    /// <para>
    /// A second attempt at a window can come back better formatted and wrong — conditioning the
    /// model on example text can leave the example in the output, which produces a confidently
    /// punctuated sentence made partly of words nobody said. Formatting alone cannot tell that
    /// apart from a genuine improvement, so the words have to be checked too. A repair that
    /// changes what was said is not a repair.
    /// </para>
    /// </summary>
    /// <param name="original">The reading being replaced.</param>
    /// <param name="candidate">The reading offered instead.</param>
    /// <param name="minimumShared">Fraction of the original's words the candidate must still contain.</param>
    public static bool SaysTheSameThing(string original, string candidate, double minimumShared = 0.7)
    {
        ArgumentNullException.ThrowIfNull(original);
        ArgumentNullException.ThrowIfNull(candidate);

        var before = Words(original);
        if (before.Count == 0)
        {
            return true;
        }

        var after = Words(candidate).ToHashSet(StringComparer.Ordinal);
        var kept = before.Count(word => after.Contains(word));

        return kept >= before.Count * minimumShared;
    }

    /// <summary>
    /// True when a cleaned-up window is a repair of the original rather than a rewrite of it.
    /// <para>
    /// Checked in both directions, because the two ways cleanup goes wrong are opposites and
    /// only one of them is caught by asking whether the words survived. A small model asked to
    /// punctuate a transcript will sometimes drop a clause it judged redundant — given
    /// </para>
    /// <code>
    /// okay i am going to test the transcription ability one more time lets see how well it
    /// works does it punctuate well i hope so
    /// </code>
    /// <para>one returned</para>
    /// <code>
    /// Okay, I'm going to test the transcription ability one more time. Does it punctuate well?
    /// I hope so.
    ///
    /// (Note: I've added a question mark at the end of the sentence…)
    /// </code>
    /// <para>
    /// — losing a whole clause the speaker said, and appending an explanation nobody asked for,
    /// in the same reply, despite being told not to. Prompting does not fix this at these model
    /// sizes; refusing the result does. A window that fails is kept as it was, on the grounds
    /// that unpunctuated and accurate beats polished and wrong.
    /// </para>
    /// </summary>
    /// <param name="original">The raw window handed to the model.</param>
    /// <param name="cleaned">What came back.</param>
    public static bool IsFaithfulCleanup(string original, string cleaned)
    {
        ArgumentNullException.ThrowIfNull(original);
        ArgumentNullException.ThrowIfNull(cleaned);

        var before = Words(original);
        if (before.Count == 0)
        {
            return true;
        }

        if (Words(cleaned).Count == 0)
        {
            return false;
        }

        // Measured over content words only. Plain retention cannot tell the two failures apart:
        // dropping "let's see how well it works" and dropping "um … uh … you know" both leave
        // three quarters of the words standing, and the first is a lost sentence while the
        // second is the job being done correctly. Which words went is the whole signal.
        var content = ContentWords(before);

        if (content.Count > 0)
        {
            var after = Words(cleaned).ToHashSet(StringComparer.Ordinal);
            var kept = content.Count(word => after.Contains(word));

            if (kept < content.Count * MinimumKeptWhenCleaning)
            {
                return false;
            }
        }

        // Growth is the tell for commentary. Punctuating text does not lengthen it; explaining
        // what you punctuated does.
        return Words(cleaned).Count <= before.Count * MaximumGrowthWhenCleaning;
    }

    /// <summary>
    /// Words a cleanup pass is expected to delete. Deliberately short: every word on this list
    /// is one the guard stops protecting, so it holds only sounds with no meaning to lose.
    /// Judgement calls like "well", "right", and "okay" are content until proven otherwise.
    /// </summary>
    private static readonly HashSet<string> Fillers = new(StringComparer.Ordinal)
    {
        "um", "umm", "uh", "uhh", "uhm", "er", "err", "erm", "ah", "ahh", "eh",
        "hmm", "hm", "mm", "mmm", "mhm", "huh",
    };

    /// <summary>
    /// Discourse markers made of words that carry meaning on their own. "You know" is filler
    /// and cleanup is right to delete it, but "you" and "know" are ordinary words that must
    /// stay protected everywhere else, so the pair has to be recognised as a pair.
    /// </summary>
    private static readonly (string First, string Second)[] FillerPhrases =
    [
        ("you", "know"),
        ("i", "mean"),
    ];

    /// <summary>The words of <paramref name="words"/> that cleanup is not licensed to remove.</summary>
    private static List<string> ContentWords(List<string> words)
    {
        var content = new List<string>(words.Count);

        for (var i = 0; i < words.Count; i++)
        {
            if (Fillers.Contains(words[i]))
            {
                continue;
            }

            if (i + 1 < words.Count
                && FillerPhrases.Any(phrase => phrase.First == words[i] && phrase.Second == words[i + 1]))
            {
                i++;
                continue;
            }

            content.Add(words[i]);
        }

        return content;
    }

    /// <summary>Fraction of the original's content words a cleaned window must still contain.</summary>
    private const double MinimumKeptWhenCleaning = 0.85;

    /// <summary>How much longer than the original a cleaned window may be.</summary>
    private const double MaximumGrowthWhenCleaning = 1.2;

    /// <summary>
    /// What to show where the model could not make out the words.
    /// </summary>
    public const string Unintelligible = "…";

    /// <summary>
    /// True when a segment is the model guessing rather than hearing.
    /// <para>
    /// Whisper does not fall silent on audio it cannot read. It invents fluent, plausible,
    /// confidently punctuated sentences out of noise and crosstalk, and nothing in the text
    /// marks them as different from the parts it heard perfectly well. A reader has no way to
    /// tell, which makes an invented sentence worse than a gap: a gap is honest and a reader
    /// can go and listen.
    /// </para>
    /// <para>
    /// The tell is the model's own confidence, which is why it is now carried on every segment.
    /// The threshold is Whisper's own: OpenAI's implementation treats a window averaging below
    /// -1 as a failed decode and retries it at a higher temperature. Ordinary speech sits near
    /// -0.2.
    /// </para>
    /// </summary>
    /// <param name="text">The words themselves, checked for the model talking in circles.</param>
    /// <param name="averageLogProbability">Mean confidence over the segment's tokens.</param>
    /// <param name="noSpeechProbability">How sure the model was that nobody was talking.</param>
    public static bool SoundsLikeGuesswork(
        string text,
        double averageLogProbability,
        double noSpeechProbability = 0) =>
        LoopsOnItself(text)
        || averageLogProbability < GuessworkBelow
        || noSpeechProbability > SilenceAbove;

    /// <summary>
    /// True when the model has got stuck repeating itself.
    /// <para>
    /// The other way Whisper fails, and the more common one on a real recording. Where it cannot
    /// follow the audio it sometimes latches onto a phrase and emits it over and over — thirty
    /// seconds of "I don't. I don't. I don't." — with perfectly ordinary confidence, because it
    /// is entirely sure of each word. Confidence cannot see this at all; the shape of the text
    /// gives it away instantly.
    /// </para>
    /// <para>
    /// Measured by how well the text compresses, which is Whisper's own test: repetitive text
    /// compresses far better than speech, and OpenAI's implementation treats a ratio above 2.4
    /// as a decode to throw away. Ordinary English runs around 1.2 to 1.6.
    /// </para>
    /// </summary>
    public static bool LoopsOnItself(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        // Too short to tell. Any brief phrase compresses badly, and repetition needs room to
        // show itself.
        if (text.Length < ShortestWorthChecking)
        {
            return false;
        }

        var raw = System.Text.Encoding.UTF8.GetBytes(text);

        using var compressed = new MemoryStream();
        using (var gzip = new System.IO.Compression.GZipStream(compressed, System.IO.Compression.CompressionLevel.Optimal, leaveOpen: true))
        {
            gzip.Write(raw, 0, raw.Length);
        }

        return raw.Length / (double)compressed.Length > RepetitionAbove;
    }

    /// <summary>
    /// Cuts a repeating tail off a segment, leaving an ellipsis in its place.
    /// <para>
    /// Where <see cref="LoopsOnItself"/> judges a whole segment, this finds the loop inside one.
    /// That is the shape the failure actually takes: the model follows the audio for twenty
    /// seconds, loses it, and spends the last ten repeating a phrase — "I don't. I don't. I
    /// don't." — while the segment as a whole still compresses like ordinary speech, because
    /// most of it is ordinary speech.
    /// </para>
    /// <para>
    /// Throwing the segment away would take the good twenty seconds with it, so only the tail
    /// goes. An ellipsis is honest about the gap and leaves the reader somewhere to listen.
    /// </para>
    /// </summary>
    public static string TrimLoopedTail(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (words.Length < MinimumRepeats * 2)
        {
            return text;
        }

        var plain = words.Select(Bare).ToArray();
        var cutAt = words.Length;

        // A window that ends mid-loop leaves a fragment of one: "I don't. I don't. I don". Read
        // strictly from the last word, "don" fails to match "don't" and the whole loop goes
        // unseen — which is exactly what happened on the recording this was built from. So a
        // few trailing words are allowed to be ignored, and the fragment goes with the loop.
        for (var ignore = 0; ignore <= PartialTailWords && ignore < plain.Length; ignore++)
        {
            var end = plain.Length - ignore;

            // The longest loop wins. A phrase of one word and a phrase of four can both be
            // repeating at the end, and cutting at the earlier point removes the whole of it.
            for (var phrase = 1; phrase <= LongestLoopedPhrase; phrase++)
            {
                var repeats = 1;

                while ((repeats + 1) * phrase <= end && RepeatsAgain(plain, end, phrase, repeats))
                {
                    repeats++;
                }

                if (repeats >= MinimumRepeats)
                {
                    cutAt = Math.Min(cutAt, end - (repeats * phrase));
                }
            }
        }

        if (cutAt >= words.Length)
        {
            return text;
        }

        // Nothing left but the loop: the segment was never anything else.
        var kept = string.Join(' ', words.Take(cutAt)).TrimEnd();

        return kept.Length == 0 ? Unintelligible : $"{kept} {Unintelligible}";
    }

    /// <summary>True when the block of <paramref name="phrase"/> words before the last
    /// <paramref name="repeats"/> blocks matches the one ending at <paramref name="end"/>.</summary>
    private static bool RepeatsAgain(string[] words, int end, int phrase, int repeats)
    {
        var last = end - phrase;
        var earlier = end - ((repeats + 1) * phrase);

        for (var i = 0; i < phrase; i++)
        {
            if (!string.Equals(words[last + i], words[earlier + i], StringComparison.Ordinal))
            {
                return false;
            }
        }

        return true;
    }

    private static string Bare(string word) =>
        new(word.Where(char.IsLetterOrDigit).Select(char.ToLowerInvariant).ToArray());

    /// <summary>
    /// How many times a phrase must repeat before it is a loop rather than emphasis. Somebody
    /// saying a thing twice means it; saying it four times running is the model stuck.
    /// </summary>
    private const int MinimumRepeats = 4;

    /// <summary>
    /// How much of a trailing fragment to disregard when hunting for the loop. A window that
    /// runs out mid-repetition ends on half a phrase, and matching from the true last word would
    /// miss the loop entirely.
    /// </summary>
    private const int PartialTailWords = 4;

    /// <summary>
    /// The longest phrase worth checking for a loop. Beyond this, a repeat is more likely to be
    /// a speaker returning to their point than a decode that has come off the rails.
    /// </summary>
    private const int LongestLoopedPhrase = 6;

    /// <summary>
    /// Compression ratio past which text is repetition rather than speech. Whisper's own figure
    /// for a decode it discards.
    /// </summary>
    private const double RepetitionAbove = 2.4;

    /// <summary>
    /// Below this many characters the compression ratio says more about gzip's header than about
    /// the words, so the check abstains.
    /// </summary>
    private const int ShortestWorthChecking = 200;

    /// <summary>
    /// Mean log probability below which a segment is the model guessing. Whisper's own figure
    /// for a decode it would rather retry than keep.
    /// </summary>
    private const double GuessworkBelow = -1.0;

    /// <summary>How sure of silence the model must be before its words are disbelieved.</summary>
    private const double SilenceAbove = 0.6;

    /// <summary>Words stripped to what two readings of the same speech would agree on.</summary>
    private static List<string> Words(string text) =>
        text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            .Select(word => new string(word.Where(char.IsLetterOrDigit).Select(char.ToLowerInvariant).ToArray()))
            .Where(word => word.Length > 0)
            .ToList();
}
