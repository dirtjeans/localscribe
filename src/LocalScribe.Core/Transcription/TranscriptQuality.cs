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
}
