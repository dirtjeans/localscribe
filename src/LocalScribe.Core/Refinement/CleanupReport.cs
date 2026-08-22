namespace LocalScribe.Core.Refinement;

/// <summary>
/// Says what cleanup could not do, in words a reader can act on.
/// <para>
/// The counts were being kept and never shown. A passage left as the recogniser produced it looks
/// like the recording was unclear there — the reader has no way to tell it apart from speech that
/// genuinely came out badly, and no reason to think trying again would change anything.
/// </para>
/// <para>
/// The two failures are worth separating, because the answer differs. A call that never came back
/// is usually the backend: not running, busy, or handed more than it can hold, and a second
/// attempt often works. An answer that came back and did not match what was said is usually the
/// model being a small one, and a second attempt may well produce the same thing. Saying which
/// happened is the difference between a button worth pressing and a button worth ignoring.
/// </para>
/// </summary>
public static class CleanupReport
{
    /// <summary>
    /// What to tell the reader, or null when cleanup did everything asked of it.
    /// </summary>
    /// <param name="failed">Calls to the model that failed outright.</param>
    /// <param name="rejected">Windows whose rewrite did not match what was said, so were kept raw.</param>
    /// <param name="lastError">What the first failure said, when there was one.</param>
    public static string? Describe(int failed, int rejected, string? lastError = null)
    {
        if (failed <= 0 && rejected <= 0)
        {
            return null;
        }

        var notice = (failed, rejected) switch
        {
            ( > 0, > 0) =>
                $"{Passages(failed + rejected)} kept as transcribed: the cleanup model did not "
                + $"answer for {failed}, and answered badly for {rejected}.",
            ( > 0, _) =>
                $"{Passages(failed)} kept as transcribed — the cleanup model did not answer.",
            _ =>
                $"{Passages(rejected)} kept as transcribed — the cleanup model's rewrite did not "
                + "match what was said, so the original was kept.",
        };

        // Only where a call actually failed. A rejection has no error to report, and the message
        // from some earlier run would be worse than nothing.
        return failed > 0 && lastError is { Length: > 0 } reason ? $"{notice} ({reason})" : notice;
    }

    private static string Passages(int count) => count == 1 ? "One passage was" : $"{count} passages were";
}
