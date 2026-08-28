namespace LocalScribe.Core.Alignment;

/// <summary>
/// Throws out stamp anchors that cannot be true.
/// <para>
/// An anchor is a claim: by this second, the transcript has reached this token. The transcriber
/// pads its final window and stamps whatever falls in it up to eighteen seconds late, and
/// clamping such a stamp to the end of the recording turns it into a confident lie — half a
/// dozen lines' worth of anchors collapse onto the last frames, the corridor's spine expects
/// all of that text crammed there, and the true placement of the whole tail drifts further
/// from the spine than the band reaches. The tail was then not merely late: replaying the
/// inflation against a known-good recording convicted seven real lines as never spoken,
/// because the only placements the corridor allowed failed to read as anything.
/// </para>
/// <para>
/// The test is speakability. Nobody speaks several times faster than the recording's own
/// average pace, so an anchor that leaves more text than the remaining time could carry at a
/// sprint — or claims more text has passed than the elapsed time could carry — is dropped, and
/// the spine interpolates across the gap from the anchors that could be true. That spreads the
/// unanchored text at the recording's own pace, which is exactly the guess the inflated stamps
/// were failing to beat.
/// </para>
/// </summary>
public static class CredibleAnchors
{
    /// <summary>
    /// How many times the recording's average pace a claim may demand before it is a lie.
    /// Bursts run well above the average; eighteen seconds of stamp inflation runs further.
    /// </summary>
    public const double SprintFactor = 2.5;

    /// <summary>
    /// The sprint floor, in tokens per second. A recording that is mostly silence has a tiny
    /// average pace, and its one dense sentence should not read as impossible.
    /// </summary>
    public const double SlowestSprint = 25;

    /// <summary>Keeps the anchors whose claims the clock allows.</summary>
    /// <param name="anchors">Claims of (second, token reached), in any order.</param>
    /// <param name="tokens">How many tokens the whole transcript holds.</param>
    /// <param name="limit">Where the recording ends, in seconds.</param>
    public static List<(double Second, int Token)> Prune(
        IReadOnlyList<(double Second, int Token)> anchors, int tokens, double limit)
    {
        ArgumentNullException.ThrowIfNull(anchors);

        var sprint = Math.Max(SlowestSprint, tokens / Math.Max(1.0, limit) * SprintFactor);

        return [.. anchors.Where(a =>
            tokens - a.Token <= Math.Max(0, limit - a.Second) * sprint
            && a.Token <= a.Second * sprint)];
    }
}
