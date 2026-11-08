namespace LocalScribe.Core.Pipeline;

/// <summary>Says how long a finished run took, in the words the progress steps used.</summary>
public static class RunTimes
{
    /// <summary>
    /// A step quicker than this did no work worth naming: no aligner installed, say, so it
    /// started and finished in the same moment.
    /// </summary>
    public const double LeastWorthNamingSeconds = 0.5;

    /// <summary>A length of time as a clock reads: 0:38, 4:12, 1:02:03.</summary>
    public static string Clock(double seconds)
    {
        var whole = (long)Math.Round(Math.Max(0, seconds));

        return whole >= 3600
            ? $"{whole / 3600}:{whole / 60 % 60:00}:{whole % 60:00}"
            : $"{whole / 60}:{whole % 60:00}";
    }

    /// <summary>The headline: the whole run against the length of the recording.</summary>
    public static string Headline(double totalSeconds, double audioSeconds) =>
        audioSeconds > 0
            ? $"Done in {Clock(totalSeconds)} for {Clock(audioSeconds)} of audio."
            : $"Done in {Clock(totalSeconds)}.";

    /// <summary>
    /// The headline and each step's own time. The steps overlap, so they add up to more than the
    /// whole; each is the time from that step starting to that step finishing.
    /// </summary>
    public static string Describe(
        double totalSeconds,
        double audioSeconds,
        IEnumerable<(string Label, double? Seconds)> steps)
    {
        ArgumentNullException.ThrowIfNull(steps);

        var named = steps
            .Where(step => step.Seconds >= LeastWorthNamingSeconds)
            .Select(step => $"{step.Label} {Clock(step.Seconds!.Value)}")
            .ToList();

        var headline = Headline(totalSeconds, audioSeconds);

        return named.Count == 0 ? headline : $"{headline} {string.Join(" · ", named)}.";
    }
}
