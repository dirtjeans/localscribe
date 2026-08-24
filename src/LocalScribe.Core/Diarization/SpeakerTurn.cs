namespace LocalScribe.Core.Diarization;

/// <summary>A span of audio attributed to one speaker.</summary>
/// <param name="Speaker">Display label, such as <c>Speaker 1</c>.</param>
/// <param name="StartSeconds">Offset from the start of the recording.</param>
/// <param name="EndSeconds">End offset from the start of the recording.</param>
public sealed record SpeakerTurn(string Speaker, double StartSeconds, double EndSeconds)
{
    public double DurationSeconds => EndSeconds - StartSeconds;

    /// <summary>Seconds this turn shares with the span from <paramref name="start"/> to <paramref name="end"/>.</summary>
    public double OverlapWith(double start, double end) =>
        Math.Max(0, Math.Min(EndSeconds, end) - Math.Max(StartSeconds, start));

    public override string ToString() => $"{Speaker} [{StartSeconds:F1}s-{EndSeconds:F1}s]";
}

/// <summary>Knobs for diarization.</summary>
public sealed record DiarizationOptions
{
    /// <summary>
    /// Exact number of speakers, when it is known. Leave <c>null</c> to let clustering decide.
    /// <para>
    /// Setting this is worth doing whenever the count really is known — a two-person interview,
    /// say. Clustering is much better at splitting a known number of speakers than at guessing
    /// how many there are, and guessing wrong is the most common way diarization output goes bad.
    /// </para>
    /// </summary>
    public int? SpeakerCount { get; init; }

    /// <summary>
    /// Clustering distance threshold, used only when <see cref="SpeakerCount"/> is null. Lower
    /// values split more readily, producing more speakers.
    /// </summary>
    public float ClusteringThreshold { get; init; } = 0.5f;

    /// <summary>Speech shorter than this is not treated as a turn, in seconds.</summary>
    public float MinimumTurnSeconds { get; init; } = 0.3f;

    /// <summary>
    /// A gap shorter than this does not end a turn, in seconds. Without it, ordinary pauses
    /// between sentences fragment one person into a run of tiny turns.
    /// </summary>
    public float MinimumGapSeconds { get; init; } = 0.5f;

    /// <summary>
    /// Below this share of a segment's duration, the assigned speaker is marked uncertain.
    /// <para>
    /// A segment that straddles a speaker change cannot be split without word-level timestamps,
    /// which Whisper does not give us. So the segment keeps its dominant speaker and is flagged,
    /// which is honest about the ambiguity rather than hiding it behind a confident label.
    /// </para>
    /// </summary>
    public double UncertainBelowFraction { get; init; } = 0.6;

    public static DiarizationOptions Default { get; } = new();

    /// <summary>Throws when the options cannot produce a sensible run.</summary>
    public void Validate()
    {
        if (SpeakerCount is <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(SpeakerCount), SpeakerCount, "Speaker count must be at least 1 when specified.");
        }

        if (ClusteringThreshold <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(ClusteringThreshold), ClusteringThreshold, "Threshold must be positive.");
        }

        if (UncertainBelowFraction is < 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(UncertainBelowFraction), UncertainBelowFraction, "Fraction must be between 0 and 1.");
        }
    }
}
