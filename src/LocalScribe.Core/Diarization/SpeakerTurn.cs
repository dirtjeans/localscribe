namespace LocalScribe.Core.Diarization;

/// <summary>A stretch of audio attributed to one speaker.</summary>
/// <param name="Speaker">Index of the speaker, stable across the whole recording.</param>
/// <param name="StartSeconds">Offset from the start of the recording.</param>
/// <param name="EndSeconds">End offset from the start of the recording.</param>
public sealed record SpeakerTurn(int Speaker, double StartSeconds, double EndSeconds)
{
    /// <summary>How long the turn lasts.</summary>
    public double DurationSeconds => EndSeconds - StartSeconds;

    /// <summary>Seconds this turn and a given span have in common.</summary>
    public double OverlapWith(double startSeconds, double endSeconds) =>
        Math.Max(0, Math.Min(EndSeconds, endSeconds) - Math.Max(StartSeconds, startSeconds));

    /// <summary>A display name, one-based because "Speaker 0" reads like a bug.</summary>
    public string Label => $"Speaker {Speaker + 1}";
}
