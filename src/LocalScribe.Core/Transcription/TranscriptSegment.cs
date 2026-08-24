namespace LocalScribe.Core.Transcription;

/// <summary>A span of recognised speech with its position in the recording.</summary>
/// <param name="Text">Recognised words. May be empty for a silent span.</param>
/// <param name="StartSeconds">Offset from the start of the recording.</param>
/// <param name="EndSeconds">End offset from the start of the recording.</param>
/// <param name="AverageLogProbability">
/// Mean token log-probability. Whisper's most useful quality signal: values below about -1.0
/// usually mean the model is guessing, which is worth flagging rather than hiding.
/// </param>
/// <param name="NoSpeechProbability">
/// The model's own estimate that this span contains no speech at all. High values on a span
/// with text are the classic hallucination signature.
/// </param>
/// <param name="Speaker">
/// Who is speaking, when diarization ran. Null means unknown, which is deliberately distinct
/// from a single unnamed speaker: no diarization at all leaves every segment null.
/// </param>
/// <param name="SpeakerOverlapFraction">
/// How much of this segment the assigned speaker actually covers, from 0 to 1. Below 1 the
/// segment straddles a speaker change that could not be split without word-level timestamps.
/// </param>
public sealed record TranscriptSegment(
    string Text,
    double StartSeconds,
    double EndSeconds,
    double AverageLogProbability = 0.0,
    double NoSpeechProbability = 0.0,
    string? Speaker = null,
    double SpeakerOverlapFraction = 0.0)
{
    /// <summary>Duration of the segment in seconds.</summary>
    public double DurationSeconds => EndSeconds - StartSeconds;

    /// <summary>
    /// True when the segment looks like a hallucination: the model reports no speech, yet
    /// produced words anyway. Whisper does this reliably over long silences and music.
    /// </summary>
    public bool LooksHallucinated =>
        NoSpeechProbability > 0.6 && AverageLogProbability < -1.0 && !string.IsNullOrWhiteSpace(Text);

    /// <summary>
    /// True when the speaker label is a majority verdict rather than a clear one. Worth showing
    /// differently in a UI: the words are right, the attribution may not be.
    /// </summary>
    public bool SpeakerIsUncertain(double belowFraction = 0.6) =>
        Speaker is not null && SpeakerOverlapFraction < belowFraction;

    public override string ToString() =>
        Speaker is null
            ? $"[{StartSeconds:F1}s-{EndSeconds:F1}s] {Text}"
            : $"[{StartSeconds:F1}s-{EndSeconds:F1}s] {Speaker}: {Text}";
}

/// <summary>The result of transcribing one recording.</summary>
/// <param name="Segments">Time-ordered segments after stitching.</param>
/// <param name="Language">Detected or forced language code.</param>
public sealed record Transcript(IReadOnlyList<TranscriptSegment> Segments, string Language = "en")
{
    /// <summary>The whole transcript as flowing text.</summary>
    public string FullText => string.Join(" ", Segments.Select(s => s.Text.Trim()).Where(t => t.Length > 0));

    /// <summary>True when at least one segment carries a speaker label.</summary>
    public bool HasSpeakers => Segments.Any(s => s.Speaker is not null);

    /// <summary>Distinct speakers, in the order they first appear.</summary>
    public IReadOnlyList<string> Speakers =>
        [.. Segments.Select(s => s.Speaker).OfType<string>().Distinct()];

    public static Transcript Empty { get; } = new([], "en");
}
