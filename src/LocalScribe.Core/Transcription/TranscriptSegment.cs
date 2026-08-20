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
/// Who was talking, or null when nobody has worked that out. Whisper does not answer this —
/// separating voices needs its own model — so the field exists to carry an answer from
/// elsewhere rather than to imply the transcriber produces one.
/// </param>
public sealed record TranscriptSegment(
    string Text,
    double StartSeconds,
    double EndSeconds,
    double AverageLogProbability = 0.0,
    double NoSpeechProbability = 0.0,
    string? Speaker = null)
{
    /// <summary>Duration of the segment in seconds.</summary>
    public double DurationSeconds => EndSeconds - StartSeconds;

    /// <summary>
    /// True when the segment looks like a hallucination: the model reports no speech, yet
    /// produced words anyway. Whisper does this reliably over long silences and music.
    /// </summary>
    public bool LooksHallucinated =>
        NoSpeechProbability > 0.6 && AverageLogProbability < -1.0 && !string.IsNullOrWhiteSpace(Text);

    public override string ToString() => $"[{StartSeconds:F1}s-{EndSeconds:F1}s] {Text}";
}

/// <summary>The result of transcribing one recording.</summary>
/// <param name="Segments">Time-ordered segments after stitching.</param>
/// <param name="Language">Detected or forced language code.</param>
public sealed record Transcript(IReadOnlyList<TranscriptSegment> Segments, string Language = "en")
{
    /// <summary>The whole transcript as flowing text.</summary>
    public string FullText => string.Join(" ", Segments.Select(s => s.Text.Trim()).Where(t => t.Length > 0));

    public static Transcript Empty { get; } = new([], "en");
}
