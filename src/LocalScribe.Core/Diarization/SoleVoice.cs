namespace LocalScribe.Core.Diarization;

/// <summary>
/// The one person heard across a stretch of audio, when there is exactly one.
/// <para>
/// A line with no word times cannot be divided between speakers, and labelling it by whoever
/// held most of it would put one name on a conversation. But a line that only one person spoke
/// needs no dividing: the diarizer already says who it was, and the label it gets now is the
/// label it will still have when the words are timed. On Windows, whose transcriber reports no
/// word times of its own, that is the difference between labels arriving when the diarizer
/// answers and labels waiting for the word-timing scan.
/// </para>
/// </summary>
public static class SoleVoice
{
    /// <summary>Speech from anyone else below this is a cough or a flicker, not a second person.</summary>
    public const double StraySeconds = 0.5;

    /// <summary>Less speech than this is too little to be sure whose stretch it is.</summary>
    public const double LeastSeconds = 1.0;

    /// <summary>
    /// The label of the only speaker heard between <paramref name="start"/> and
    /// <paramref name="end"/>, or null when nobody, or more than one person, was.
    /// </summary>
    public static string? Of(IReadOnlyList<SpeakerTurn> turns, double start, double end)
    {
        ArgumentNullException.ThrowIfNull(turns);

        if (end <= start)
        {
            return null;
        }

        var heard = new Dictionary<int, double>();

        foreach (var turn in turns)
        {
            var seconds = turn.OverlapWith(start, end);
            if (seconds > 0)
            {
                heard[turn.Speaker] = heard.GetValueOrDefault(turn.Speaker) + seconds;
            }
        }

        if (heard.Count == 0)
        {
            return null;
        }

        var (speaker, most) = heard.MaxBy(pair => pair.Value);
        var others = heard.Where(pair => pair.Key != speaker).Sum(pair => pair.Value);

        return most >= LeastSeconds && others < StraySeconds
            ? new SpeakerTurn(speaker, start, end).Label
            : null;
    }
}
