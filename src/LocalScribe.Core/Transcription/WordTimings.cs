using LocalScribe.Core.Audio;

namespace LocalScribe.Core.Transcription;

/// <summary>
/// Works out roughly when each word was said, so a word can be clicked and heard.
/// <para>
/// Whisper times whole segments, not words. The proper way to get words is to align them against
/// the decoder's cross-attention, which is what OpenAI's implementation does — but the exported
/// graphs this app runs emit only logits and key-value caches. The attention weights are computed
/// inside the decoder and never surface, so that route is closed.
/// </para>
/// <para>
/// What is left is to share out the segment's span between its words. Sharing it evenly is bad
/// in a specific and avoidable way: a segment usually contains pauses, and even division puts
/// words inside them, so the error accumulates across the segment and the last word can be
/// seconds adrift. But the pauses are plainly visible in the audio. Allocating by loudness
/// instead of by clock skips the silence, and every word lands in a stretch where somebody was
/// actually talking.
/// </para>
/// <para>
/// It is an estimate and is not sold as more. Clicking a word seeks to within a word or so,
/// which is what the feature is for; nobody should read these as measurements.
/// </para>
/// </summary>
public static class WordTimings
{
    /// <param name="Text">The word as it appears, punctuation and all.</param>
    /// <param name="StartSeconds">Where to seek to hear it.</param>
    public sealed record Word(string Text, double StartSeconds, double EndSeconds)
    {
        /// <summary>Position of the word's first character in the segment's text.</summary>
        public int Offset { get; init; }
    }

    /// <summary>
    /// Times the words of one segment against the recording.
    /// </summary>
    /// <param name="audio">The recording, for finding where the speech actually is.</param>
    /// <param name="segment">The segment to break up.</param>
    public static IReadOnlyList<Word> For(PcmAudio audio, TranscriptSegment segment)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(segment);

        var words = Split(segment.Text);
        if (words.Count == 0)
        {
            return [];
        }

        var start = segment.StartSeconds;
        var end = Math.Max(segment.EndSeconds, start);

        if (words.Count == 1 || end - start <= 0)
        {
            return [new Word(words[0].Text, start, end) { Offset = words[0].Offset }];
        }

        var loudness = Loudness(audio, start, end);

        // How much of the segment's speech each word accounts for, by length. Longer words take
        // longer to say — crudely true, and the only thing available without the alignment.
        var weights = words.Select(word => (double)Math.Max(1, word.Text.Length)).ToArray();
        var total = weights.Sum();

        var timed = new List<Word>(words.Count);
        var consumed = 0.0;

        for (var i = 0; i < words.Count; i++)
        {
            var from = SecondsAt(loudness, consumed / total, start, end);
            consumed += weights[i];
            var to = SecondsAt(loudness, consumed / total, start, end);

            timed.Add(new Word(words[i].Text, from, Math.Max(to, from)) { Offset = words[i].Offset });
        }

        return timed;
    }

    /// <summary>
    /// Where a given fraction of the segment's speech has been heard, in seconds.
    /// <para>
    /// The inverse of the cumulative loudness curve. A pause contributes nothing to it, so no
    /// word is ever placed inside one.
    /// </para>
    /// </summary>
    private static double SecondsAt(double[] cumulative, double fraction, double start, double end)
    {
        if (cumulative.Length < 2)
        {
            return start + ((end - start) * fraction);
        }

        var target = Math.Clamp(fraction, 0, 1) * cumulative[^1];

        // Binary search for the first frame past the target.
        var low = 0;
        var high = cumulative.Length - 1;

        while (low < high)
        {
            var middle = (low + high) / 2;

            if (cumulative[middle] < target)
            {
                low = middle + 1;
            }
            else
            {
                high = middle;
            }
        }

        var at = low / (double)(cumulative.Length - 1);

        return start + ((end - start) * at);
    }

    /// <summary>
    /// Cumulative loudness across the segment, one entry per short frame.
    /// <para>
    /// Root mean square rather than raw amplitude, and never zero: a segment of pure silence
    /// must still divide into words rather than piling them all at one instant.
    /// </para>
    /// </summary>
    private static double[] Loudness(PcmAudio audio, double start, double end)
    {
        var first = Math.Clamp((int)(start * audio.SampleRate), 0, audio.Samples.Length);
        var last = Math.Clamp((int)(end * audio.SampleRate), first, audio.Samples.Length);

        var frame = Math.Max(1, audio.SampleRate / FramesPerSecond);
        var count = Math.Max(2, (last - first) / frame);

        var cumulative = new double[count + 1];

        for (var i = 0; i < count; i++)
        {
            var from = first + (i * frame);
            var to = Math.Min(from + frame, last);

            var sum = 0.0;
            for (var s = from; s < to; s++)
            {
                sum += audio.Samples[s] * (double)audio.Samples[s];
            }

            var rms = to > from ? Math.Sqrt(sum / (to - from)) : 0;

            // A floor, so silence still advances the clock a little. Without it a long pause
            // would be crossed instantaneously and the words either side would collide.
            cumulative[i + 1] = cumulative[i] + Math.Max(rms, QuietestFrame);
        }

        return cumulative;
    }

    /// <summary>Words with their positions in the original text, for highlighting.</summary>
    private static List<Word> Split(string text)
    {
        var words = new List<Word>();
        var at = 0;

        while (at < text.Length)
        {
            while (at < text.Length && char.IsWhiteSpace(text[at]))
            {
                at++;
            }

            if (at >= text.Length)
            {
                break;
            }

            var from = at;
            while (at < text.Length && !char.IsWhiteSpace(text[at]))
            {
                at++;
            }

            words.Add(new Word(text[from..at], 0, 0) { Offset = from });
        }

        return words;
    }

    /// <summary>Frames per second of loudness. Fine enough to resolve a syllable.</summary>
    private const int FramesPerSecond = 100;

    /// <summary>
    /// The weight a silent frame still carries. Small enough that pauses are mostly skipped,
    /// large enough that a silent stretch is not crossed in an instant.
    /// </summary>
    private const double QuietestFrame = 0.002;
}
