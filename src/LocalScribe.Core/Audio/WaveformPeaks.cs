namespace LocalScribe.Core.Audio;

/// <summary>
/// Reduces a recording to a few hundred peaks, for drawing.
/// <para>
/// A waveform is the one view of a recording that shows structure without being read: where
/// someone spoke, where they stopped, how long the pauses ran. That makes it a far better thing
/// to scrub along than a featureless bar, because the target is usually "just after that gap"
/// rather than "four minutes eleven".
/// </para>
/// <para>
/// Peaks rather than averages. An average of a bucket of speech tends towards silence — signed
/// samples cancel — and the result is a flat ribbon that shows nothing. The largest excursion in
/// each bucket is what the eye reads as loudness.
/// </para>
/// </summary>
public static class WaveformPeaks
{
    /// <summary>
    /// Computes one peak per bucket, scaled so the loudest is 1.
    /// </summary>
    /// <param name="samples">The audio.</param>
    /// <param name="buckets">How many peaks to produce, typically a few hundred.</param>
    /// <returns>Values in [0, 1], one per bucket. Empty when there is no audio.</returns>
    public static float[] Compute(ReadOnlySpan<float> samples, int buckets)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(buckets, 1);

        if (samples.Length == 0)
        {
            return [];
        }

        var peaks = new float[buckets];
        var loudest = 0f;

        for (var bucket = 0; bucket < buckets; bucket++)
        {
            // Computed from the bucket index rather than by accumulating a width, so rounding
            // cannot drift and leave the last bucket short of the end of the recording.
            var start = (int)((long)bucket * samples.Length / buckets);
            var end = (int)((long)(bucket + 1) * samples.Length / buckets);

            if (end <= start)
            {
                end = Math.Min(start + 1, samples.Length);
            }

            var peak = 0f;

            for (var i = start; i < end; i++)
            {
                var magnitude = Math.Abs(samples[i]);
                if (magnitude > peak)
                {
                    peak = magnitude;
                }
            }

            peaks[bucket] = peak;

            if (peak > loudest)
            {
                loudest = peak;
            }
        }

        // Normalised against the recording's own loudest moment. A quiet recording should look
        // like a waveform, not like a flat line that happens to be technically accurate.
        if (loudest > 1e-6f)
        {
            for (var i = 0; i < peaks.Length; i++)
            {
                peaks[i] /= loudest;
            }
        }

        return peaks;
    }
}
