using LocalScribe.Core.Audio;
using Xunit;

namespace LocalScribe.Core.Tests;

public class WaveformPeaksTests
{
    [Fact]
    public void ThereIsOneValuePerBucket() =>
        Assert.Equal(120, WaveformPeaks.Compute(new float[16000], 120).Length);

    [Fact]
    public void NoAudioGivesNoPeaks() =>
        Assert.Empty(WaveformPeaks.Compute([], 100));

    /// <summary>
    /// Scaled against the recording's own loudest moment. A quiet recording should look like a
    /// waveform rather than like a flat line that happens to be technically accurate.
    /// </summary>
    [Fact]
    public void TheLoudestMomentReachesTheTop()
    {
        var samples = new float[1000];
        samples[500] = 0.02f;   // quiet, but the loudest thing here

        var peaks = WaveformPeaks.Compute(samples, 10);

        Assert.Equal(1.0f, peaks.Max(), 3);
    }

    [Fact]
    public void SilenceIsFlat() =>
        Assert.All(WaveformPeaks.Compute(new float[1000], 10), p => Assert.Equal(0, p));

    /// <summary>
    /// Peaks, not averages. Signed samples of speech average out towards silence, and a
    /// waveform drawn from the mean is a flat ribbon that shows nothing.
    /// </summary>
    [Fact]
    public void ATonesShapeSurvives()
    {
        var samples = new float[16000];
        for (var i = 0; i < samples.Length; i++)
        {
            samples[i] = (float)Math.Sin(2 * Math.PI * 200 * i / 16000.0);
        }

        var peaks = WaveformPeaks.Compute(samples, 50);

        // Every bucket holds whole cycles, so every one should reach near full height. An
        // averaging implementation would return roughly zero here.
        Assert.All(peaks, p => Assert.True(p > 0.9f, $"bucket peaked at {p}"));
    }

    /// <summary>
    /// Bucket edges are computed from the index rather than accumulated, so rounding cannot
    /// leave the last bucket short of the end of the recording.
    /// </summary>
    [Fact]
    public void TheLastBucketReachesTheEnd()
    {
        var samples = new float[997];       // deliberately not divisible by the bucket count
        samples[^1] = 1.0f;

        var peaks = WaveformPeaks.Compute(samples, 10);

        Assert.Equal(1.0f, peaks[^1], 3);
    }

    [Fact]
    public void MoreBucketsThanSamplesStillWorks() =>
        Assert.Equal(100, WaveformPeaks.Compute(new float[10], 100).Length);
}
