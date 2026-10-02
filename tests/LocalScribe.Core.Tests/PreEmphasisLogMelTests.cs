using LocalScribe.Core.Audio;
using Xunit;

namespace LocalScribe.Core.Tests;

public class PreEmphasisLogMelTests
{
    private const int Bins = 257;
    private const int SampleRate = 16_000;

    /// <summary>
    /// Four flat bands of 2 kHz each. Not a mel scale, and it need not be: the scale ships as a
    /// table beside the weights. What is under test is everything around the table.
    /// </summary>
    private static float[] FourBands()
    {
        var filters = new float[4 * Bins];

        for (var bin = 0; bin < Bins - 1; bin++)
        {
            filters[((bin / 64) * Bins) + bin] = 1f;
        }

        return filters;
    }

    private static float[] Tone(double hertz, int samples, float amplitude = 0.5f) =>
        [.. Enumerable.Range(0, samples).Select(i => amplitude * (float)Math.Sin(2 * Math.PI * hertz * i / SampleRate))];

    [Fact]
    public void ATrailingPartHopIsDropped()
    {
        Assert.Equal(100, PreEmphasisLogMel.FrameCount(SampleRate));
        Assert.Equal(100, PreEmphasisLogMel.FrameCount(SampleRate + 159));
        Assert.Equal(101, PreEmphasisLogMel.FrameCount(SampleRate + 160));
    }

    [Fact]
    public void DigitalSilenceIsFiniteAndFlat()
    {
        var features = new PreEmphasisLogMel(FourBands()).Frames(new float[SampleRate], 0, 100);

        Assert.All(features, value => Assert.Equal(Math.Log(Math.Pow(2, -24)), value, 4));
    }

    [Theory]
    [InlineData(1000, 0)]
    [InlineData(3000, 1)]
    [InlineData(5000, 2)]
    [InlineData(7000, 3)]
    public void AToneLandsInTheBandThatHoldsItsFrequency(double hertz, int band)
    {
        var mel = new PreEmphasisLogMel(FourBands());
        var features = mel.Frames(Tone(hertz, SampleRate), 50, 51);

        Assert.Equal(band, Array.IndexOf(features, features.Max()));
    }

    [Fact]
    public void ARangeIsExactlyThatPartOfTheWholeFile()
    {
        var mel = new PreEmphasisLogMel(FourBands());
        var samples = Tone(440, SampleRate).Zip(Tone(3100, SampleRate), (a, b) => a + b).ToArray();

        var whole = mel.Frames(samples, 0, 100);
        var part = mel.Frames(samples, 37, 61);

        Assert.Equal(whole.AsSpan(37 * mel.MelBins, 24 * mel.MelBins).ToArray(), part);
    }

    [Fact]
    public void PreEmphasisTiltsTowardTheHighFrequencies()
    {
        var mel = new PreEmphasisLogMel(FourBands());

        // Two tones of the same amplitude. A flat front end would report them equally loud;
        // this one subtracts most of each sample's predecessor, which cancels a slow wave far
        // more than a fast one.
        var low = mel.Frames(Tone(300, SampleRate), 50, 51).Max();
        var high = mel.Frames(Tone(6000, SampleRate), 50, 51).Max();

        Assert.True(high - low > Math.Log(10), $"low {low:F2}, high {high:F2}");
    }

    [Fact]
    public void FramesPastTheEndOfTheRecordingHearSilence()
    {
        var mel = new PreEmphasisLogMel(FourBands());
        var features = mel.Frames(Tone(1000, 1600), 20, 21);

        Assert.All(features, value => Assert.Equal(Math.Log(Math.Pow(2, -24)), value, 4));
    }

    [Fact]
    public void AFilterbankThatIsNotWholeBandsIsRefused()
    {
        Assert.Throws<ArgumentException>(() => new PreEmphasisLogMel(new float[Bins + 1]));
    }
}
