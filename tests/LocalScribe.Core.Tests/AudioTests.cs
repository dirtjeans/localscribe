using LocalScribe.Core.Audio;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// Signal-processing bugs are the dangerous kind here: a wrong mel filterbank does not crash,
/// it just makes Whisper produce fluent nonsense. So the maths is checked against first
/// principles rather than against a recorded snapshot of its own output.
/// </summary>
public sealed class FftTests
{
    /// <summary>The definition of the DFT, used as an independent oracle for the fast version.</summary>
    private static (double[] Real, double[] Imaginary) NaiveDft(double[] input)
    {
        var n = input.Length;
        var real = new double[n];
        var imaginary = new double[n];

        for (var k = 0; k < n; k++)
        {
            for (var t = 0; t < n; t++)
            {
                var angle = -2.0 * Math.PI * t * k / n;
                real[k] += input[t] * Math.Cos(angle);
                imaginary[k] += input[t] * Math.Sin(angle);
            }
        }

        return (real, imaginary);
    }

    [Theory]
    [InlineData(8)]    // pure power of two
    [InlineData(25)]   // odd, hits the direct path
    [InlineData(50)]   // one factor of two above an odd base
    [InlineData(400)]  // the length Whisper actually uses
    public void FastTransformMatchesTheDefinition(int length)
    {
        var random = new Random(Seed: 1234);
        var input = new double[length];
        for (var i = 0; i < length; i++)
        {
            input[i] = (random.NextDouble() * 2.0) - 1.0;
        }

        var (expectedReal, expectedImaginary) = NaiveDft(input);

        var actualReal = (double[])input.Clone();
        var actualImaginary = new double[length];
        Fft.Transform(actualReal, actualImaginary);

        for (var k = 0; k < length; k++)
        {
            Assert.Equal(expectedReal[k], actualReal[k], precision: 8);
            Assert.Equal(expectedImaginary[k], actualImaginary[k], precision: 8);
        }
    }

    [Fact]
    public void ConstantSignalHasAllEnergyInTheZeroBin()
    {
        var real = Enumerable.Repeat(1.0, 400).ToArray();
        var imaginary = new double[400];

        Fft.Transform(real, imaginary);

        Assert.Equal(400.0, real[0], precision: 6);
        for (var k = 1; k < 400; k++)
        {
            Assert.Equal(0.0, Math.Sqrt((real[k] * real[k]) + (imaginary[k] * imaginary[k])), precision: 6);
        }
    }

    [Fact]
    public void MismatchedLengthsAreRejected()
    {
        Assert.Throws<ArgumentException>(() => Fft.Transform(new double[4], new double[5]));
    }
}

public sealed class LogMelSpectrogramTests
{
    private const int SampleRate = PcmAudio.WhisperSampleRate;

    private static float[] Sine(double frequencyHz, double seconds, double amplitude = 0.5)
    {
        var samples = new float[(int)(seconds * SampleRate)];
        for (var i = 0; i < samples.Length; i++)
        {
            samples[i] = (float)(amplitude * Math.Sin(2.0 * Math.PI * frequencyHz * i / SampleRate));
        }

        return samples;
    }

    [Fact]
    public void ThirtySecondWindowProducesTheShapeTheEncoderExpects()
    {
        // The QNN encoder graph is compiled for (1, 80, 3000). Anything else fails to bind.
        var spectrogram = new LogMelSpectrogram();
        var output = spectrogram.Compute(Sine(440, 30.0));

        Assert.Equal(LogMelSpectrogram.MelBands * LogMelSpectrogram.FramesPerWindow, output.Length);
    }

    [Fact]
    public void OutputStaysInTheNormalisedRange()
    {
        var spectrogram = new LogMelSpectrogram();
        var output = spectrogram.Compute(Sine(1000, 2.0));

        Assert.All(output, value =>
        {
            Assert.True(float.IsFinite(value), "Spectrogram produced a non-finite value.");
            Assert.InRange(value, -1.5f, 1.5f);
        });
    }

    [Fact]
    public void ALowToneLandsInALowMelBand()
    {
        var spectrogram = new LogMelSpectrogram();
        var frames = 200;
        var output = spectrogram.Compute(Sine(440, frames * LogMelSpectrogram.HopLength / (double)SampleRate));
        var actualFrames = output.Length / LogMelSpectrogram.MelBands;

        var peakBand = PeakBand(output, actualFrames);

        // 440 Hz sits in the linear region of the mel scale, around band 10 of 80.
        Assert.InRange(peakBand, 5, 18);
    }

    [Fact]
    public void AHighToneLandsInAHighMelBand()
    {
        var spectrogram = new LogMelSpectrogram();
        var output = spectrogram.Compute(Sine(6000, 2.0));
        var actualFrames = output.Length / LogMelSpectrogram.MelBands;

        var peakBand = PeakBand(output, actualFrames);

        Assert.True(peakBand > 55, $"Expected a 6 kHz tone above band 55 but the peak was band {peakBand}.");
    }

    [Fact]
    public void TonesAreOrderedByFrequencyAcrossTheFilterbank()
    {
        // The single strongest check that the mel scale is not inverted or mis-scaled.
        var spectrogram = new LogMelSpectrogram();
        var previousBand = -1;

        foreach (var frequency in new double[] { 200, 500, 1000, 2000, 4000, 7000 })
        {
            var output = spectrogram.Compute(Sine(frequency, 1.0));
            var band = PeakBand(output, output.Length / LogMelSpectrogram.MelBands);

            Assert.True(
                band > previousBand,
                $"{frequency} Hz peaked at band {band}, which is not above the previous {previousBand}.");
            previousBand = band;
        }
    }

    [Fact]
    public void SilenceProducesFiniteOutputRatherThanNegativeInfinity()
    {
        // log(0) is the classic way this code explodes. The floor exists to prevent it.
        var spectrogram = new LogMelSpectrogram();
        var output = spectrogram.Compute(new float[SampleRate]);

        Assert.All(output, value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void EmptyInputDoesNotThrow()
    {
        var spectrogram = new LogMelSpectrogram();
        var output = spectrogram.Compute([]);

        Assert.All(output, value => Assert.True(float.IsFinite(value)));
    }

    /// <summary>Finds the mel band holding the most energy, averaged over the middle of the clip.</summary>
    private static int PeakBand(float[] spectrogram, int frames)
    {
        var best = 0;
        var bestEnergy = float.NegativeInfinity;

        // Skip the edges, where reflect padding distorts the first and last few frames.
        var from = Math.Min(5, frames / 4);
        var to = Math.Max(from + 1, frames - from);

        for (var band = 0; band < LogMelSpectrogram.MelBands; band++)
        {
            var energy = 0f;
            for (var frame = from; frame < to; frame++)
            {
                energy += spectrogram[(band * frames) + frame];
            }

            if (energy > bestEnergy)
            {
                bestEnergy = energy;
                best = band;
            }
        }

        return best;
    }
}

public sealed class AudioChunkerTests
{
    private static PcmAudio Audio(double seconds) =>
        new(new float[(int)(seconds * PcmAudio.WhisperSampleRate)]);

    [Fact]
    public void ShortAudioBecomesOnePaddedWindow()
    {
        var chunks = new AudioChunker().Chunk(Audio(5));

        var chunk = Assert.Single(chunks);
        Assert.Equal(30 * PcmAudio.WhisperSampleRate, chunk.Samples.Length);
        Assert.Equal(5.0, chunk.ContentSeconds, precision: 3);
        Assert.Equal(0.0, chunk.StartSeconds);
    }

    [Fact]
    public void EmptyAudioProducesNoChunks()
    {
        Assert.Empty(new AudioChunker().Chunk(new PcmAudio([])));
    }

    [Fact]
    public void WindowsOverlapByTheRequestedAmount()
    {
        var chunks = new AudioChunker(overlapSeconds: 2.0).Chunk(Audio(90));

        Assert.True(chunks.Count > 1);
        Assert.Equal(28.0, chunks[1].StartSeconds, precision: 3);
        Assert.Equal(56.0, chunks[2].StartSeconds, precision: 3);
    }

    [Fact]
    public void EveryWindowIsExactlyEncoderSized()
    {
        var chunks = new AudioChunker().Chunk(Audio(100));

        Assert.All(chunks, chunk => Assert.Equal(30 * PcmAudio.WhisperSampleRate, chunk.Samples.Length));
    }

    [Fact]
    public void ChunkingCoversTheWholeRecording()
    {
        var chunks = new AudioChunker().Chunk(Audio(95));
        var last = chunks[^1];

        Assert.True(
            last.StartSeconds + last.ContentSeconds >= 94.9,
            "The final window must reach the end of the audio.");
    }

    [Fact]
    public void PaddingOnlyWindowsAreFlagged()
    {
        // Whisper invents text when handed silence, so these are worth skipping entirely.
        Assert.True(new AudioChunk(new float[480000], 30, 0.4).IsMostlyPadding);
        Assert.False(new AudioChunk(new float[480000], 0, 12.0).IsMostlyPadding);
    }

    [Theory]
    [InlineData(-1.0)]
    [InlineData(30.0)]
    [InlineData(45.0)]
    public void NonsensicalOverlapIsRejected(double overlap)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new AudioChunker(overlap));
    }

    [Fact]
    public void WrongSampleRateIsCaughtBeforeItReachesTheModel()
    {
        var audio = new PcmAudio(new float[1000], SampleRate: 44_100);

        var exception = Assert.Throws<InvalidOperationException>(audio.EnsureWhisperFormat);
        Assert.Contains("16000", exception.Message, StringComparison.Ordinal);
    }
}
