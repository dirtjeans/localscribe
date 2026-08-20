namespace LocalScribe.Core.Audio;

/// <summary>
/// Kaldi-compatible log-mel filterbank features.
/// <para>
/// A second feature extractor, separate from <see cref="LogMelSpectrogram"/>, because speaker
/// embedding models want a different one and the difference is not a parameter. Whisper's front
/// end uses a Hann window, the Slaney mel scale, a log10 with clamping, and normalisation across
/// the whole 30-second window. Kaldi's uses a Povey window, the HTK mel scale, a natural log
/// with a floor, per-frame DC removal and pre-emphasis, and mean normalisation over the
/// utterance. Feeding one to a model trained on the other produces embeddings that are not
/// noise — they are plausible, stable, and wrong, which clusters into confident nonsense rather
/// than failing.
/// </para>
/// <para>
/// The defaults here are kaldi-native-fbank's, which is what WeSpeaker and the sherpa-onnx
/// speaker models were trained and are run against.
/// </para>
/// </summary>
public sealed class KaldiFbank
{
    /// <summary>Analysis window: 25 ms at 16 kHz.</summary>
    public const int FrameLength = 400;

    /// <summary>Hop between frames: 10 ms at 16 kHz.</summary>
    public const int FrameShift = 160;

    /// <summary>Filterbank channels. WeSpeaker's models take 80.</summary>
    public const int DefaultMelBins = 80;

    /// <summary>Kaldi rounds the FFT up to a power of two, so 400 samples become 512.</summary>
    private const int FftSize = 512;

    private const int SpectrumBins = (FftSize / 2) + 1;

    /// <summary>Kaldi's default. Removes the low-frequency tilt of speech before analysis.</summary>
    private const double PreEmphasis = 0.97;

    /// <summary>Kaldi's mel range. Everything below 20 Hz is rumble, not voice.</summary>
    private const double LowFrequency = 20.0;

    /// <summary>
    /// Kaldi works on waveforms in 16-bit integer range, not the [-1, 1] floats the rest of this
    /// app uses. WeSpeaker's models say so themselves — their metadata carries
    /// <c>normalize_samples 0</c>, meaning the samples were never scaled down during training.
    /// <para>
    /// A constant factor looks harmless because mean normalisation removes a constant offset,
    /// but it is not: the log is floored, and at [-1, 1] the quieter mel bands fall under the
    /// floor and flatten to the same value. What comes out is a vector that varies almost not at
    /// all between speakers, which is not obviously broken — it is a confident, stable, useless
    /// embedding.
    /// </para>
    /// </summary>
    private const double KaldiWaveformScale = 32768.0;

    private readonly double[] _window;
    private readonly double[][] _filters;

    /// <summary>Number of features per frame.</summary>
    public int MelBins { get; }

    public KaldiFbank(int sampleRate = PcmAudio.WhisperSampleRate, int melBins = DefaultMelBins)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(melBins, 1);

        MelBins = melBins;
        _window = BuildPoveyWindow(FrameLength);
        _filters = BuildMelFilterbank(sampleRate, melBins);
    }

    /// <summary>
    /// Computes features for a span of audio.
    /// </summary>
    /// <returns>
    /// Frames × <see cref="MelBins"/> in frame-major order, mean-normalised over the utterance,
    /// ready to reshape into the model's (batch, frames, bins) input.
    /// </returns>
    /// <param name="subtractMean">
    /// Cepstral mean normalisation. Whether a model wants it is a property of the model, not of
    /// the features: sherpa reads it from a <c>feature_normalize_type</c> entry in the model's
    /// own metadata, and the WeSpeaker embedding models carry none, meaning none.
    /// </param>
    public float[] Compute(ReadOnlySpan<float> samples, bool subtractMean = false)
    {
        // Kaldi snips edges: only whole windows count, and a clip shorter than one is no frames
        // rather than a padded one.
        var frameCount = samples.Length < FrameLength
            ? 0
            : ((samples.Length - FrameLength) / FrameShift) + 1;

        if (frameCount == 0)
        {
            return [];
        }

        var features = new float[frameCount * MelBins];

        var frame = new double[FrameLength];
        var real = new double[FftSize];
        var imaginary = new double[FftSize];
        var power = new double[SpectrumBins];

        for (var f = 0; f < frameCount; f++)
        {
            var start = f * FrameShift;

            for (var i = 0; i < FrameLength; i++)
            {
                frame[i] = samples[start + i] * KaldiWaveformScale;
            }

            ProcessWindow(frame);

            Array.Clear(real);
            Array.Clear(imaginary);
            frame.AsSpan().CopyTo(real);

            Fft.Transform(real, imaginary);

            for (var bin = 0; bin < SpectrumBins; bin++)
            {
                power[bin] = (real[bin] * real[bin]) + (imaginary[bin] * imaginary[bin]);
            }

            for (var mel = 0; mel < MelBins; mel++)
            {
                var filter = _filters[mel];
                var energy = 0.0;

                for (var bin = 0; bin < SpectrumBins; bin++)
                {
                    energy += filter[bin] * power[bin];
                }

                // Kaldi floors on float epsilon, not the smallest representable double. The
                // difference only shows on near-silent bands, which is where it matters.
                features[(f * MelBins) + mel] = (float)Math.Log(Math.Max(energy, float.Epsilon));
            }
        }

        if (subtractMean)
        {
            SubtractMean(features, frameCount);
        }

        return features;
    }

    /// <summary>
    /// Kaldi's per-frame preparation, in its order: remove the DC offset, pre-emphasise, then
    /// window. Doing any of these out of order changes the result.
    /// </summary>
    private void ProcessWindow(double[] frame)
    {
        var sum = 0.0;
        foreach (var value in frame)
        {
            sum += value;
        }

        var mean = sum / frame.Length;
        for (var i = 0; i < frame.Length; i++)
        {
            frame[i] -= mean;
        }

        // Backwards, so each sample is emphasised against its original predecessor rather than
        // the one just rewritten. The first sample uses itself, as Kaldi does.
        for (var i = frame.Length - 1; i > 0; i--)
        {
            frame[i] -= PreEmphasis * frame[i - 1];
        }

        frame[0] -= PreEmphasis * frame[0];

        for (var i = 0; i < frame.Length; i++)
        {
            frame[i] *= _window[i];
        }
    }

    /// <summary>
    /// Cepstral mean normalisation over the utterance. WeSpeaker applies this before the model,
    /// and without it every embedding carries the recording's channel as well as its speaker.
    /// </summary>
    private void SubtractMean(float[] features, int frameCount)
    {
        for (var mel = 0; mel < MelBins; mel++)
        {
            var total = 0.0;
            for (var f = 0; f < frameCount; f++)
            {
                total += features[(f * MelBins) + mel];
            }

            var mean = (float)(total / frameCount);
            for (var f = 0; f < frameCount; f++)
            {
                features[(f * MelBins) + mel] -= mean;
            }
        }
    }

    /// <summary>Hann raised to 0.85, which is what Kaldi calls the Povey window.</summary>
    private static double[] BuildPoveyWindow(int length)
    {
        var window = new double[length];

        for (var i = 0; i < length; i++)
        {
            var hann = 0.5 - (0.5 * Math.Cos(2.0 * Math.PI * i / (length - 1)));
            window[i] = Math.Pow(hann, 0.85);
        }

        return window;
    }

    /// <summary>
    /// Kaldi's triangular filters, spaced evenly on the HTK mel scale and left unnormalised.
    /// Slaney-style area normalisation, which Whisper's front end applies, would scale every
    /// high band differently and is exactly the kind of difference that survives as plausible
    /// output.
    /// </summary>
    private static double[][] BuildMelFilterbank(int sampleRate, int melBins)
    {
        var nyquist = sampleRate / 2.0;
        var lowMel = ToMel(LowFrequency);
        var highMel = ToMel(nyquist);
        var delta = (highMel - lowMel) / (melBins + 1);

        var binFrequencies = new double[SpectrumBins];
        for (var bin = 0; bin < SpectrumBins; bin++)
        {
            binFrequencies[bin] = bin * sampleRate / (double)FftSize;
        }

        var filters = new double[melBins][];

        for (var mel = 0; mel < melBins; mel++)
        {
            var left = lowMel + (mel * delta);
            var centre = left + delta;
            var right = left + (2 * delta);

            var filter = new double[SpectrumBins];

            for (var bin = 0; bin < SpectrumBins; bin++)
            {
                var melFrequency = ToMel(binFrequencies[bin]);

                if (melFrequency <= left || melFrequency >= right)
                {
                    continue;
                }

                filter[bin] = melFrequency <= centre
                    ? (melFrequency - left) / (centre - left)
                    : (right - melFrequency) / (right - centre);
            }

            filters[mel] = filter;
        }

        return filters;
    }

    /// <summary>The HTK mel scale Kaldi uses, which is not the Slaney one Whisper uses.</summary>
    private static double ToMel(double hertz) => 1127.0 * Math.Log(1.0 + (hertz / 700.0));
}
