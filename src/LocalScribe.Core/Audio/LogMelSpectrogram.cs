namespace LocalScribe.Core.Audio;

/// <summary>
/// Converts audio into the log-mel spectrogram Whisper's encoder expects.
/// <para>
/// The timing constants are not tunable. They match OpenAI's reference implementation exactly,
/// and an encoder graph is compiled for one input shape. Change any of them and the model still
/// runs, but it produces confident nonsense, which is a much harder bug to notice than a crash.
/// </para>
/// <para>
/// The band count is the one exception, because Whisper itself is not consistent about it: every
/// model through large-v2 uses 80, and large-v3 and its turbo derivative use 128. Getting this
/// wrong is precisely the confident-nonsense failure above, so the caller reads it off the
/// encoder's own input shape rather than assuming.
/// </para>
/// </summary>
public sealed class LogMelSpectrogram
{
    /// <summary>Window length in samples: 25 ms at 16 kHz.</summary>
    public const int FftSize = 400;

    /// <summary>Hop between frames: 10 ms at 16 kHz.</summary>
    public const int HopLength = 160;

    /// <summary>Band count for Whisper tiny through large-v2.</summary>
    public const int DefaultMelBands = 80;

    /// <summary>Band count for large-v3 and large-v3-turbo.</summary>
    public const int LargeV3MelBands = 128;

    /// <summary>Number of mel filterbank channels this instance produces.</summary>
    public int MelBands { get; }

    /// <summary>Frames produced from one 30-second window.</summary>
    public const int FramesPerWindow = 3000;

    private const int SpectrumBins = (FftSize / 2) + 1;

    private readonly double[] _window;
    private readonly double[][] _melFilters;

    /// <param name="sampleRate">Input rate. Whisper expects 16 kHz.</param>
    /// <param name="melBands">
    /// Filterbank channels. Must match the encoder being fed; see
    /// <see cref="DefaultMelBands"/> and <see cref="LargeV3MelBands"/>.
    /// </param>
    public LogMelSpectrogram(int sampleRate = PcmAudio.WhisperSampleRate, int melBands = DefaultMelBands)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(melBands, 1);

        MelBands = melBands;
        _window = BuildHannWindow(FftSize);
        _melFilters = BuildMelFilterbank(sampleRate, FftSize, melBands);
    }

    /// <summary>
    /// Computes the spectrogram for one chunk.
    /// </summary>
    /// <returns>
    /// A flat array of <see cref="MelBands"/> × frames values in mel-major order, ready to be
    /// reshaped into the encoder's (1, MelBands, N) input tensor.
    /// </returns>
    public float[] Compute(ReadOnlySpan<float> samples)
    {
        var frameCount = Math.Max(1, samples.Length / HopLength);
        var magnitudes = new double[frameCount][];

        var frameReal = new double[FftSize];
        var frameImaginary = new double[FftSize];

        for (var frame = 0; frame < frameCount; frame++)
        {
            // Reference implementations centre each frame on its hop position, which means
            // reflecting the signal at both edges rather than padding with zeros.
            var centre = frame * HopLength;
            for (var i = 0; i < FftSize; i++)
            {
                var index = centre + i - (FftSize / 2);
                frameReal[i] = ReflectSample(samples, index) * _window[i];
                frameImaginary[i] = 0;
            }

            Fft.Transform(frameReal, frameImaginary);

            var power = new double[SpectrumBins];
            for (var bin = 0; bin < SpectrumBins; bin++)
            {
                power[bin] = (frameReal[bin] * frameReal[bin]) + (frameImaginary[bin] * frameImaginary[bin]);
            }

            magnitudes[frame] = power;
        }

        return ApplyFilterbankAndNormalise(magnitudes, frameCount);
    }

    /// <summary>
    /// Applies the mel filters, takes the log, then clamps and rescales into roughly [-1, 1].
    /// The 8-decade floor stops silent passages from dominating the dynamic range.
    /// </summary>
    private float[] ApplyFilterbankAndNormalise(double[][] magnitudes, int frameCount)
    {
        var output = new float[MelBands * frameCount];
        var maximum = double.NegativeInfinity;

        for (var mel = 0; mel < MelBands; mel++)
        {
            var filter = _melFilters[mel];
            for (var frame = 0; frame < frameCount; frame++)
            {
                var power = magnitudes[frame];
                double sum = 0;
                for (var bin = 0; bin < SpectrumBins; bin++)
                {
                    sum += filter[bin] * power[bin];
                }

                var logValue = Math.Log10(Math.Max(sum, 1e-10));
                output[(mel * frameCount) + frame] = (float)logValue;
                if (logValue > maximum)
                {
                    maximum = logValue;
                }
            }
        }

        var floor = maximum - 8.0;
        for (var i = 0; i < output.Length; i++)
        {
            var clamped = Math.Max(output[i], floor);
            output[i] = (float)((clamped + 4.0) / 4.0);
        }

        return output;
    }

    /// <summary>
    /// Mirrors indices that fall outside the signal, matching the reflect padding the reference
    /// implementation uses. Out-of-range reads are common at the first and last few frames.
    /// </summary>
    private static float ReflectSample(ReadOnlySpan<float> samples, int index)
    {
        if (samples.Length == 0)
        {
            return 0f;
        }

        if (index < 0)
        {
            index = -index;
        }

        if (index >= samples.Length)
        {
            index = (2 * (samples.Length - 1)) - index;
        }

        return index >= 0 && index < samples.Length ? samples[index] : 0f;
    }

    /// <summary>Periodic Hann window, the variant used for spectrogram analysis.</summary>
    private static double[] BuildHannWindow(int size)
    {
        var window = new double[size];
        for (var i = 0; i < size; i++)
        {
            window[i] = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / size));
        }

        return window;
    }

    /// <summary>
    /// Builds Slaney-normalised triangular mel filters. Slaney rather than HTK: the two mel
    /// scales disagree above 1 kHz, and Whisper was trained on the Slaney variant.
    /// </summary>
    private static double[][] BuildMelFilterbank(int sampleRate, int fftSize, int melBands)
    {
        var maxFrequency = sampleRate / 2.0;
        var lowestMel = HzToMel(0.0);
        var highestMel = HzToMel(maxFrequency);

        // One point per band plus the two outer edges of the first and last triangles.
        var points = new double[melBands + 2];
        for (var i = 0; i < points.Length; i++)
        {
            var mel = lowestMel + ((highestMel - lowestMel) * i / (melBands + 1));
            points[i] = MelToHz(mel);
        }

        var binFrequencies = new double[SpectrumBins];
        for (var bin = 0; bin < SpectrumBins; bin++)
        {
            binFrequencies[bin] = bin * sampleRate / (double)fftSize;
        }

        var filters = new double[melBands][];
        for (var mel = 0; mel < melBands; mel++)
        {
            var left = points[mel];
            var centre = points[mel + 1];
            var right = points[mel + 2];

            // Slaney normalisation keeps each filter's area constant, so wide high-frequency
            // triangles do not swamp the narrow low-frequency ones.
            var scale = 2.0 / (right - left);

            var filter = new double[SpectrumBins];
            for (var bin = 0; bin < SpectrumBins; bin++)
            {
                var frequency = binFrequencies[bin];
                double weight;

                if (frequency < left || frequency > right)
                {
                    weight = 0;
                }
                else if (frequency <= centre)
                {
                    weight = centre > left ? (frequency - left) / (centre - left) : 0;
                }
                else
                {
                    weight = right > centre ? (right - frequency) / (right - centre) : 0;
                }

                filter[bin] = Math.Max(0, weight) * scale;
            }

            filters[mel] = filter;
        }

        return filters;
    }

    // The Slaney mel scale is linear below 1 kHz and logarithmic above it.
    private const double LinearRegionSlope = 200.0 / 3.0;
    private const double LogRegionStartHz = 1000.0;
    private const double LogRegionStartMel = LogRegionStartHz / LinearRegionSlope;
    private static readonly double LogStep = Math.Log(6.4) / 27.0;

    private static double HzToMel(double hz) =>
        hz >= LogRegionStartHz
            ? LogRegionStartMel + (Math.Log(hz / LogRegionStartHz) / LogStep)
            : hz / LinearRegionSlope;

    private static double MelToHz(double mel) =>
        mel >= LogRegionStartMel
            ? LogRegionStartHz * Math.Exp(LogStep * (mel - LogRegionStartMel))
            : mel * LinearRegionSlope;
}
