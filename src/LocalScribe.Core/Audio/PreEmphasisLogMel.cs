namespace LocalScribe.Core.Audio;

/// <summary>
/// The log-mel front end NVIDIA's speech models are trained behind: pre-emphasis, a 25 ms Hann
/// window in a 512-point transform every 10 ms, mel bands, and a bare logarithm.
/// <para>
/// A third featurizer beside <see cref="LogMelSpectrogram"/> and <see cref="KaldiFbank"/>, and
/// not out of taste. Each family of models hears through the exact arithmetic it was trained
/// on — Whisper clamps and rescales, Kaldi dithers and removes the mean, this one does neither
/// and leans on pre-emphasis instead — and a model fed its neighbour's features does not fail,
/// it just gets quietly worse. So the recipe is followed to the rounding: the pre-emphasised
/// sample is stored as a float before it is windowed, because the reference stores it so.
/// </para>
/// <para>
/// The mel filterbank is passed in rather than computed. It ships beside the weights as a
/// table, and a table cannot drift from what the model saw the way a second implementation of
/// the Slaney scale could.
/// </para>
/// </summary>
public sealed class PreEmphasisLogMel
{
    /// <summary>Samples between one frame and the next: ten milliseconds at 16 kHz.</summary>
    public const int HopSamples = 160;

    private const int TransformSize = 512;
    private const int WindowSamples = 400;
    private const int Bins = (TransformSize / 2) + 1;
    private const double PreEmphasis = 0.97;

    /// <summary>Keeps the logarithm finite on digital silence. The reference's own constant.</summary>
    private static readonly double Guard = Math.Pow(2, -24);

    private readonly float[] _filters;
    private readonly int[] _firstBin;
    private readonly int[] _endBin;
    private readonly double[] _window = new double[TransformSize];
    private readonly int[] _reversed = new int[TransformSize];
    private readonly double[] _cos = new double[TransformSize / 2];
    private readonly double[] _sin = new double[TransformSize / 2];

    /// <param name="filters">
    /// The mel filterbank, row-major: one row of 257 weights per band.
    /// </param>
    public PreEmphasisLogMel(float[] filters)
    {
        ArgumentNullException.ThrowIfNull(filters);

        if (filters.Length == 0 || filters.Length % Bins != 0)
        {
            throw new ArgumentException(
                $"A mel filterbank for a {TransformSize}-point transform has {Bins} weights per band; "
                + $"{filters.Length} values is not a whole number of bands.",
                nameof(filters));
        }

        _filters = filters;
        MelBins = filters.Length / Bins;

        // Mel filters are narrow triangles, so most of each row is zero. Remembering where the
        // weights are turns 257 multiplications per band into a dozen.
        _firstBin = new int[MelBins];
        _endBin = new int[MelBins];

        for (var band = 0; band < MelBins; band++)
        {
            _firstBin[band] = Bins;

            for (var bin = 0; bin < Bins; bin++)
            {
                if (filters[(band * Bins) + bin] != 0)
                {
                    _firstBin[band] = Math.Min(_firstBin[band], bin);
                    _endBin[band] = bin + 1;
                }
            }
        }

        // A symmetric Hann of 400, centred in the 512 — what torch.stft does with a window
        // shorter than its transform.
        var offset = (TransformSize - WindowSamples) / 2;
        for (var i = 0; i < WindowSamples; i++)
        {
            _window[offset + i] = 0.5 - (0.5 * Math.Cos(2 * Math.PI * i / (WindowSamples - 1)));
        }

        var levels = (int)Math.Log2(TransformSize);
        for (var i = 0; i < TransformSize; i++)
        {
            var reversed = 0;
            for (var bit = 0; bit < levels; bit++)
            {
                reversed |= ((i >> bit) & 1) << (levels - 1 - bit);
            }

            _reversed[i] = reversed;
        }

        for (var i = 0; i < TransformSize / 2; i++)
        {
            _cos[i] = Math.Cos(2 * Math.PI * i / TransformSize);
            _sin[i] = Math.Sin(2 * Math.PI * i / TransformSize);
        }
    }

    /// <summary>Mel bands per frame, as the filterbank defines them.</summary>
    public int MelBins { get; }

    /// <summary>Frames a recording yields. A trailing part-hop is dropped, as the reference drops it.</summary>
    public static int FrameCount(int samples) => samples / HopSamples;

    /// <summary>
    /// Features for frames <paramref name="first"/> up to but not including
    /// <paramref name="end"/>, row-major.
    /// <para>
    /// By range, so an hour of audio never needs an hour of features in memory at once: every
    /// frame depends only on the samples under its own window, and computing them a chunk at a
    /// time gives exactly the whole-file result.
    /// </para>
    /// </summary>
    public float[] Frames(ReadOnlySpan<float> samples, int first, int end)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(first);
        ArgumentOutOfRangeException.ThrowIfLessThan(end, first);

        var features = new float[(end - first) * MelBins];

        Span<double> real = stackalloc double[TransformSize];
        Span<double> imaginary = stackalloc double[TransformSize];
        Span<double> power = stackalloc double[Bins];

        for (var frame = first; frame < end; frame++)
        {
            // Centred: the frame's window straddles its nominal position, and whatever falls
            // off either end of the recording is silence.
            var start = (frame * HopSamples) - (TransformSize / 2);

            for (var i = 0; i < TransformSize; i++)
            {
                real[i] = Emphasised(samples, start + i) * _window[i];
                imaginary[i] = 0;
            }

            Transform(real, imaginary);

            for (var bin = 0; bin < Bins; bin++)
            {
                power[bin] = (real[bin] * real[bin]) + (imaginary[bin] * imaginary[bin]);
            }

            var row = (frame - first) * MelBins;

            for (var band = 0; band < MelBins; band++)
            {
                double energy = 0;
                var weights = band * Bins;

                for (var bin = _firstBin[band]; bin < _endBin[band]; bin++)
                {
                    energy += _filters[weights + bin] * power[bin];
                }

                features[row + band] = (float)Math.Log(energy + Guard);
            }
        }

        return features;
    }

    /// <summary>
    /// One sample after pre-emphasis: each sample less 0.97 of the one before, which tilts the
    /// spectrum toward the consonants. The first sample has no predecessor and passes through.
    /// </summary>
    private static double Emphasised(ReadOnlySpan<float> samples, int index)
    {
        if (index < 0 || index >= samples.Length)
        {
            return 0;
        }

        return index == 0
            ? samples[0]
            : (float)(samples[index] - (PreEmphasis * samples[index - 1]));
    }

    /// <summary>
    /// An in-place radix-2 transform with its tables built once. <see cref="Fft"/> handles any
    /// length and allocates as it recurses, which is right for Whisper's 400 and wasteful for a
    /// power of two run a hundred times a second of audio.
    /// </summary>
    private void Transform(Span<double> real, Span<double> imaginary)
    {
        for (var i = 0; i < TransformSize; i++)
        {
            var j = _reversed[i];
            if (j > i)
            {
                (real[i], real[j]) = (real[j], real[i]);
                (imaginary[i], imaginary[j]) = (imaginary[j], imaginary[i]);
            }
        }

        for (var size = 2; size <= TransformSize; size *= 2)
        {
            var half = size / 2;
            var step = TransformSize / size;

            for (var block = 0; block < TransformSize; block += size)
            {
                for (int j = 0, k = 0; j < half; j++, k += step)
                {
                    var a = block + j;
                    var b = a + half;

                    var turnedReal = (real[b] * _cos[k]) + (imaginary[b] * _sin[k]);
                    var turnedImaginary = (imaginary[b] * _cos[k]) - (real[b] * _sin[k]);

                    real[b] = real[a] - turnedReal;
                    imaginary[b] = imaginary[a] - turnedImaginary;
                    real[a] += turnedReal;
                    imaginary[a] += turnedImaginary;
                }
            }
        }
    }
}
