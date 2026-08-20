namespace LocalScribe.Core.Audio;

/// <summary>
/// A mixed-radix complex FFT that handles any transform length.
/// <para>
/// Whisper's window is 400 samples, which is not a power of two, so a plain radix-2 routine
/// cannot be used directly. This splits out factors of two recursively and falls back to a
/// direct transform once the remaining length is odd. For 400 that means recursing
/// 400 → 200 → 100 → 50 → 25 and running a 25-point direct transform at the bottom, which is
/// far cheaper than the 400-point one it replaces.
/// </para>
/// </summary>
internal static class Fft
{
    /// <summary>
    /// Transforms in place. <paramref name="real"/> and <paramref name="imaginary"/> must be
    /// the same length.
    /// </summary>
    public static void Transform(Span<double> real, Span<double> imaginary)
    {
        if (real.Length != imaginary.Length)
        {
            throw new ArgumentException("Real and imaginary parts must have the same length.", nameof(imaginary));
        }

        var n = real.Length;
        if (n <= 1)
        {
            return;
        }

        if (n % 2 != 0)
        {
            DirectTransform(real, imaginary);
            return;
        }

        var half = n / 2;

        // Deinterleave into even- and odd-indexed halves.
        Span<double> evenReal = new double[half];
        Span<double> evenImaginary = new double[half];
        Span<double> oddReal = new double[half];
        Span<double> oddImaginary = new double[half];

        for (var i = 0; i < half; i++)
        {
            evenReal[i] = real[2 * i];
            evenImaginary[i] = imaginary[2 * i];
            oddReal[i] = real[(2 * i) + 1];
            oddImaginary[i] = imaginary[(2 * i) + 1];
        }

        Transform(evenReal, evenImaginary);
        Transform(oddReal, oddImaginary);

        for (var k = 0; k < half; k++)
        {
            var angle = -2.0 * Math.PI * k / n;
            var cos = Math.Cos(angle);
            var sin = Math.Sin(angle);

            var twiddleReal = (cos * oddReal[k]) - (sin * oddImaginary[k]);
            var twiddleImaginary = (cos * oddImaginary[k]) + (sin * oddReal[k]);

            real[k] = evenReal[k] + twiddleReal;
            imaginary[k] = evenImaginary[k] + twiddleImaginary;
            real[k + half] = evenReal[k] - twiddleReal;
            imaginary[k + half] = evenImaginary[k] - twiddleImaginary;
        }
    }

    /// <summary>The textbook O(n²) transform, used only for short odd lengths.</summary>
    private static void DirectTransform(Span<double> real, Span<double> imaginary)
    {
        var n = real.Length;
        var outReal = new double[n];
        var outImaginary = new double[n];

        for (var k = 0; k < n; k++)
        {
            double sumReal = 0;
            double sumImaginary = 0;

            for (var t = 0; t < n; t++)
            {
                var angle = -2.0 * Math.PI * t * k / n;
                var cos = Math.Cos(angle);
                var sin = Math.Sin(angle);
                sumReal += (real[t] * cos) - (imaginary[t] * sin);
                sumImaginary += (real[t] * sin) + (imaginary[t] * cos);
            }

            outReal[k] = sumReal;
            outImaginary[k] = sumImaginary;
        }

        outReal.CopyTo(real);
        outImaginary.CopyTo(imaginary);
    }
}
