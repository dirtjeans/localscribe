using LocalScribe.Core.Transcription;

namespace LocalScribe.Onnx.Decoding;

/// <summary>
/// One window's worth of decoded tokens, with the model's confidence in each.
/// <para>
/// The two travel together because separating them loses the alignment, and the alignment is
/// the whole point: confidence is only useful attached to the words it is about.
/// </para>
/// </summary>
internal sealed record DecodedWindow(List<int> Tokens, List<float> LogProbabilities)
{
    public static DecodedWindow Empty { get; } = new([], []);

    /// <summary>Mean confidence over a run of tokens, or zero when there is nothing to average.</summary>
    public double MeanLogProbability(int from, int count)
    {
        if (count <= 0 || from < 0 || from >= LogProbabilities.Count)
        {
            return 0;
        }

        var last = Math.Min(from + count, LogProbabilities.Count);
        var total = 0.0;

        for (var i = from; i < last; i++)
        {
            total += LogProbabilities[i];
        }

        return total / (last - from);
    }
}

/// <summary>
/// Turns one window's mel spectrogram into token ids.
/// <para>
/// The two implementations differ in the shape of the decode loop, not merely in tensor names,
/// which is why this is an interface rather than a set of branches inside one method. Everything
/// either of them needs to know about the loaded model comes from
/// <see cref="WhisperModelSignature"/>.
/// </para>
/// </summary>
internal interface IWhisperDecodeStrategy
{
    /// <summary>Runs encoder and decoder over one 30-second window.</summary>
    /// <param name="mel">Flat mel data in mel-major order.</param>
    /// <param name="frames">Frame count, so the mel tensor can be shaped.</param>
    /// <param name="prompt">
    /// Tokens to condition on, or null. Used to show the model the shape of output wanted, which
    /// is the only lever available when it decides to return a window unpunctuated.
    /// </param>
    /// <returns>Token ids and how sure the model was of each, prompt excluded.</returns>
    DecodedWindow Decode(
        float[] mel,
        int frames,
        IReadOnlyList<int>? prompt,
        CancellationToken cancellationToken);
}
