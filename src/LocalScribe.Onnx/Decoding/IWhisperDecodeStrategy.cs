using LocalScribe.Core.Transcription;

namespace LocalScribe.Onnx.Decoding;

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
    /// <returns>Token ids, prompt excluded.</returns>
    List<int> Decode(float[] mel, int frames, CancellationToken cancellationToken);
}
