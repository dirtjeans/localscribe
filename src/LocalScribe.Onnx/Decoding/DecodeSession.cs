namespace LocalScribe.Onnx.Decoding;

/// <summary>
/// What one decode learns that the next should not have to work out again.
/// <para>
/// Only the language so far, and that is enough to matter. Whisper is asked to name the language
/// at the start of every decode, and left to itself it will answer differently on different
/// passes over the same audio — which is how a live transcript of English speech opened with
/// "Gracias." and how the same sentence came back properly punctuated on one pass and as a bare
/// lowercase run of words on the next. The language of a recording does not change between
/// passes, so the answer is settled once and then asserted.
/// </para>
/// </summary>
internal sealed class DecodeSession
{
    /// <summary>
    /// The detected language token id, -1 when the model has no language tokens, or null when
    /// detection has not run yet.
    /// </summary>
    public int? Language { get; set; }
}
