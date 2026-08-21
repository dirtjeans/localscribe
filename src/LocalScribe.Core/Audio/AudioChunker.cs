namespace LocalScribe.Core.Audio;

/// <summary>One 30-second window handed to the encoder, and where it sat in the original audio.</summary>
/// <param name="Samples">Exactly 30 seconds of samples, zero-padded when the source ran out.</param>
/// <param name="StartSeconds">Offset of this window within the full recording.</param>
/// <param name="ContentSeconds">
/// How much of the window is real audio rather than padding. The last chunk of a file is
/// usually part padding, and timestamps past this point should be discarded.
/// </param>
public sealed record AudioChunk(float[] Samples, double StartSeconds, double ContentSeconds)
{
    /// <summary>True when the chunk is mostly padding, which tends to make Whisper hallucinate.</summary>
    public bool IsMostlyPadding => ContentSeconds < 1.0;
}

/// <summary>
/// Splits audio into the fixed 30-second windows the Whisper encoder requires.
/// <para>
/// Windows overlap slightly. Without overlap, a word landing on a boundary gets cut in half and
/// both halves transcribe badly; with it, the word appears whole in at least one window and the
/// stitcher drops the duplicate.
/// </para>
/// </summary>
public sealed class AudioChunker
{
    /// <summary>The encoder's input length. Not adjustable: the QNN graph is compiled for it.</summary>
    public const double WindowSeconds = 30.0;

    private readonly double _overlapSeconds;
    private readonly int _sampleRate;

    /// <param name="overlapSeconds">
    /// How much each window repeats from the one before. Two seconds comfortably covers a
    /// spoken word plus its surrounding pause.
    /// </param>
    /// <param name="sampleRate">Sample rate of the audio to be chunked.</param>
    public AudioChunker(double overlapSeconds = 2.0, int sampleRate = PcmAudio.WhisperSampleRate)
    {
        if (overlapSeconds < 0 || overlapSeconds >= WindowSeconds)
        {
            throw new ArgumentOutOfRangeException(
                nameof(overlapSeconds),
                overlapSeconds,
                $"Overlap must be at least 0 and less than the {WindowSeconds}s window.");
        }

        _overlapSeconds = overlapSeconds;
        _sampleRate = sampleRate > 0
            ? sampleRate
            : throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Sample rate must be positive.");
    }

    private int WindowSamples => (int)(WindowSeconds * _sampleRate);

    private int AdvanceSamples => Math.Max(1, WindowSamples - (int)(_overlapSeconds * _sampleRate));

    /// <summary>
    /// One window starting at a given offset.
    /// <para>
    /// Exposed so a caller can decide where the next window begins rather than taking a fixed
    /// stride. Whisper routinely stops transcribing before the end of a window — it prefers to
    /// finish on a sentence rather than cut one — and when it stops further back than the fixed
    /// overlap, the audio in between is transcribed by nobody.
    /// </para>
    /// </summary>
    public AudioChunk WindowAt(PcmAudio audio, double startSeconds)
    {
        ArgumentNullException.ThrowIfNull(audio);

        var start = Math.Clamp((int)(startSeconds * _sampleRate), 0, Math.Max(0, audio.Samples.Length));
        var available = Math.Max(0, Math.Min(WindowSamples, audio.Samples.Length - start));

        var window = new float[WindowSamples];
        if (available > 0)
        {
            Array.Copy(audio.Samples, start, window, 0, available);
        }

        return new AudioChunk(window, start / (double)_sampleRate, available / (double)_sampleRate);
    }

    /// <summary>How far a window steps when nothing better is known.</summary>
    public double DefaultAdvanceSeconds => AdvanceSamples / (double)_sampleRate;

    /// <summary>How far back to step from the end of transcribed speech, to avoid clipping it.</summary>
    public double OverlapSeconds => _overlapSeconds;

    /// <summary>
    /// Cuts audio into encoder-sized windows. Silence-only trailing padding is included so the
    /// tensor shape stays fixed, but is reported through <see cref="AudioChunk.ContentSeconds"/>.
    /// </summary>
    public IReadOnlyList<AudioChunk> Chunk(PcmAudio audio)
    {
        ArgumentNullException.ThrowIfNull(audio);

        var samples = audio.Samples;
        var chunks = new List<AudioChunk>();

        if (samples.Length == 0)
        {
            return chunks;
        }

        var windowSamples = WindowSamples;
        var advance = AdvanceSamples;

        for (var start = 0; start < samples.Length; start += advance)
        {
            var available = Math.Min(windowSamples, samples.Length - start);
            var window = new float[windowSamples];
            Array.Copy(samples, start, window, 0, available);

            chunks.Add(new AudioChunk(
                window,
                start / (double)_sampleRate,
                available / (double)_sampleRate));

            // Once a window reaches the end of the audio, further windows would be pure padding.
            if (start + available >= samples.Length)
            {
                break;
            }
        }

        return chunks;
    }
}
