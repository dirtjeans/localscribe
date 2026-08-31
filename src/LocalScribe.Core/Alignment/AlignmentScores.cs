namespace LocalScribe.Core.Alignment;

/// <summary>
/// What the recogniser thought of every frame of a recording, kept so that words can be placed
/// against it later.
/// <para>
/// Alignment has two halves with very different costs. Running the network over the audio is
/// nearly all of the work and depends on nothing but the audio. Placing known words onto the
/// frames it produced is a Viterbi pass over a few hundred frames and costs almost nothing.
/// Splitting them lets the expensive half run early — beside cleanup and speaker detection,
/// which is where it was always wanted — while the cheap half waits for the text those two
/// leave behind.
/// </para>
/// <para>
/// The frames are a fixed grid over the whole recording rather than one run per segment, because
/// segment boundaries are not known when the scan happens and will not survive cleanup and
/// attribution anyway. A grid is indifferent to where the segments end up.
/// </para>
/// <para>
/// Thirty-one tokens at twenty milliseconds is about eight megabytes an hour, so a recording is
/// held comfortably. A very long one is not free, and this is why it is dropped as soon as the
/// words are placed.
/// </para>
/// </summary>
public sealed class AlignmentScores
{
    private readonly float[] _scores;

    private AlignmentScores(float[] scores, int frames, int alphabet, double frameSeconds)
    {
        _scores = scores;
        Frames = frames;
        Alphabet = alphabet;
        FrameSeconds = frameSeconds;
    }

    /// <summary>
    /// The first so-many frames of this grid, as a grid of their own, sharing the same memory.
    /// <para>
    /// This is what lets words be placed against a scan that is still running: the scan fills
    /// the grid front to back, so everything before its frontier is final while everything
    /// after is still zeros — and a prefix view never reads past the frontier it was cut at.
    /// </para>
    /// </summary>
    public AlignmentScores Prefix(int frames) =>
        frames >= Frames
            ? this
            : new AlignmentScores(_scores, Math.Max(1, frames), Alphabet, FrameSeconds);

    /// <param name="frames">How many frames the whole recording comes to.</param>
    /// <param name="alphabet">How many tokens the model knows.</param>
    /// <param name="frameSeconds">How much audio one frame covers.</param>
    public AlignmentScores(int frames, int alphabet, double frameSeconds)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(frames);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(alphabet);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(frameSeconds);

        Frames = frames;
        Alphabet = alphabet;
        FrameSeconds = frameSeconds;
        _scores = new float[(long)frames * alphabet <= int.MaxValue
            ? frames * alphabet
            : throw new ArgumentOutOfRangeException(nameof(frames), "Recording too long to score in one grid.")];
    }

    public int Frames { get; }

    public int Alphabet { get; }

    /// <summary>How much audio one frame covers, in seconds.</summary>
    public double FrameSeconds { get; }

    /// <summary>Copies one window's worth of scores into the grid.</summary>
    /// <param name="atFrame">Where in the recording this window's first row belongs.</param>
    /// <param name="window">Frame-major log probabilities, a whole number of frames.</param>
    /// <remarks>
    /// Anything reaching past the end of the grid is dropped rather than refused. The number of
    /// frames a network returns for a given number of samples is its own business — a frame or
    /// two either way at the last window is arithmetic, not a mistake.
    /// </remarks>
    public void Fill(int atFrame, ReadOnlySpan<float> window)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(atFrame);

        if (window.Length % Alphabet != 0)
        {
            throw new ArgumentException("A window must be a whole number of frames.", nameof(window));
        }

        var rows = Math.Min(window.Length / Alphabet, Frames - atFrame);
        if (rows <= 0)
        {
            return;
        }

        window[..(rows * Alphabet)].CopyTo(_scores.AsSpan(atFrame * Alphabet));
    }

    /// <summary>The frame covering an instant, clamped to the recording.</summary>
    public int FrameAt(double seconds) =>
        Math.Clamp((int)(seconds / FrameSeconds), 0, Frames);

    /// <summary>Where a frame begins, in seconds.</summary>
    public double SecondsAt(int frame) => frame * FrameSeconds;

    /// <summary>The scores for a run of frames, for handing straight to the aligner.</summary>
    public ReadOnlySpan<float> Between(int firstFrame, int frameCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(firstFrame);
        ArgumentOutOfRangeException.ThrowIfNegative(frameCount);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(firstFrame + frameCount, Frames);

        return _scores.AsSpan(firstFrame * Alphabet, frameCount * Alphabet);
    }
}
