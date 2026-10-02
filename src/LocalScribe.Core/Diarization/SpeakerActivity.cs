namespace LocalScribe.Core.Diarization;

/// <summary>
/// What an end-to-end diarizer hears: for every hundredth of a second, how likely each of its
/// speakers is to be talking.
/// <para>
/// Unlike a clustering pipeline's turns, these are independent per speaker — two can be high
/// at once, and that is the model saying two people are talking, not a boundary it has yet to
/// resolve. Crosstalk is read straight off the frames instead of being inferred afterwards.
/// </para>
/// </summary>
public sealed class SpeakerActivity
{
    /// <summary>Seconds per frame.</summary>
    public const double FrameSeconds = 0.01;

    private readonly float[] _probabilities;

    /// <param name="probabilities">Row-major: one row per frame, one value per speaker.</param>
    /// <param name="frames">Rows that are real. The buffer may run past them.</param>
    /// <param name="speakers">Values per row.</param>
    public SpeakerActivity(float[] probabilities, int frames, int speakers)
    {
        ArgumentNullException.ThrowIfNull(probabilities);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(speakers);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(frames * speakers, probabilities.Length);

        _probabilities = probabilities;
        Frames = frames;
        Speakers = speakers;
    }

    public int Frames { get; }

    public int Speakers { get; }

    /// <summary>How likely a speaker is to be talking in a frame.</summary>
    public float this[int frame, int speaker] => _probabilities[(frame * Speakers) + speaker];

    /// <summary>
    /// Unbroken runs of each speaker above the threshold, before any tidying. The honest count
    /// of what the model said, for judging the model rather than the tidying.
    /// </summary>
    public int RawRuns(double threshold = 0.5) => Runs(threshold).Count;

    /// <summary>
    /// Turns per speaker, in time order. Turns of different speakers may overlap; that is
    /// crosstalk and is left in.
    /// </summary>
    /// <param name="threshold">Probability above which a speaker counts as talking.</param>
    /// <param name="bridgeSeconds">
    /// A speaker who pauses for less than this is still mid-turn. A breath is not a boundary.
    /// </param>
    /// <param name="minimumSeconds">
    /// A turn shorter than this, after bridging, is dropped: the model flickers above the
    /// threshold for a frame or two on coughs and laughter, and a speaker made only of
    /// flickers is not a person.
    /// </param>
    public IReadOnlyList<SpeakerTurn> Turns(
        double threshold = 0.5,
        double bridgeSeconds = 0.25,
        double minimumSeconds = 0.25)
    {
        var turns = new List<SpeakerTurn>();

        foreach (var perSpeaker in Runs(threshold).GroupBy(run => run.Speaker))
        {
            SpeakerTurn? open = null;

            foreach (var run in perSpeaker.OrderBy(run => run.StartSeconds))
            {
                if (open is not null && run.StartSeconds - open.EndSeconds <= bridgeSeconds)
                {
                    open = open with { EndSeconds = run.EndSeconds };
                    continue;
                }

                Keep(open);
                open = run;
            }

            Keep(open);
        }

        // Numbered by first appearance among the turns that survived, so the labels are dense
        // and a channel that only ever flickered leaves no gap in them.
        var order = turns
            .OrderBy(turn => turn.StartSeconds)
            .Select(turn => turn.Speaker)
            .Distinct()
            .Select((speaker, index) => (speaker, index))
            .ToDictionary(pair => pair.speaker, pair => pair.index);

        return [.. turns
            .Select(turn => turn with { Speaker = order[turn.Speaker] })
            .OrderBy(turn => turn.StartSeconds)
            .ThenBy(turn => turn.Speaker)];

        void Keep(SpeakerTurn? turn)
        {
            if (turn is not null && turn.DurationSeconds >= minimumSeconds)
            {
                turns.Add(turn);
            }
        }
    }

    /// <summary>Stretches where at least two speakers are above the threshold at once.</summary>
    public IReadOnlyList<(double Start, double End)> Overlaps(double threshold = 0.5)
    {
        var spans = new List<(double Start, double End)>();
        var start = -1;

        for (var frame = 0; frame <= Frames; frame++)
        {
            var contested = frame < Frames && Talking(frame, threshold) >= 2;

            if (contested && start < 0)
            {
                start = frame;
            }
            else if (!contested && start >= 0)
            {
                spans.Add((start * FrameSeconds, frame * FrameSeconds));
                start = -1;
            }
        }

        return spans;
    }

    private int Talking(int frame, double threshold)
    {
        var count = 0;

        for (var speaker = 0; speaker < Speakers; speaker++)
        {
            if (this[frame, speaker] > threshold)
            {
                count++;
            }
        }

        return count;
    }

    private List<SpeakerTurn> Runs(double threshold)
    {
        var runs = new List<SpeakerTurn>();

        for (var speaker = 0; speaker < Speakers; speaker++)
        {
            var start = -1;

            for (var frame = 0; frame <= Frames; frame++)
            {
                var talking = frame < Frames && this[frame, speaker] > threshold;

                if (talking && start < 0)
                {
                    start = frame;
                }
                else if (!talking && start >= 0)
                {
                    runs.Add(new SpeakerTurn(speaker, start * FrameSeconds, frame * FrameSeconds));
                    start = -1;
                }
            }
        }

        return runs;
    }
}
