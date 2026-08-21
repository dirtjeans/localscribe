namespace LocalScribe.Core.Diarization;

/// <summary>
/// Follows a speaker from one window to the next by the audio the windows share, rather than by
/// what their voice sounds like.
/// <para>
/// The segmentation model separates speakers inside a ten-second window and numbers them, but
/// the numbering means nothing outside that window: the person called B at 0:30 may be C at
/// 0:35. Turning those local tracks into people has until now been done entirely by comparing
/// voice embeddings, which is the step that fails first on a poor recording — on a phone-quality
/// interview the same voice a second apart measured 0.809 apart, where 0.42 already means
/// "different people".
/// </para>
/// <para>
/// But the windows overlap by ninety percent, so consecutive ones share nine seconds. Whoever
/// was talking during those nine seconds is the same person in both, whatever each window chose
/// to call them, and that can be settled by looking at the clock instead of at the voice. On the
/// recording that prompted this, the same turn boundary was found at 37.8 seconds by five
/// consecutive windows while the embedding model could not recognise either speaker as
/// themselves.
/// </para>
/// <para>
/// Chaining those links carries an identity for as long as somebody keeps appearing at least
/// once per window. Embeddings are still needed to reunite a person after a longer silence, but
/// that is a smaller job on much more audio, and getting it wrong costs a rejoin rather than the
/// whole result.
/// </para>
/// </summary>
public static class SpeakerTracks
{
    /// <summary>
    /// Assigns a track number to every local speaker, shared by the ones that are the same
    /// person in neighbouring windows.
    /// </summary>
    /// <param name="windows">
    /// Per window, per local speaker, the spans they were active for — in seconds from the start
    /// of the recording, not from the start of the window.
    /// </param>
    /// <param name="minimumOverlap">
    /// How many seconds two local speakers must share before they are called the same person.
    /// Guards against linking on a fragment of a word.
    /// </param>
    /// <returns>A track number per local speaker, in the same shape as the input.</returns>
    public static int[][] Link(
        IReadOnlyList<IReadOnlyList<IReadOnlyList<(double Start, double End)>>> windows,
        double minimumOverlap = MinimumOverlapSeconds)
    {
        ArgumentNullException.ThrowIfNull(windows);

        // Flat index per local speaker, so the union-find below can be a plain array.
        var offsets = new int[windows.Count + 1];
        for (var w = 0; w < windows.Count; w++)
        {
            offsets[w + 1] = offsets[w] + windows[w].Count;
        }

        var parent = new int[offsets[^1]];
        for (var i = 0; i < parent.Length; i++)
        {
            parent[i] = i;
        }

        int Find(int i)
        {
            while (parent[i] != i)
            {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }

            return i;
        }

        void Union(int a, int b)
        {
            var (rootA, rootB) = (Find(a), Find(b));
            if (rootA != rootB)
            {
                parent[rootB] = rootA;
            }
        }

        for (var w = 0; w + 1 < windows.Count; w++)
        {
            var here = windows[w];
            var next = windows[w + 1];

            // Every pairing, best first. Taken greedily and one to one: a local speaker belongs
            // to exactly one person, and letting two of them claim the same track would merge
            // two speakers the window had just gone to the trouble of separating.
            var pairs = new List<(double Shared, int Here, int There)>();

            for (var a = 0; a < here.Count; a++)
            {
                for (var b = 0; b < next.Count; b++)
                {
                    var shared = SharedSeconds(here[a], next[b]);
                    if (shared >= minimumOverlap)
                    {
                        pairs.Add((shared, a, b));
                    }
                }
            }

            var takenHere = new bool[here.Count];
            var takenThere = new bool[next.Count];

            foreach (var (_, a, b) in pairs.OrderByDescending(pair => pair.Shared))
            {
                if (takenHere[a] || takenThere[b])
                {
                    continue;
                }

                takenHere[a] = true;
                takenThere[b] = true;
                Union(offsets[w] + a, offsets[w + 1] + b);
            }
        }

        // Numbered by first appearance, so track 0 is whoever spoke first.
        var numbers = new Dictionary<int, int>();
        var result = new int[windows.Count][];

        for (var w = 0; w < windows.Count; w++)
        {
            result[w] = new int[windows[w].Count];

            for (var s = 0; s < windows[w].Count; s++)
            {
                var root = Find(offsets[w] + s);

                if (!numbers.TryGetValue(root, out var number))
                {
                    number = numbers.Count;
                    numbers[root] = number;
                }

                result[w][s] = number;
            }
        }

        return result;
    }

    /// <summary>
    /// Turns linked windows into turns, by letting every window vote on every moment.
    /// <para>
    /// Each instant of the recording is covered by about ten overlapping windows, which until
    /// now was pure waste — the same speech embedded ten times over, and near-duplicate spans
    /// thrown away afterwards. Counted instead, it is ten independent opinions about who was
    /// talking, and the majority is steadier than any single window.
    /// </para>
    /// </summary>
    /// <param name="windows">Spans per window per local speaker, as given to <see cref="Link"/>.</param>
    /// <param name="tracks">The track numbers <see cref="Link"/> returned.</param>
    /// <param name="durationSeconds">Length of the recording.</param>
    public static IReadOnlyList<SpeakerTurn> ToTurns(
        IReadOnlyList<IReadOnlyList<IReadOnlyList<(double Start, double End)>>> windows,
        int[][] tracks,
        double durationSeconds)
    {
        ArgumentNullException.ThrowIfNull(windows);
        ArgumentNullException.ThrowIfNull(tracks);

        var frames = (int)Math.Ceiling(durationSeconds / FrameSeconds);
        if (frames <= 0)
        {
            return [];
        }

        var speakers = tracks.SelectMany(window => window).DefaultIfEmpty(-1).Max() + 1;
        if (speakers <= 0)
        {
            return [];
        }

        var votes = new int[frames, speakers];

        for (var w = 0; w < windows.Count; w++)
        {
            for (var s = 0; s < windows[w].Count; s++)
            {
                var track = tracks[w][s];

                foreach (var (start, end) in windows[w][s])
                {
                    // The end of a span is exclusive. Treating it as inclusive gave every turn
                    // one frame of its neighbour, so two adjacent turns each claimed the moment
                    // they met and the boundary landed a frame late.
                    var from = Math.Max(0, (int)(start / FrameSeconds));
                    var to = Math.Min(frames - 1, (int)Math.Ceiling(end / FrameSeconds) - 1);

                    for (var f = from; f <= to; f++)
                    {
                        votes[f, track]++;
                    }
                }
            }
        }

        var winner = new int[frames];

        for (var f = 0; f < frames; f++)
        {
            var best = -1;
            var most = 0;

            for (var s = 0; s < speakers; s++)
            {
                if (votes[f, s] > most)
                {
                    most = votes[f, s];
                    best = s;
                }
            }

            // A tie goes to whoever held the previous frame, which keeps a turn whole rather
            // than shredding its edges where two tracks are equally supported.
            winner[f] = most == 0
                ? -1
                : (f > 0 && winner[f - 1] >= 0 && votes[f, winner[f - 1]] == most ? winner[f - 1] : best);
        }

        var turns = new List<SpeakerTurn>();
        var at = 0;

        while (at < frames)
        {
            if (winner[at] < 0)
            {
                at++;
                continue;
            }

            var speaker = winner[at];
            var start = at;

            while (at < frames && winner[at] == speaker)
            {
                at++;
            }

            turns.Add(new SpeakerTurn(speaker, start * FrameSeconds, at * FrameSeconds));
        }

        return turns;
    }

    /// <summary>Seconds two sets of spans have in common.</summary>
    private static double SharedSeconds(
        IReadOnlyList<(double Start, double End)> left,
        IReadOnlyList<(double Start, double End)> right)
    {
        var total = 0.0;

        foreach (var (aStart, aEnd) in left)
        {
            foreach (var (bStart, bEnd) in right)
            {
                total += Math.Max(0, Math.Min(aEnd, bEnd) - Math.Max(aStart, bStart));
            }
        }

        return total;
    }

    /// <summary>
    /// How much speech two local speakers must share to be called the same person. Half a second
    /// is short enough to link a brief interjection and long enough that a stray frame at a turn
    /// boundary does not join two people together.
    /// </summary>
    private const double MinimumOverlapSeconds = 0.5;

    /// <summary>Resolution of the vote. Finer than any turn boundary is worth arguing about.</summary>
    private const double FrameSeconds = 0.05;
}
