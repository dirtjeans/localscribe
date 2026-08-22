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
                // A window always reports a slot per local speaker, and on a recording with one
                // person talking most of them are empty. Numbering those gave a track to every
                // silence: 1,260 windows produced 2,196 tracks, of which the overwhelming
                // majority were nobody at all, and they swamped everything downstream.
                if (windows[w][s].Count == 0)
                {
                    result[w][s] = Silent;
                    continue;
                }

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

        var speakers = tracks.SelectMany(window => window).DefaultIfEmpty(Silent).Max() + 1;
        if (speakers <= 0)
        {
            return [];
        }

        var votes = Votes(windows, tracks, frames, speakers);

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

            // A run of one frame is a flicker in the vote, not a turn. Twenty milliseconds of
            // somebody is nothing a reader can click on and nothing a listener would notice.
            if (at - start > 1)
            {
                turns.Add(new SpeakerTurn(speaker, start * FrameSeconds, at * FrameSeconds));
            }
        }

        // Numbered by who speaks first, which is the order a reader meets them in. Anything else
        // puts Speaker 2 at the top of the transcript and asks the reader to wonder where
        // Speaker 1 went — numbering by who talks most did exactly that.
        var order = new Dictionary<int, int>();

        foreach (var turn in turns)
        {
            if (!order.ContainsKey(turn.Speaker))
            {
                order[turn.Speaker] = order.Count;
            }
        }

        return [.. turns.Select(turn => turn with { Speaker = order[turn.Speaker] })];
    }

    /// <summary>
    /// Sorts the tracks into two people using only what the segmentation model saw, without
    /// asking what anybody sounds like.
    /// <para>
    /// Two local speakers active in the same window are two different people. That is a fact the
    /// segmentation model supplies directly, it needs no voice comparison, and it survives audio
    /// that defeats the embedding model completely. On the interview that prompted this, 465
    /// windows produced 501 such facts, and all but 8 of them were mutually consistent — the
    /// graph they form is very nearly two-colourable, and colouring it splits the recording
    /// 48/52. Clustering the same recording by voice gave 98/2.
    /// </para>
    /// <para>
    /// What the facts cannot settle is which colour is which person across a gap in the
    /// conversation: they connect tracks that were talking at the same moment, so a stretch
    /// where nobody interrupted anybody forms an island of its own. Those islands are joined by
    /// the one other thing known without listening — that consecutive turns are usually
    /// different people — voted over every pair of tracks that follow one another in time.
    /// </para>
    /// <para>
    /// Two people only. The reasoning is a two-colouring and does not generalise to three: with
    /// more speakers the constraints stop determining an answer and voices have to be compared
    /// after all.
    /// </para>
    /// </summary>
    /// <returns>Track numbers renumbered to 0 and 1, or null when the constraints say too little.</returns>
    public static int[][]? SeparateTwo(
        IReadOnlyList<IReadOnlyList<IReadOnlyList<(double Start, double End)>>> windows,
        int[][] tracks)
    {
        ArgumentNullException.ThrowIfNull(windows);
        ArgumentNullException.ThrowIfNull(tracks);

        var extent = new Dictionary<int, (double Start, double End, double Seconds)>();
        var conflicts = new Dictionary<int, HashSet<int>>();

        for (var w = 0; w < windows.Count; w++)
        {
            var active = new List<int>();

            for (var s = 0; s < windows[w].Count; s++)
            {
                var track = tracks[w][s];
                if (track == Silent || windows[w][s].Count == 0)
                {
                    continue;
                }

                var seconds = windows[w][s].Sum(span => span.End - span.Start);
                if (seconds <= 0)
                {
                    continue;
                }

                var start = windows[w][s].Min(span => span.Start);
                var end = windows[w][s].Max(span => span.End);

                extent[track] = extent.TryGetValue(track, out var was)
                    ? (Math.Min(was.Start, start), Math.Max(was.End, end), was.Seconds + seconds)
                    : (start, end, seconds);

                conflicts.TryAdd(track, []);

                if (seconds >= MinimumOverlapSeconds)
                {
                    active.Add(track);
                }
            }

            foreach (var one in active.Distinct())
            {
                foreach (var other in active.Distinct())
                {
                    if (one != other)
                    {
                        conflicts[one].Add(other);
                    }
                }
            }
        }

        if (extent.Count == 0 || conflicts.Values.All(set => set.Count == 0))
        {
            return null;
        }

        // Two-colour each island of mutually-constrained tracks.
        var colour = new Dictionary<int, int>();
        var island = new Dictionary<int, int>();
        var islands = 0;

        foreach (var start in extent.Keys.OrderBy(t => extent[t].Start))
        {
            if (colour.ContainsKey(start))
            {
                continue;
            }

            colour[start] = 0;
            island[start] = islands;

            var queue = new Queue<int>([start]);
            while (queue.Count > 0)
            {
                var at = queue.Dequeue();

                foreach (var other in conflicts[at])
                {
                    if (colour.ContainsKey(other))
                    {
                        continue;
                    }

                    colour[other] = 1 - colour[at];
                    island[other] = islands;
                    queue.Enqueue(other);
                }
            }

            islands++;
        }

        // Join the islands. Whoever spoke next is usually not whoever spoke last, so every pair
        // of consecutive tracks is a vote that their colours should differ. Counted across the
        // recording, that settles which way round each island goes relative to the biggest one.
        var order = extent.Keys.OrderBy(t => extent[t].Start).ToList();
        var flipVotes = new Dictionary<int, double>();

        for (var i = 0; i + 1 < order.Count; i++)
        {
            var (a, b) = (order[i], order[i + 1]);
            if (island[a] == island[b])
            {
                continue;
            }

            // Positive means the islands disagree about which colour is which.
            var wantsSame = colour[a] == colour[b];
            var weight = Math.Min(extent[a].Seconds, extent[b].Seconds);

            var key = island[b];
            flipVotes[key] = flipVotes.GetValueOrDefault(key) + (wantsSame ? weight : -weight);
        }

        var anchor = island[extent.OrderByDescending(e => e.Value.Seconds).First().Key];

        foreach (var track in order)
        {
            if (island[track] != anchor && flipVotes.GetValueOrDefault(island[track]) > 0)
            {
                colour[track] = 1 - colour[track];
            }
        }

        return [.. tracks.Select(window =>
            window.Select(track => track == Silent || !colour.TryGetValue(track, out var c) ? Silent : c)
                .ToArray())];
    }

    /// <summary>
    /// Groups tracks into people by voice, but forbidden from ever putting two of them together
    /// when the segmentation model saw them talking at the same moment.
    /// <para>
    /// The general form of <see cref="SeparateTwo"/>, for recordings with more than two people
    /// in them. Two-colouring settles a two-speaker recording outright and needs no voices at
    /// all; with three or more, the constraints stop determining an answer on their own and
    /// something has to choose between the arrangements that satisfy them. Voice similarity is
    /// what chooses — but only among legal arrangements, which is the difference between using
    /// weak evidence and being led by it.
    /// </para>
    /// <para>
    /// That matters most exactly where the voices are hardest to tell apart. Unconstrained
    /// clustering on a poor recording collapses everyone into one person; here the collapse is
    /// impossible, because the pairs it would have to merge are the pairs known to be different.
    /// </para>
    /// </summary>
    /// <param name="voices">One embedding per track, in track order.</param>
    /// <param name="conflicts">For each track, the tracks it cannot share a person with.</param>
    /// <param name="wanted">How many people to end up with.</param>
    /// <returns>A person number per track.</returns>
    public static int[] GroupWithConstraints(
        IReadOnlyList<float[]> voices,
        IReadOnlyList<IReadOnlyList<int>> conflicts,
        int wanted,
        double threshold = SpeakerClustering.DefaultThreshold)
    {
        ArgumentNullException.ThrowIfNull(voices);
        ArgumentNullException.ThrowIfNull(conflicts);

        var people = new int[voices.Count];
        Array.Fill(people, -1);

        if (voices.Count == 0 || wanted <= 0)
        {
            return people;
        }

        // Hardest first. A track that cannot sit with many others has the fewest places to go,
        // and placing it early is what keeps the colouring tight; leaving it late is how a
        // colouring ends up needing more colours than the graph does.
        var order = Enumerable.Range(0, voices.Count)
            .OrderByDescending(i => conflicts[i].Count)
            .ToList();

        var members = Enumerable.Range(0, wanted).Select(_ => new List<int>()).ToList();

        foreach (var track in order)
        {
            var barred = conflicts[track]
                .Where(other => people[other] >= 0)
                .Select(other => people[other])
                .ToHashSet();

            var legal = Enumerable.Range(0, wanted).Where(p => !barred.Contains(p)).ToList();

            if (legal.Count == 0)
            {
                // More people talking at once than we were told were in the room. Somebody has
                // to take it; the group they clash with least is the least wrong answer.
                people[track] = Enumerable.Range(0, wanted)
                    .OrderBy(p => conflicts[track].Count(other => people[other] == p))
                    .First();

                members[people[track]].Add(track);
                continue;
            }

            var occupied = legal.Where(p => members[p].Count > 0).ToList();
            var empty = legal.FirstOrDefault(p => members[p].Count == 0, -1);

            double DistanceTo(int person) =>
                members[person].Average(other => SpeakerClustering.CosineDistance(voices[track], voices[other]));

            var nearest = occupied.Count == 0 ? -1 : occupied.OrderBy(DistanceTo).First();

            // An empty group is opened for someone who sounds like nobody already placed. Where
            // the voices cannot be told apart this rarely fires, and the constraints above are
            // doing the work instead — which is the intended division of labour, not a
            // shortcoming.
            people[track] = nearest < 0 || (empty >= 0 && DistanceTo(nearest) > threshold)
                ? (empty >= 0 ? empty : nearest)
                : nearest;

            members[people[track]].Add(track);
        }

        // Compacted so the numbers run 0..n-1 with no gaps. Which number means which person is
        // settled later, by who speaks first — see ToTurns.
        var ranking = Enumerable.Range(0, wanted)
            .Where(p => members[p].Count > 0)
            .OrderByDescending(p => members[p].Count)
            .Select((p, rank) => (p, rank))
            .ToDictionary(x => x.p, x => x.rank);

        return [.. people.Select(p => p >= 0 && ranking.TryGetValue(p, out var rank) ? rank : 0)];
    }

    /// <summary>
    /// The fewest people the segmentation model's own evidence can be explained by.
    /// <para>
    /// Two tracks talking at the same moment are two people, so colouring the graph of those
    /// facts puts a floor under the speaker count — reached without comparing a single voice,
    /// which is what makes it worth having on a recording where voices cannot be compared. It is
    /// a floor and not an answer: people who never talk over anybody leave no evidence here at
    /// all, so a tidy conversation of six can look like one.
    /// </para>
    /// </summary>
    public static int AtLeastThisManyPeople(IReadOnlyList<IReadOnlyList<int>> conflicts)
    {
        ArgumentNullException.ThrowIfNull(conflicts);

        if (conflicts.Count == 0)
        {
            return 0;
        }

        // Most-constrained first, which is what makes greedy colouring tight in practice.
        var order = Enumerable.Range(0, conflicts.Count)
            .OrderByDescending(i => conflicts[i].Count)
            .ToList();

        var colour = new int[conflicts.Count];
        Array.Fill(colour, -1);

        foreach (var track in order)
        {
            var taken = conflicts[track].Where(other => colour[other] >= 0).Select(other => colour[other]).ToHashSet();

            var pick = 0;
            while (taken.Contains(pick))
            {
                pick++;
            }

            colour[track] = pick;
        }

        return colour.Max() + 1;
    }

    /// <summary>
    /// The stretches where more than one person was talking at once.
    /// <para>
    /// Already known and previously discarded. Every instant is covered by about ten windows,
    /// each with an opinion about who was speaking; <see cref="ToTurns"/> takes the winner and
    /// throws away the fact that the vote was contested. Where two tracks both hold a real share
    /// of the frames, two people were talking — the segmentation model separated them at the
    /// time, and only the insistence on one name per moment loses it.
    /// </para>
    /// <para>
    /// Worth reporting rather than resolving. Attributing crosstalk to whichever voice happened
    /// to win is wrong by construction, and the words there are unreliable anyway: a transcriber
    /// hearing two people produces one stream with both of them interleaved. Marking the stretch
    /// says something true where a name would say something false.
    /// </para>
    /// </summary>
    /// <returns>Spans of time, in order, where two or more people overlap.</returns>
    public static IReadOnlyList<(double Start, double End)> Overlaps(
        IReadOnlyList<IReadOnlyList<IReadOnlyList<(double Start, double End)>>> windows,
        int[][] tracks,
        double durationSeconds)
    {
        ArgumentNullException.ThrowIfNull(windows);
        ArgumentNullException.ThrowIfNull(tracks);

        var frames = (int)Math.Ceiling(durationSeconds / FrameSeconds);
        var speakers = tracks.SelectMany(window => window).DefaultIfEmpty(Silent).Max() + 1;

        if (frames <= 0 || speakers <= 1)
        {
            return [];
        }

        var votes = Votes(windows, tracks, frames, speakers);
        var contested = new bool[frames];

        for (var f = 0; f < frames; f++)
        {
            var most = 0;
            for (var s = 0; s < speakers; s++)
            {
                most = Math.Max(most, votes[f, s]);
            }

            if (most == 0)
            {
                continue;
            }

            // A second voice counts when a real share of the windows heard it, not when one
            // window in ten disagreed with the other nine. That is the difference between two
            // people talking and one boundary being drawn a moment early.
            var voices = 0;
            for (var s = 0; s < speakers; s++)
            {
                if (votes[f, s] >= most * ContestedShare)
                {
                    voices++;
                }
            }

            contested[f] = voices > 1;
        }

        var spans = new List<(double Start, double End)>();
        var at = 0;

        while (at < frames)
        {
            if (!contested[at])
            {
                at++;
                continue;
            }

            var from = at;
            while (at < frames && contested[at])
            {
                at++;
            }

            // A flicker of disagreement at a turn boundary is not people talking over each
            // other; it is the boundary itself being uncertain by a frame or two.
            if ((at - from) * FrameSeconds >= ShortestOverlap)
            {
                spans.Add((from * FrameSeconds, at * FrameSeconds));
            }
        }

        return spans;
    }

    /// <summary>
    /// How much of the leading voice's support a second one needs before both are believed.
    /// </summary>
    private const double ContestedShare = 0.6;

    /// <summary>
    /// Shorter than this and it is a turn boundary drawn uncertainly rather than two people
    /// speaking at once.
    /// </summary>
    private const double ShortestOverlap = 0.4;

    /// <summary>
    /// How many windows say each track was talking, frame by frame.
    /// <para>
    /// Every instant is covered by about ten overlapping windows. Counting them is what makes a
    /// single window's mistake harmless, and it is the same table whether the question is who
    /// won or whether anyone disagreed.
    /// </para>
    /// </summary>
    private static int[,] Votes(
        IReadOnlyList<IReadOnlyList<IReadOnlyList<(double Start, double End)>>> windows,
        int[][] tracks,
        int frames,
        int speakers)
    {
        var votes = new int[frames, speakers];

        for (var w = 0; w < windows.Count; w++)
        {
            for (var s = 0; s < windows[w].Count; s++)
            {
                var track = tracks[w][s];
                if (track == Silent || track >= speakers)
                {
                    continue;
                }

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

        return votes;
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

    /// <summary>Marks a local speaker slot that nobody occupied.</summary>
    public const int Silent = -1;

    /// <summary>Resolution of the vote. Finer than any turn boundary is worth arguing about.</summary>
    private const double FrameSeconds = 0.05;
}
