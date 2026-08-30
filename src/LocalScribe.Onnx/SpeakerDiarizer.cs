using LocalScribe.Core.Audio;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Transcription;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalScribe.Onnx;

/// <summary>
/// Works out who spoke when.
/// <para>
/// Two models, because the question has two halves that nothing answers at once. Segmentation
/// says how many voices are active in a ten-second window and where each one starts and stops,
/// but numbers them locally — its speaker 0 in one window has nothing to do with speaker 0 in
/// the next. Embeddings turn a stretch of one voice into a vector that is close to other
/// stretches of the same voice, which is what carries identity between windows. Clustering those
/// vectors is what turns a pile of local guesses into a recording with two people in it.
/// </para>
/// <para>
/// Whisper does neither of these things and cannot be made to. This runs alongside it, over the
/// same audio, and the result is joined to the transcript by time.
/// </para>
/// <para>
/// Both models are float32 with dynamic shapes, so they run on the CPU. The QNN provider needs
/// fixed shapes and quantised weights; making these NPU-capable is a separate exercise, and at
/// six and twenty-six megabytes they are not what makes a transcription slow.
/// </para>
/// </summary>
public sealed class SpeakerDiarizer : IDisposable
{
    private readonly InferenceSession _segmentation;
    private readonly InferenceSession _embedding;
    private readonly KaldiFbank _fbank = new();

    private readonly int _windowSamples;
    private readonly int _localSpeakers;
    private readonly int _maxOverlap;
    private readonly double _receptiveFieldShift;
    private readonly double _receptiveFieldStart;
    private readonly IReadOnlyList<IReadOnlyList<int>> _mapping;

    /// <summary>
    /// How far the analysis window moves each step. A tenth of the window, as pyannote uses:
    /// enough overlap that a speaker change is seen by several windows rather than landing on a
    /// boundary and being missed.
    /// </summary>
    private const double WindowShiftFraction = 0.1;

    /// <summary>
    /// Speech shorter than this is not given its own embedding. A third of a second of one voice
    /// is not enough to identify it, and a vector built from too little audio clusters by
    /// accident rather than by speaker.
    /// </summary>
    private const double MinimumTurnSeconds = 0.35;

    private SpeakerDiarizer(InferenceSession segmentation, InferenceSession embedding)
    {
        _segmentation = segmentation;
        _embedding = embedding;

        var metadata = segmentation.ModelMetadata.CustomMetadataMap;

        _windowSamples = Read(metadata, "window_size", 160_000);
        _localSpeakers = Read(metadata, "num_speakers", 3);
        _maxOverlap = Read(metadata, "powerset_max_classes", 2);

        // The model's output frames are not evenly spaced over the window: each one summarises a
        // receptive field, so the first is centred partway in. Getting this wrong shifts every
        // boundary by a fixed amount, which looks like a model that is slightly late rather than
        // like an arithmetic mistake.
        var fieldSize = Read(metadata, "receptive_field_size", 991);
        var fieldShift = Read(metadata, "receptive_field_shift", 270);
        var sampleRate = Read(metadata, "sample_rate", PcmAudio.WhisperSampleRate);

        _receptiveFieldShift = fieldShift / (double)sampleRate;
        _receptiveFieldStart = fieldSize / 2.0 / sampleRate;

        _mapping = PowersetDecoder.Mapping(_localSpeakers, _maxOverlap);
    }

    /// <summary>Opens both models from a directory holding segmentation.onnx and embedding.onnx.</summary>
    /// <param name="plan">
    /// Supplies the CPU thread budget, or null to let ONNX Runtime take every core — which is
    /// what these sessions always did, and on Windows still do until a --diarize turn diff on
    /// the reference recordings proves the cap moves no boundary. Thread count can reorder
    /// float summation, and the tuning is frozen: a boundary moving a hundredth of a second is
    /// a tuning change whatever it was called. On the Mac the diff has been run; see
    /// docs/handoff-macos.md.
    /// </param>
    public static SpeakerDiarizer Load(string modelDirectory, ExecutionPlan? plan = null)
    {
        using var options = plan is null
            ? null
            : new SessionOptions
            {
                IntraOpNumThreads = plan.CpuBudget.IntraOpThreads,
                InterOpNumThreads = plan.CpuBudget.InterOpThreads,
            };

        var segmentation = options is null
            ? new InferenceSession(Require(modelDirectory, "segmentation.onnx"))
            : new InferenceSession(Require(modelDirectory, "segmentation.onnx"), options);

        try
        {
            var embedding = options is null
                ? new InferenceSession(Require(modelDirectory, "embedding.onnx"))
                : new InferenceSession(Require(modelDirectory, "embedding.onnx"), options);
            return new SpeakerDiarizer(segmentation, embedding);
        }
        catch
        {
            segmentation.Dispose();
            throw;
        }
    }

    private static string Require(string directory, string name)
    {
        var path = Path.Combine(directory, name);

        return File.Exists(path)
            ? path
            : throw new FileNotFoundException(
                $"No {name} in {directory}. Run 'localscribe-doctor --fetch-models --diarization' "
                + "to download the speaker models.",
                path);
    }

    /// <summary>Finds the speaker turns in a recording.</summary>
    /// <param name="audio">16 kHz mono, the same audio the transcriber saw.</param>
    /// <param name="threshold">Cosine distance at which two voices count as different people.</param>
    /// <param name="maxSpeakers">Upper bound on speakers, or null to let the threshold decide.</param>
    /// <param name="exactSpeakers">
    /// The known number of speakers, when the user has said. Far more reliable than inferring it
    /// from a distance threshold, which on a real recording tends to fail in both directions at
    /// once.
    /// </param>
    public IReadOnlyList<SpeakerTurn> Diarize(
        PcmAudio audio,
        double threshold = SpeakerClustering.DefaultThreshold,
        int? maxSpeakers = null,
        int? exactSpeakers = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var voices = Describe(audio, progress, cancellationToken);
        if (voices.Count == 0)
        {
            return [];
        }

        var labels = Assign(voices, threshold, maxSpeakers, exactSpeakers);

        var turns = voices
            .Select((voice, i) => new SpeakerTurn(labels[i], voice.StartSeconds, voice.EndSeconds))
            .OrderBy(t => t.StartSeconds)
            .ToList();

        return Tidy(turns);
    }

    /// <summary>
    /// Decides who is who, given the stretches <see cref="Describe"/> found.
    /// <para>
    /// The half of diarization that has no models in it, split out so that the doctor can sweep
    /// a threshold across a real recording without paying for the scan each time — and, more to
    /// the point, so that what it sweeps is the same code the app runs. A sweep that clustered
    /// differently from the thing being calibrated would be worse than no sweep at all.
    /// </para>
    /// </summary>
    public static int[] Assign(
        IReadOnlyList<Voice> voices,
        double threshold = SpeakerClustering.DefaultThreshold,
        int? maxSpeakers = null,
        int? exactSpeakers = null)
    {
        ArgumentNullException.ThrowIfNull(voices);

        if (voices.Count == 0)
        {
            return [];
        }

        return ClusterWithShortSpansAttached(
            [.. voices.Select(voice => voice.Embedding)],
            [.. voices.Select(voice => (Start: voice.StartSeconds, End: voice.EndSeconds))],
            threshold,
            maxSpeakers,
            exactSpeakers);
    }

    /// <summary>
    /// Where two people talked over each other in the last run of
    /// <see cref="DiarizeByTracking"/>.
    /// <para>
    /// Reported rather than resolved. Handing crosstalk to whichever voice won the vote is wrong
    /// by construction, and the words there are unreliable anyway — a transcriber hearing two
    /// people at once returns one stream with both of them interleaved. Saying so is true where
    /// a name would not be.
    /// </para>
    /// </summary>
    public IReadOnlyList<(double Start, double End)> LastOverlaps { get; private set; } = [];

    /// <summary>A stretch of speech and what the embedding model made of it.</summary>
    public sealed record Voice(double StartSeconds, double EndSeconds, float[] Embedding)
    {
        public double DurationSeconds => EndSeconds - StartSeconds;
    }

    /// <summary>
    /// Everything the clustering works from: where the speech is, and an embedding of each
    /// stretch — but no decision about who is who.
    /// <para>
    /// Exposed for diagnosis. Whether two people are told apart comes down to one distance
    /// threshold, and the right value for it is a property of the recording rather than of the
    /// code: how alike the voices are, how much room noise there is, how long anyone speaks
    /// uninterrupted. Guessing it from a sample of synthesised speech and hoping is how it came
    /// to be wrong. This lets the doctor sweep the threshold over a real recording and show
    /// where the answer actually changes.
    /// </para>
    /// </summary>
    public IReadOnlyList<Voice> Describe(
        PcmAudio audio,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);

        var local = CollectSpeechSpans(audio, progress, cancellationToken);
        var voices = new List<Voice>(local.Count);

        for (var i = 0; i < local.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            progress?.Report(ScanShare + ((1 - ScanShare) * i / local.Count));

            var (start, end) = local[i];

            if (Embed(audio, start, end) is { } embedding)
            {
                voices.Add(new Voice(start, end, embedding));
            }
        }

        return voices;
    }

    /// <summary>
    /// How much of the reported progress the window scan accounts for. Measured roughly: the
    /// scan runs the segmentation model over every window, the embeddings run a smaller model
    /// over a few dozen short spans.
    /// </summary>
    private const double ScanShare = 0.8;

    /// <summary>
    /// Speech shorter than this identifies a voice poorly. Long enough to say "Good." and not
    /// much else, and an embedding built from that clusters by accident as often as by speaker.
    /// </summary>
    private const double ReliableTurnSeconds = 0.9;

    /// <summary>
    /// How much a second of separation counts against a match, when placing a short span. Small
    /// enough that a clearly different voice still wins, large enough to decide between two that
    /// sound alike.
    /// </summary>
    private const double GapWeightPerSecond = 0.02;

    /// <summary>
    /// How much further than the different-speaker threshold a short stretch must sit from every
    /// known voice before it is allowed to be somebody new.
    /// <para>
    /// A margin rather than the threshold itself, because half a second of speech identifies
    /// nobody reliably and a noisy embedding should not invent a speaker. Beyond this, though,
    /// the evidence is not marginal: the interjections that prompted this measured 0.54 and 0.70
    /// against a threshold of 0.42.
    /// </para>
    /// </summary>
    private const double StrangerMargin = 1.25;

    /// <summary>
    /// Clusters the spans long enough to be trusted, then attaches the short ones to whichever
    /// of those they most resemble.
    /// <para>
    /// Short turns are not noise — "Good.", "Right.", "Mm." are most of what a conversation is
    /// made of — but they are too brief to identify on their own. Left in the clustering they
    /// invent speakers: half a second of one word became a third person in a two-person
    /// recording. Attached afterwards, they go to the nearest voice actually established
    /// elsewhere, which is the answer a listener would give.
    /// </para>
    /// </summary>
    private static int[] ClusterWithShortSpansAttached(
        List<float[]> embeddings,
        List<(double Start, double End)> spans,
        double threshold,
        int? maxSpeakers,
        int? exactSpeakers)
    {
        var reliable = Enumerable.Range(0, spans.Count)
            .Where(i => spans[i].End - spans[i].Start >= ReliableTurnSeconds)
            .ToList();

        // Nothing long enough to anchor on: cluster everything and take what comes.
        if (reliable.Count < 2)
        {
            return SpeakerClustering.Cluster(embeddings, threshold, maxSpeakers, exactSpeakers);
        }

        var reliableLabels = SpeakerClustering.Cluster(
            reliable.Select(i => embeddings[i]).ToList(), threshold, maxSpeakers, exactSpeakers);

        var labels = new int[spans.Count];
        for (var i = 0; i < reliable.Count; i++)
        {
            labels[reliable[i]] = reliableLabels[i];
        }

        var strangers = new List<int>();

        for (var i = 0; i < spans.Count; i++)
        {
            if (reliable.Contains(i))
            {
                continue;
            }

            var nearest = 0;
            var best = double.MaxValue;
            var closestVoice = double.MaxValue;

            for (var r = 0; r < reliable.Count; r++)
            {
                var distance = SpeakerClustering.CosineDistance(
                    Unit(embeddings[i]), Unit(embeddings[reliable[r]]));

                closestVoice = Math.Min(closestVoice, distance);

                // Weighted by how far away in time it is, gently. Half a second of speech does
                // not identify anyone confidently, and when the voice is ambiguous the next best
                // evidence is who was talking either side of it — an interjection belongs to the
                // conversation around it far more often than to someone across the recording.
                var gap = Math.Max(0, Math.Max(
                    spans[reliable[r]].Start - spans[i].End,
                    spans[i].Start - spans[reliable[r]].End));

                var score = distance + (GapWeightPerSecond * gap);

                if (score < best)
                {
                    best = score;
                    nearest = reliableLabels[r];
                }
            }

            // Unless it sounds like nobody here. Attaching short speech to a neighbour is a way
            // of handling evidence too thin to trust, not a rule that everyone brief must be
            // someone already known — and applied unconditionally it silently deletes a speaker
            // who only ever interjects. On a four-minute recording of one person, the two
            // half-second interruptions by somebody else sat 0.54 and 0.70 from the main voice,
            // which this same code calls a different person anywhere else, and both were handed
            // to the person they had interrupted.
            if (closestVoice > threshold * StrangerMargin)
            {
                strangers.Add(i);
            }
            else
            {
                labels[i] = nearest;
            }
        }

        if (strangers.Count > 0)
        {
            // Clustered among themselves, so that one person interjecting five times is one new
            // speaker rather than five.
            var next = reliableLabels.Max() + 1;

            var among = SpeakerClustering.Cluster(
                [.. strangers.Select(i => embeddings[i])], threshold);

            for (var i = 0; i < strangers.Count; i++)
            {
                labels[strangers[i]] = next + among[i];
            }
        }

        return labels;
    }

    private static float[] Unit(float[] vector)
    {
        var sum = 0.0;
        foreach (var value in vector)
        {
            sum += value * value;
        }

        var magnitude = Math.Sqrt(sum);

        return magnitude < 1e-12 ? vector : vector.Select(v => (float)(v / magnitude)).ToArray();
    }

    /// <summary>
    /// Runs the segmentation model across the recording and collects every stretch where one
    /// local speaker was active.
    /// <para>
    /// Kept separate, one span per speaker per window, and deliberately not merged. Merging them
    /// into a single timeline first is the obvious thing to do and it is wrong: two people
    /// either side of a boundary become one span, and every speaker change inside a window
    /// disappears before anything has a chance to notice it. What survives is a recording that
    /// looks like one long turn each.
    /// </para>
    /// <para>
    /// Local speaker numbers are still meaningless between windows — the embeddings decide
    /// identity — but within a window they separate voices, and that is worth keeping.
    /// </para>
    /// </summary>
    /// <summary>
    /// Works out who spoke when by following speakers between overlapping windows, instead of
    /// by comparing what their voices sound like.
    /// <para>
    /// The alternative to <see cref="Diarize"/>, and the better one wherever the recording is
    /// poor. Both start from the same segmentation model; they differ in how they turn its
    /// per-window numbering into people. Clustering embeddings needs the voices to be
    /// distinguishable, and on a phone-quality interview they are not — the same voice a second
    /// apart measured 0.809 apart there, against a 0.42 threshold for being different people.
    /// Following the clock needs only that somebody keeps talking often enough to appear in
    /// consecutive windows, which is a far weaker thing to ask.
    /// </para>
    /// <para>
    /// What it cannot do on its own is reunite someone after a long silence: a person who says
    /// nothing for more than a window comes back as a new track. Reducing those tracks to the
    /// number of people in the room is what <paramref name="exactSpeakers"/> is for, and that
    /// last step does use embeddings — but over all of a track's audio at once rather than a few
    /// seconds at a time, which is the one condition under which they are worth trusting here.
    /// </para>
    /// </summary>
    public IReadOnlyList<SpeakerTurn> DiarizeByTracking(
        PcmAudio audio,
        int? maxSpeakers = null,
        int? exactSpeakers = null,
        double threshold = SpeakerClustering.DefaultThreshold,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);

        var windows = DescribeWindows(audio, progress, cancellationToken);
        if (windows.Count == 0)
        {
            return [];
        }

        var spans = windows.Select(window => window.Speakers).ToList();
        var tracks = SpeakerTracks.Link(spans);

        // Following speakers between windows can chain through a mistake. Where the segmentation
        // model briefly puts one person's speech in the other's slot, the link is made on real
        // shared audio and the two are welded into one track — after which nothing downstream can
        // separate them, because they have become a single object before anyone asks who they
        // are. Listening to each track is the check, and it is only worth doing now that the
        // embeddings work: a track holding two people shows 0.78 between its own halves.
        tracks = SplitImpureTracks(audio, spans, tracks, threshold, cancellationToken);

        var wanted = exactSpeakers ?? maxSpeakers;

        if (wanted is { } limit && limit > 0)
        {
            // Two people can be told apart by the segmentation model alone: two local speakers
            // in one window are two different people, and that fact needs no voice comparison.
            // Tried first because it survives audio the embedding model cannot read.
            tracks = (limit == 2 ? SpeakerTracks.SeparateTwo(spans, tracks) : null)
                ?? Merge(audio, spans, tracks, limit, threshold, cancellationToken);
        }
        else
        {
            // No count given. The constraints still put a floor under it, without listening to
            // anybody: tracks that talked over one another are different people, and colouring
            // that graph says how few people the evidence can be explained by. Better than
            // letting a distance threshold guess, which is what produced 19 speakers on a
            // recording with three.
            var floor = SpeakerTracks.AtLeastThisManyPeople(ConflictsAmong(spans, tracks).Edges);

            if (floor >= 2)
            {
                tracks = Merge(audio, spans, tracks, floor, threshold, cancellationToken);
            }
        }

        LastOverlaps = SpeakerTracks.Overlaps(spans, tracks, audio.DurationSeconds);

        return Tidy([.. SpeakerTracks.ToTurns(spans, tracks, audio.DurationSeconds)]);
    }

    /// <summary>
    /// Splits any track that turns out to hold more than one voice.
    /// <para>
    /// The one failure that following speakers through the audio cannot catch by itself. Linking
    /// joins whoever was talking during the seconds two windows share, which is right whenever
    /// the segmentation model is right — and where it briefly puts one person in the other's
    /// slot, the link is made on genuinely shared audio and two people become one track. Every
    /// later stage then treats them as one person by construction: the conflict graph, the
    /// colouring and the grouping all take tracks as given.
    /// </para>
    /// <para>
    /// So each track is listened to. Its own stretches are embedded and clustered among
    /// themselves, and a track that divides is divided. This was not worth attempting until the
    /// embeddings were mean-normalised: before that, one person and two measured 0.06 and 0.21,
    /// far too close to act on. They now measure 0.25 and 0.78.
    /// </para>
    /// </summary>
    private int[][] SplitImpureTracks(
        PcmAudio audio,
        IReadOnlyList<IReadOnlyList<IReadOnlyList<(double Start, double End)>>> spans,
        int[][] tracks,
        double threshold,
        CancellationToken cancellationToken)
    {
        var regions = new Dictionary<int, List<(double Start, double End)>>();

        for (var w = 0; w < spans.Count; w++)
        {
            for (var s = 0; s < spans[w].Count; s++)
            {
                var track = tracks[w][s];
                if (track == SpeakerTracks.Silent)
                {
                    continue;
                }

                regions.TryAdd(track, []);
                regions[track].AddRange(spans[w][s]);
            }
        }

        var split = new Dictionary<int, List<(double Start, double End, int Group)>>();
        var next = tracks.SelectMany(window => window).DefaultIfEmpty(0).Max() + 1;

        foreach (var (track, all) in regions)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // The same speech arrives from every window covering it, so overlapping stretches are
            // joined before anything is embedded.
            var distinct = Distinct(all)
                .Where(r => r.End - r.Start >= ShortestRegionToJudge)
                .OrderByDescending(r => r.End - r.Start)
                .Take(RegionsPerTrack)
                .OrderBy(r => r.Start)
                .ToList();

            if (distinct.Count < 2)
            {
                continue;
            }

            var voices = new List<float[]>(distinct.Count);
            var kept = new List<(double Start, double End)>(distinct.Count);

            foreach (var region in distinct)
            {
                if (Embed(audio, region.Start, region.End) is { } embedding)
                {
                    voices.Add(embedding);
                    kept.Add(region);
                }
            }

            if (voices.Count < 2)
            {
                continue;
            }

            var labels = SpeakerClustering.Cluster(voices, threshold);
            if (labels.Distinct().Count() < 2)
            {
                continue;
            }

            // Only where both sides are substantial. One odd stretch is a bad embedding; several
            // seconds either way is two people.
            var sides = labels
                .Select((label, i) => (label, seconds: kept[i].End - kept[i].Start))
                .GroupBy(x => x.label)
                .Select(g => g.Sum(x => x.seconds))
                .OrderByDescending(x => x)
                .ToList();

            if (sides[1] < LeastConvincingSide)
            {
                continue;
            }

            split[track] = [.. kept.Select((r, i) => (r.Start, r.End, labels[i]))];
        }

        if (split.Count == 0)
        {
            return tracks;
        }

        var renumbered = new Dictionary<(int Track, int Group), int>();
        var result = tracks.Select(window => window.ToArray()).ToArray();

        for (var w = 0; w < spans.Count; w++)
        {
            for (var s = 0; s < spans[w].Count; s++)
            {
                var track = result[w][s];

                if (track == SpeakerTracks.Silent
                    || spans[w][s].Count == 0
                    || !split.TryGetValue(track, out var groups))
                {
                    continue;
                }

                var middle = (spans[w][s].Min(x => x.Start) + spans[w][s].Max(x => x.End)) / 2;

                // Whichever of the track's own stretches this piece sits nearest to.
                var group = groups
                    .OrderBy(g => Math.Abs(((g.Start + g.End) / 2) - middle))
                    .First()
                    .Group;

                if (group == 0)
                {
                    continue;
                }

                if (!renumbered.TryGetValue((track, group), out var id))
                {
                    id = next++;
                    renumbered[(track, group)] = id;
                }

                result[w][s] = id;
            }
        }

        return result;
    }

    /// <summary>Overlapping stretches joined into distinct ones.</summary>
    private static List<(double Start, double End)> Distinct(List<(double Start, double End)> spans)
    {
        var merged = new List<(double Start, double End)>();

        foreach (var span in spans.OrderBy(s => s.Start))
        {
            if (merged.Count > 0 && span.Start <= merged[^1].End)
            {
                merged[^1] = (merged[^1].Start, Math.Max(merged[^1].End, span.End));
                continue;
            }

            merged.Add(span);
        }

        return merged;
    }

    /// <summary>
    /// Shortest stretch worth embedding when judging a track. Below this the embedding says more
    /// about which words were spoken than about who spoke them.
    /// </summary>
    private const double ShortestRegionToJudge = 1.5;

    /// <summary>How many of a track to listen to. The longest stretches, which say most.</summary>
    private const int RegionsPerTrack = 12;

    /// <summary>
    /// How much speech the smaller half of a split must hold before the split is believed.
    /// </summary>
    private const double LeastConvincingSide = 2.0;

    /// <summary>
    /// Reduces the tracks to at most <paramref name="wanted"/> people, by what they sound like.
    /// <para>
    /// Each track is embedded from its own longest stretches of speech rather than span by span.
    /// A track usually holds tens of seconds of one person, and a minute of somebody talking
    /// identifies them where two seconds does not — which is why this step can be trusted on a
    /// recording where per-span clustering could not be.
    /// </para>
    /// </summary>
    private int[][] Merge(
        PcmAudio audio,
        IReadOnlyList<IReadOnlyList<IReadOnlyList<(double Start, double End)>>> spans,
        int[][] tracks,
        int wanted,
        double threshold,
        CancellationToken cancellationToken)
    {
        var present = tracks.SelectMany(window => window).Distinct().OrderBy(t => t).ToList();
        if (present.Count <= wanted)
        {
            return tracks;
        }

        var voices = new Dictionary<int, float[]>();

        foreach (var track in present)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // The longest few stretches this track was heard in, which is the best look at it
            // going. Short ones add noise without adding evidence.
            var best = Enumerable.Range(0, spans.Count)
                .SelectMany(w => Enumerable.Range(0, spans[w].Count)
                    .Where(s => tracks[w][s] == track)
                    .SelectMany(s => spans[w][s]))
                .OrderByDescending(span => span.End - span.Start)
                .Take(EmbeddingsPerTrack)
                .ToList();

            float[]? total = null;
            var counted = 0;

            foreach (var (start, end) in best)
            {
                if (Embed(audio, start, end) is not { } embedding)
                {
                    continue;
                }

                var unit = Unit(embedding);
                total ??= new float[unit.Length];

                for (var i = 0; i < unit.Length; i++)
                {
                    total[i] += unit[i];
                }

                counted++;
            }

            if (counted > 0)
            {
                voices[track] = Unit(total!);
            }
        }

        var known = present.Where(voices.ContainsKey).ToList();
        if (known.Count <= wanted)
        {
            return tracks;
        }

        // Grouped by voice, but never putting two tracks together that the segmentation model
        // saw talking at the same moment. On a recording where the voices are hard to tell
        // apart, that constraint is most of what keeps everyone from collapsing into one person.
        var (edges, order) = ConflictsAmong(spans, tracks);

        var positionOf = order.Select((track, i) => (track, i)).ToDictionary(x => x.track, x => x.i);
        var knownAt = known.Select((track, i) => (track, i)).ToDictionary(x => x.track, x => x.i);

        var constraints = known
            .Select(track => positionOf.TryGetValue(track, out var at)
                ? (IReadOnlyList<int>)[.. edges[at]
                    .Select(other => knownAt.TryGetValue(order[other], out var k) ? k : -1)
                    .Where(k => k >= 0)]
                : [])
            .ToList();

        var labels = SpeakerTracks.GroupWithConstraints(
            [.. known.Select(track => voices[track])], constraints, wanted);

        var moved = known
            .Select((track, i) => (track, label: labels[i]))
            .ToDictionary(x => x.track, x => x.label);

        // A track too quiet to embed keeps its own number rather than being guessed at.
        var spare = moved.Count == 0 ? 0 : moved.Values.Max() + 1;

        return [.. tracks.Select(window =>
            window.Select(track => moved.TryGetValue(track, out var label) ? label : spare++).ToArray())];
    }

    /// <summary>
    /// Which tracks were heard talking at the same moment as which others, and so cannot be the
    /// same person. Straight from the segmentation model; no voices involved.
    /// </summary>
    /// <param name="spans">Per window, per local speaker, when they were talking.</param>
    /// <param name="tracks">Track numbers from <see cref="SpeakerTracks.Link"/>.</param>
    /// <returns>
    /// Edges indexed by position, referring to other positions, alongside the track each
    /// position stands for. Positions rather than track numbers because the colouring walks the
    /// edges as an array, and mixing the two identifier spaces is a bug that reads as a working
    /// program.
    /// </returns>
    private static (IReadOnlyList<IReadOnlyList<int>> Edges, IReadOnlyList<int> Tracks) ConflictsAmong(
        IReadOnlyList<IReadOnlyList<IReadOnlyList<(double Start, double End)>>> spans,
        int[][] tracks)
    {
        var index = new Dictionary<int, int>();
        var sets = new List<HashSet<int>>();

        int Slot(int track)
        {
            if (!index.TryGetValue(track, out var at))
            {
                at = sets.Count;
                index[track] = at;
                sets.Add([]);
            }

            return at;
        }

        for (var w = 0; w < spans.Count; w++)
        {
            var active = new List<int>();

            for (var s = 0; s < spans[w].Count; s++)
            {
                var track = tracks[w][s];
                if (track == SpeakerTracks.Silent)
                {
                    continue;
                }

                Slot(track);

                if (spans[w][s].Sum(span => span.End - span.Start) >= MinimumTurnSeconds)
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
                        sets[Slot(one)].Add(Slot(other));
                    }
                }
            }
        }

        var order = new int[sets.Count];
        foreach (var (track, at) in index)
        {
            order[at] = track;
        }

        return ([.. sets.Select(set => (IReadOnlyList<int>)[.. set])], order);
    }

    /// <summary>
    /// How many of a track's stretches to average into its voice. Enough to cover a change of
    /// microphone distance or tone; few enough that the tail of short fragments cannot outvote
    /// the long stretches that actually identify someone.
    /// </summary>
    private const int EmbeddingsPerTrack = 8;

    /// <summary>
    /// One pass of the segmentation model: which local speakers were talking, and when.
    /// <para>
    /// "Local" because the numbering means nothing outside the window. The segmentation model
    /// answers "how many people are talking here and when does each start and stop" without
    /// identifying anyone, and it stays reliable on audio far too poor for the embedding model
    /// to work on. Joining those local tracks into people across a whole recording is the job
    /// the embeddings do, and the job that fails first.
    /// </para>
    /// </summary>
    /// <param name="StartSeconds">Where the window begins in the recording.</param>
    /// <param name="Speakers">Spans per local speaker, indexed by the window's own numbering.</param>
    public sealed record Window(
        double StartSeconds,
        IReadOnlyList<IReadOnlyList<(double Start, double End)>> Speakers);

    /// <summary>
    /// Runs the segmentation model across the recording and reports each window untouched, with
    /// no deduplication and no clustering. For diagnosis, and for anything that wants to work
    /// from local speaker tracks rather than from embeddings.
    /// </summary>
    public IReadOnlyList<Window> DescribeWindows(
        PcmAudio audio,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);

        var shift = (int)(_windowSamples * WindowShiftFraction);
        var windows = new List<Window>();
        var contested = new List<(double Start, double End)>();

        for (var offset = 0; offset < audio.Samples.Length; offset += shift)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Scanning is most of the work, so it gets most of the bar. The embeddings that
            // follow are a few dozen short runs and finish quickly by comparison.
            progress?.Report(ScanShare * offset / audio.Samples.Length);

            var window = new float[_windowSamples];
            var available = Math.Min(_windowSamples, audio.Samples.Length - offset);

            if (available < _windowSamples / 4)
            {
                break;
            }

            Array.Copy(audio.Samples, offset, window, 0, available);

            using var outputs = _segmentation.Run(
            [
                NamedOnnxValue.CreateFromTensor(
                    _segmentation.InputMetadata.Keys.First(),
                    new DenseTensor<float>(window, [1, 1, _windowSamples])),
            ]);

            var tensor = outputs.First().AsTensor<float>();
            var frames = tensor.Dimensions[1];
            var scores = tensor.ToArray();

            var active = PowersetDecoder.Decode(scores, frames, _mapping, _localSpeakers);
            var windowStart = offset / (double)audio.SampleRate;

            var bySpeaker = new List<IReadOnlyList<(double Start, double End)>>(_localSpeakers);
            for (var speaker = 0; speaker < _localSpeakers; speaker++)
            {
                bySpeaker.Add(
                    [.. SpansFor(active, frames, speaker, windowStart, available, audio.SampleRate)]);
            }

            windows.Add(new Window(windowStart, bySpeaker));

            // The frames where the model heard two voices at once, kept as times. Everything
            // else this method feeds resolves each moment to one winner; this is the record of
            // the moments that had no clean winner to give.
            var limit = windowStart + (available / (double)audio.SampleRate);

            foreach (var (first, until) in
                PowersetDecoder.OverlappedFrames(active, frames, _localSpeakers))
            {
                var start = windowStart + _receptiveFieldStart + (first * _receptiveFieldShift);
                var end = Math.Min(
                    limit, windowStart + _receptiveFieldStart + (until * _receptiveFieldShift));

                if (end > start)
                {
                    contested.Add((start, end));
                }
            }
        }

        // Windows overlap by ninety percent, so the same contested moment arrives many times
        // over; the union is the moment itself. The tracking path overwrites this afterwards
        // with its own between-window measure, which was proven on real recordings first.
        LastOverlaps = MergeSpans(contested);

        return windows;
    }

    /// <summary>Overlapping or touching spans folded into one, in order.</summary>
    private static IReadOnlyList<(double Start, double End)> MergeSpans(
        List<(double Start, double End)> spans)
    {
        var merged = new List<(double Start, double End)>();

        foreach (var span in spans.OrderBy(s => s.Start))
        {
            if (merged.Count > 0 && span.Start <= merged[^1].End)
            {
                merged[^1] = (merged[^1].Start, Math.Max(merged[^1].End, span.End));
            }
            else
            {
                merged.Add(span);
            }
        }

        return merged;
    }

    private List<(double Start, double End)> CollectSpeechSpans(
        PcmAudio audio,
        IProgress<double>? progress,
        CancellationToken cancellationToken)
    {
        var spans = DescribeWindows(audio, progress, cancellationToken)
            .SelectMany(window => window.Speakers.SelectMany(spans => spans))
            .ToList();

        // Windows overlap by ninety percent, so the same speech arrives many times over. Near
        // duplicates are dropped rather than merged: merging would join neighbours across a
        // speaker change, which is the thing this is avoiding.
        return Deduplicate(spans);
    }

    /// <summary>
    /// Drops spans that say the same thing as one already kept. Two spans covering nearly the
    /// same seconds come from adjacent windows seeing the same voice; the longer is the better
    /// look at it.
    /// </summary>
    private List<(double Start, double End)> Deduplicate(List<(double Start, double End)> spans)
    {
        var ordered = spans
            .OrderByDescending(s => s.End - s.Start)
            .ToList();

        var kept = new List<(double Start, double End)>();
        var nested = 0;

        foreach (var span in ordered)
        {
            var duration = span.End - span.Start;
            var swallowedBy = 0.0;

            var covered = kept.Any(k =>
            {
                var overlap = Math.Min(k.End, span.End) - Math.Max(k.Start, span.Start);

                if (overlap <= 0 || overlap < duration * 0.75)
                {
                    return false;
                }

                swallowedBy = Math.Max(swallowedBy, k.End - k.Start);
                return true;
            });

            if (!covered)
            {
                kept.Add(span);
                continue;
            }

            if (swallowedBy > duration * NestedRatio)
            {
                nested++;
            }
        }

        LastSpansFound = spans.Count;
        LastSpansKept = kept.Count;
        LastNestedDropped = nested;

        return kept.OrderBy(s => s.Start).ToList();
    }

    /// <summary>
    /// How much longer a covering span must be before the two are counted as different speech.
    /// <para>
    /// Used for reporting only. Keeping the spans this identifies was tried and was worse: it
    /// took 109 kept spans to 261 and the turns from 22 to 25, while the speaker count went from
    /// a correct five to seven. Short spans embed badly and cluster into people who are not
    /// there, and three turns is not worth two invented speakers.
    /// </para>
    /// </summary>
    private const double NestedRatio = 2.0;

    /// <summary>How many speech spans the windows produced, before near-duplicates were dropped.</summary>
    public int LastSpansFound { get; private set; }

    /// <summary>How many survived.</summary>
    public int LastSpansKept { get; private set; }

    /// <summary>
    /// How many were dropped despite being far shorter than what covered them.
    /// <para>
    /// Overlapping windows hear the same speech many times, and those copies are the same length,
    /// so dropping one is right. A short span sitting inside a much longer one is a different
    /// thing wearing the same shape: two voices at once, one of them briefly. Counted separately
    /// because the rule cannot currently tell them apart and this is how much that costs.
    /// </para>
    /// </summary>
    public int LastNestedDropped { get; private set; }

    /// <summary>Runs of consecutive active frames for one local speaker, as times.</summary>
    private IEnumerable<(double Start, double End)> SpansFor(
        bool[] active,
        int frames,
        int speaker,
        double windowStart,
        int availableSamples,
        int sampleRate)
    {
        var limit = windowStart + (availableSamples / (double)sampleRate);
        int? runStart = null;

        for (var frame = 0; frame <= frames; frame++)
        {
            var isActive = frame < frames && active[(frame * _localSpeakers) + speaker];

            if (isActive && runStart is null)
            {
                runStart = frame;
            }
            else if (!isActive && runStart is { } begin)
            {
                var start = windowStart + _receptiveFieldStart + (begin * _receptiveFieldShift);
                var end = windowStart + _receptiveFieldStart + (frame * _receptiveFieldShift);

                runStart = null;

                // The last window is zero-padded; anything the model hears in the padding is an
                // artefact of the padding.
                if (end > limit)
                {
                    end = limit;
                }

                if (end - start >= MinimumTurnSeconds)
                {
                    yield return (start, end);
                }
            }
        }
    }

    /// <summary>
    /// Joins consecutive turns the clustering gave the same speaker, and only those.
    /// <para>
    /// The gap allowed is small on purpose. Someone pausing for breath mid-thought should read
    /// as one turn, but a second of silence is also exactly where the other person starts
    /// talking, and joining across that is how a speaker change gets lost after surviving
    /// everything else.
    /// </para>
    /// </summary>
    private static IReadOnlyList<SpeakerTurn> Tidy(List<SpeakerTurn> turns)
    {
        var merged = new List<SpeakerTurn>();

        foreach (var turn in turns)
        {
            if (merged.Count > 0
                && merged[^1].Speaker == turn.Speaker
                && turn.StartSeconds - merged[^1].EndSeconds < 0.75)
            {
                merged[^1] = merged[^1] with { EndSeconds = Math.Max(merged[^1].EndSeconds, turn.EndSeconds) };
                continue;
            }

            merged.Add(turn);
        }

        return merged;
    }

    /// <summary>Embeds an explicit span. Exposed so the distances can be measured directly.</summary>
    public float[]? EmbedSpan(PcmAudio audio, double startSeconds, double endSeconds) =>
        Embed(audio, startSeconds, endSeconds);

    /// <summary>A speaker embedding for one stretch of audio, or null when it is too short.</summary>
    private float[]? Embed(PcmAudio audio, double startSeconds, double endSeconds)
    {
        var start = Math.Clamp((int)(startSeconds * audio.SampleRate), 0, audio.Samples.Length);
        var end = Math.Clamp((int)(endSeconds * audio.SampleRate), start, audio.Samples.Length);

        if (end - start < MinimumTurnSeconds * audio.SampleRate)
        {
            return null;
        }

        // Mean-normalised, as WeSpeaker's own pipeline does. Without it the embedding is
        // dominated by the recording channel rather than the voice, and every speaker in one
        // recording looks like every other: on a two-person debate, different speakers measured
        // 0.21 apart where 0.42 means "different people", so nothing could be told apart. With
        // it, the same pairs measure 0.73 to 0.84 and the same speaker 0.18 to 0.30 — the
        // threshold sits in the gap instead of above everything.
        var features = _fbank.Compute(audio.Samples.AsSpan(start, end - start), subtractMean: true);
        if (features.Length == 0)
        {
            return null;
        }

        var frames = features.Length / _fbank.MelBins;

        using var outputs = _embedding.Run(
        [
            NamedOnnxValue.CreateFromTensor(
                _embedding.InputMetadata.Keys.First(),
                new DenseTensor<float>(features, [1, frames, _fbank.MelBins])),
        ]);

        return outputs.First().AsTensor<float>().ToArray();
    }

    /// <summary>
    /// Attaches speakers to a transcript, splitting segments that span more than one turn.
    /// The work is in <see cref="SpeakerAttribution"/>, which is pure and tested.
    /// </summary>
    public static IReadOnlyList<TranscriptSegment> Attribute(
        IReadOnlyList<TranscriptSegment> segments,
        IReadOnlyList<SpeakerTurn> turns) =>
        SpeakerAttribution.Apply(segments, turns);

    private static int Read(IReadOnlyDictionary<string, string> metadata, string key, int fallback) =>
        metadata.TryGetValue(key, out var value) && int.TryParse(value, out var parsed) ? parsed : fallback;

    public void Dispose()
    {
        _segmentation.Dispose();
        _embedding.Dispose();
    }
}
