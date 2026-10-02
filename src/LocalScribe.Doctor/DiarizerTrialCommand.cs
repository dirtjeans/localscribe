using System.Diagnostics;
using LocalScribe.Core.Archive;
using LocalScribe.Core.Audio;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Transcription;
using LocalScribe.Onnx;

namespace LocalScribe.Doctor;

/// <summary>
/// Implements <c>--diarize-trial</c>: runs the diarizer the app uses and the Nemotron model
/// over the same recording, and reports where they differ.
/// <para>
/// It cannot say which is right. There is no labelled truth for anyone's own recordings, and
/// agreement between two systems proves only that they agree. What it can do is put the
/// evidence where a person can settle it quickly: how many people each heard, how the time
/// divides between them, and every stretch where they disagree about who is talking — with the
/// words, when the recording is a saved transcript, so the reader can tell from what was said.
/// </para>
/// </summary>
internal static class DiarizerTrialCommand
{
    private const double Frame = 0.01;

    /// <summary>Shortest disagreement worth a line. Shorter ones are boundaries drawn a little apart.</summary>
    private const double WorthListing = 1.5;

    public static int Run(
        string path,
        string currentModels,
        string nemotronModels,
        string modelRoot,
        bool printTranscripts,
        string? only,
        bool fullPrecision,
        bool allCores,
        bool printTurns,
        string? threads = null,
        string? speakers = null)
    {
        if (!File.Exists(path))
        {
            Console.Error.WriteLine($"No such file: {path}");
            return 1;
        }

        Heading("Diarizer trial");
        Console.WriteLine($"  Recording  {path}");

        PcmAudio audio;
        IReadOnlyList<TranscriptSegment> saved = [];

        if (TranscriptArchive.IsArchive(path))
        {
            var contents = TranscriptArchive.Load(path);
            audio = contents.Audio;
            saved = contents.Segments;
        }
        else
        {
            audio = WavReader.Read(path);
        }

        audio.EnsureWhisperFormat();

        // The doctor's convention is the capped budget, which is the product requirement. The
        // app on Windows still hands the pyannote sessions every core, so --all-cores exists to
        // time both the way the app does today.
        var plan = allCores ? null : AcceleratorPlanner.Plan(DeviceProbe.Probe());

        // The budget depends on what else the plan found to do: with the encoder on the NPU the
        // app keeps as few as two CPU threads for everything else. --threads times a model
        // under the budget it would really be given.
        if (plan is not null && int.TryParse(threads, out var count) && count > 0)
        {
            plan = plan with { CpuBudget = plan.CpuBudget with { IntraOpThreads = count } };
        }

        Console.WriteLine($"  Duration   {audio.DurationSeconds:F1}s");
        Console.WriteLine($"  Threads    {(plan is null ? "every core" : $"{plan.CpuBudget.IntraOpThreads} (the capped budget)")}");

        var process = Process.GetCurrentProcess();

        Outcome? current = null;
        Outcome? nemotron = null;

        try
        {
            if (only is null or "current")
            {
                current = RunCurrent(audio, currentModels, plan, process);
                Report(current, audio);
            }

            if (only is null or "nemotron")
            {
                nemotron = RunNemotron(
                    audio,
                    nemotronModels,
                    plan,
                    fullPrecision,
                    process,
                    int.TryParse(speakers, out var wanted) && wanted > 0 ? wanted : null,
                    currentModels);
                Report(nemotron, audio);
            }
        }
        catch (FileNotFoundException exception)
        {
            Console.Error.WriteLine($"  {exception.Message}");
            return 1;
        }

        if (printTurns)
        {
            foreach (var outcome in new[] { current, nemotron }.OfType<Outcome>())
            {
                Heading($"Turns — {outcome.Name}");

                foreach (var turn in outcome.Turns)
                {
                    Console.WriteLine(
                        $"  {turn.StartSeconds,8:F2} - {turn.EndSeconds,8:F2}  ({turn.DurationSeconds,6:F2}s)  {turn.Label}");
                }
            }
        }

        if (current is not null && nemotron is not null)
        {
            Compare(current, nemotron, audio, saved);

            if (saved.Count > 0)
            {
                CompareWords(current, nemotron, audio, saved, modelRoot, plan, printTranscripts);
            }
        }

        return 0;
    }

    private sealed record Outcome(
        string Name,
        IReadOnlyList<SpeakerTurn> Turns,
        IReadOnlyList<(double Start, double End)> Overlaps,
        TimeSpan Load,
        TimeSpan Took,
        TimeSpan Processor,
        long PeakBytes,
        string? Detail);

    private static Outcome RunCurrent(PcmAudio audio, string models, ExecutionPlan? plan, Process process)
    {
        var method = DiarizationChoice.Read(models);

        var watch = Stopwatch.StartNew();
        using var diarizer = SpeakerDiarizer.Load(models, plan);
        var load = watch.Elapsed;

        process.Refresh();
        var before = process.TotalProcessorTime;
        watch.Restart();

        // The app's own call, with the app's own defaults: no speaker count given, and the
        // method recorded beside the models.
        var turns = method == DiarizationMethod.Voices
            ? diarizer.Diarize(audio)
            : diarizer.DiarizeByTracking(audio);

        var took = watch.Elapsed;
        process.Refresh();

        return new Outcome(
            $"current (pyannote, {DiarizationChoice.Name(method)})",
            turns,
            diarizer.LastOverlaps,
            load,
            took,
            process.TotalProcessorTime - before,
            process.PeakWorkingSet64,
            null);
    }

    private static Outcome RunNemotron(
        PcmAudio audio,
        string models,
        ExecutionPlan? plan,
        bool fullPrecision,
        Process process,
        int? wanted = null,
        string? voiceModels = null)
    {
        var watch = Stopwatch.StartNew();
        using var diarizer = SortformerDiarizer.Load(models, plan, fullPrecision);
        var load = watch.Elapsed;

        process.Refresh();
        var before = process.TotalProcessorTime;
        watch.Restart();

        var activity = diarizer.Diarize(audio);

        // Raw probabilities, frame by frame, for checking this port against the reference
        // implementation outside .NET. Row-major float32, eight per frame.
        if (Environment.GetEnvironmentVariable("LOCALSCRIBE_DUMP_ACTIVITY") is { Length: > 0 } dump)
        {
            using var writer = new BinaryWriter(File.Create(dump));
            for (var frame = 0; frame < activity.Frames; frame++)
            {
                for (var speaker = 0; speaker < activity.Speakers; speaker++)
                {
                    writer.Write(activity[frame, speaker]);
                }
            }
        }
        var turns = activity.Turns();
        var overlaps = activity.Overlaps();
        var fitted = string.Empty;

        // The app's answer to "How many speakers?": extras folded into whoever they sound like.
        if (wanted is { } count && turns.Select(t => t.Speaker).Distinct().Count() > count && voiceModels is not null)
        {
            using var voices = SpeakerDiarizer.Load(voiceModels, plan);
            var prints = SortformerDiarizer.VoicePrints(voices, audio, turns);
            turns = SpeakerMerging.ToCount(turns, prints, count);
            overlaps = SpeakerMerging.Overlaps(turns);
            fitted = $", fitted to {count} speakers";
        }

        var took = watch.Elapsed;
        process.Refresh();

        var (features, embedding, transformer) = diarizer.LastTimings;

        return new Outcome(
            $"nemotron ({(fullPrecision ? "float32" : "int8")}{fitted})",
            turns,
            overlaps,
            load,
            took,
            process.TotalProcessorTime - before,
            process.PeakWorkingSet64,
            $"features {features.TotalSeconds:F1}s, embedding {embedding.TotalSeconds:F1}s, "
            + $"transformer {transformer.TotalSeconds:F1}s; {activity.RawRuns()} runs before tidying");
    }

    private static void Report(Outcome outcome, PcmAudio audio)
    {
        Heading(outcome.Name);

        var bySpeaker = outcome.Turns
            .GroupBy(turn => turn.Speaker)
            .OrderBy(group => group.Key)
            .Select(group => (Speaker: group.Key, Seconds: group.Sum(t => t.DurationSeconds), Turns: group.Count()))
            .ToList();

        var speech = bySpeaker.Sum(s => s.Seconds);

        Console.WriteLine($"  Speakers   {bySpeaker.Count}");
        Console.WriteLine($"  Turns      {outcome.Turns.Count}");
        Console.WriteLine($"  Speech     {speech:F1}s of {audio.DurationSeconds:F1}s");
        Console.WriteLine($"  Crosstalk  {outcome.Overlaps.Sum(o => o.End - o.Start):F1}s in {outcome.Overlaps.Count} stretch(es)");
        Console.WriteLine($"  Loaded in  {outcome.Load.TotalSeconds:F1}s");
        Console.WriteLine($"  Took       {outcome.Took.TotalSeconds:F1}s "
            + $"({audio.DurationSeconds / Math.Max(0.001, outcome.Took.TotalSeconds):F1}x real time)");
        Console.WriteLine($"  Processor  {outcome.Processor.TotalSeconds:F1}s of CPU time "
            + $"({outcome.Processor.TotalSeconds / Math.Max(0.001, outcome.Took.TotalSeconds):F1} cores busy on average)");
        Console.WriteLine($"  Peak RAM   {outcome.PeakBytes / (1024.0 * 1024):F0} MB for the process so far");

        if (outcome.Detail is not null)
        {
            Console.WriteLine($"  Detail     {outcome.Detail}");
        }

        Console.WriteLine();

        // With where to listen: whether a minor speaker is a person or an artefact is settled
        // in seconds by ear, and not at all by a table.
        foreach (var (speaker, seconds, turns) in bySpeaker)
        {
            var own = outcome.Turns.Where(t => t.Speaker == speaker).ToList();
            var longest = own.MaxBy(t => t.DurationSeconds)!;

            Console.WriteLine(
                $"    Speaker {speaker + 1,-3} {seconds,8:F1}s  {seconds / Math.Max(0.001, speech),6:P0}  in {turns,4} turn(s)"
                + $"   first at {Clock(own.Min(t => t.StartSeconds))}, longest {longest.DurationSeconds,5:F1}s at {Clock(longest.StartSeconds)}");
        }
    }

    private static void Compare(
        Outcome current,
        Outcome nemotron,
        PcmAudio audio,
        IReadOnlyList<TranscriptSegment> saved)
    {
        var frames = (int)(audio.DurationSeconds / Frame);

        var ours = Paint(current.Turns, frames);
        var theirs = Paint(nemotron.Turns, frames);

        var ourCount = current.Turns.Count == 0 ? 0 : current.Turns.Max(t => t.Speaker) + 1;
        var theirCount = nemotron.Turns.Count == 0 ? 0 : nemotron.Turns.Max(t => t.Speaker) + 1;

        // Only frames where each hears exactly one person can be compared label to label.
        // Crosstalk and silence are counted apart.
        var shared = new int[ourCount, theirCount];
        int bothQuiet = 0, onlyOurs = 0, onlyTheirs = 0, comparable = 0, tangled = 0;

        // How loud the audio is under each verdict. The two disagree about a lot of time that
        // one calls speech and the other silence, and the signal itself can say which it is
        // without either model's opinion: a pause inside a turn is as quiet as silence, and
        // speech that was missed is as loud as speech.
        double quietPower = 0, oursPower = 0, theirsPower = 0, speechPower = 0;
        var samplesPerFrame = (int)(audio.SampleRate * Frame);

        for (var frame = 0; frame < frames; frame++)
        {
            var a = ours[frame];
            var b = theirs[frame];

            double power = 0;
            var from = frame * samplesPerFrame;
            var to = Math.Min(audio.Samples.Length, from + samplesPerFrame);
            for (var i = from; i < to; i++)
            {
                power += audio.Samples[i] * (double)audio.Samples[i];
            }

            power /= Math.Max(1, to - from);

            if (a == Quiet && b == Quiet)
            {
                bothQuiet++;
                quietPower += power;
            }
            else if (b == Quiet)
            {
                onlyOurs++;
                oursPower += power;
            }
            else if (a == Quiet)
            {
                onlyTheirs++;
                theirsPower += power;
            }
            else
            {
                speechPower += power;
            }

            if (a == Quiet || b == Quiet)
            {
                continue;
            }
            if (a == Several || b == Several)
            {
                tangled++;
            }
            else
            {
                comparable++;
                shared[a, b]++;
            }
        }

        // Each Nemotron speaker is paired with the current speaker they share most time with,
        // largest pairing first, one to one. Greedy rather than optimal; with the handful of
        // real speakers a recording has, the two only differ when the answer is already a mess.
        var ourMatch = Enumerable.Repeat(-1, ourCount).ToArray();
        var theirMatch = Enumerable.Repeat(-1, theirCount).ToArray();

        while (true)
        {
            var best = 0;
            var (bestA, bestB) = (-1, -1);

            for (var a = 0; a < ourCount; a++)
            {
                for (var b = 0; b < theirCount; b++)
                {
                    if (ourMatch[a] < 0 && theirMatch[b] < 0 && shared[a, b] > best)
                    {
                        (best, bestA, bestB) = (shared[a, b], a, b);
                    }
                }
            }

            if (bestA < 0)
            {
                break;
            }

            ourMatch[bestA] = bestB;
            theirMatch[bestB] = bestA;
        }

        var agreed = 0;
        for (var a = 0; a < ourCount; a++)
        {
            if (ourMatch[a] >= 0)
            {
                agreed += shared[a, ourMatch[a]];
            }
        }

        Heading("Where the time goes, current speaker by Nemotron speaker (seconds)");
        Console.Write("              ");
        for (var b = 0; b < theirCount; b++)
        {
            Console.Write($"{"N" + (b + 1),8}");
        }

        Console.WriteLine();

        for (var a = 0; a < ourCount; a++)
        {
            Console.Write($"  Speaker {a + 1,-4}");
            for (var b = 0; b < theirCount; b++)
            {
                var seconds = shared[a, b] * Frame;
                Console.Write(seconds < 0.05 ? $"{".",8}" : $"{seconds,8:F1}");
            }

            Console.WriteLine(ourMatch[a] >= 0 ? $"   = N{ourMatch[a] + 1}" : "   (no counterpart)");
        }

        Heading("Agreement");
        Console.WriteLine($"  Both hear one speaker   {comparable * Frame,8:F1}s");
        Console.WriteLine($"    and name the same one {agreed * Frame,8:F1}s  ({agreed / (double)Math.Max(1, comparable):P1})");
        Console.WriteLine($"  Either hears crosstalk  {tangled * Frame,8:F1}s");
        Console.WriteLine($"  Speech to current only  {onlyOurs * Frame,8:F1}s");
        Console.WriteLine($"  Speech to Nemotron only {onlyTheirs * Frame,8:F1}s");
        Console.WriteLine($"  Silence to both         {bothQuiet * Frame,8:F1}s");

        Heading("How loud the audio is under each verdict (dB below full scale)");
        Console.WriteLine($"  Speech to both          {Level(speechPower, comparable + tangled),8:F1}");
        Console.WriteLine($"  Speech to current only  {Level(oursPower, onlyOurs),8:F1}");
        Console.WriteLine($"  Speech to Nemotron only {Level(theirsPower, onlyTheirs),8:F1}");
        Console.WriteLine($"  Silence to both         {Level(quietPower, bothQuiet),8:F1}");
        Console.WriteLine("  Time only one calls speech is a pause if it sits near the silence level, and");
        Console.WriteLine("  missed speech if it sits near the speech level.");

        Heading($"Disagreements of {WorthListing:F1}s or more");

        var listed = 0;
        var start = -1;
        var (wasA, wasB) = (-1, -1);

        for (var frame = 0; frame <= frames; frame++)
        {
            var a = frame < frames ? ours[frame] : Quiet;
            var b = frame < frames ? theirs[frame] : Quiet;
            var differs = a >= 0 && b >= 0 && ourMatch[a] != b;

            if (differs && start >= 0 && (a, b) == (wasA, wasB))
            {
                continue;
            }

            if (start >= 0 && (frame - start) * Frame >= WorthListing)
            {
                listed++;
                var named = theirMatch[wasB] >= 0 ? $"their N{wasB + 1} = Speaker {theirMatch[wasB] + 1}" : $"their N{wasB + 1}, who has no counterpart";

                Console.WriteLine(
                    $"  {Clock(start * Frame)} - {Clock(frame * Frame)}  ({(frame - start) * Frame,5:F1}s)  "
                    + $"current: Speaker {wasA + 1}   nemotron: {named}");

                var words = Said(saved, start * Frame, frame * Frame);
                if (words.Length > 0)
                {
                    Console.WriteLine($"      \"{words}\"");
                }
            }

            (start, wasA, wasB) = differs ? (frame, a, b) : (-1, -1, -1);
        }

        if (listed == 0)
        {
            Console.WriteLine("  (none)");
        }

        if (saved.Count > 0)
        {
            BySegment(current, nemotron, theirMatch, saved);
        }
    }

    /// <summary>
    /// The saved transcript's segments, each with who it was saved under and who each diarizer
    /// gives most of its time to now. Only the rows where the two diarizers differ are printed.
    /// </summary>
    private static void BySegment(
        Outcome current,
        Outcome nemotron,
        int[] theirMatch,
        IReadOnlyList<TranscriptSegment> saved)
    {
        Heading("Saved segments the two attribute differently");

        var spoken = saved.Where(s => s.Text.Any(char.IsLetter)).ToList();
        var differing = 0;

        foreach (var segment in spoken)
        {
            var a = Dominant(current.Turns, segment);
            var b = Dominant(nemotron.Turns, segment);

            var translated = b >= 0 ? theirMatch[b] : -1;

            // Agreeing that nobody is talking is agreement too: a saved segment both hear as
            // silence is usually words Whisper found in music or room noise.
            if ((a == translated && a >= 0) || (a < 0 && b < 0))
            {
                continue;
            }

            differing++;

            var ours = a >= 0 ? $"Speaker {a + 1}" : "nobody";
            var theirs = b < 0 ? "nobody" : translated >= 0 ? $"Speaker {translated + 1}" : $"N{b + 1} (new)";

            Console.WriteLine(
                $"  {Clock(segment.StartSeconds)}  saved: {segment.Speaker ?? "-",-10} now: {ours,-10} nemotron: {theirs,-10}");
            Console.WriteLine($"      \"{Shorten(segment.Text.Trim(), 110)}\"");
        }

        Console.WriteLine();
        Console.WriteLine($"  {differing} of {spoken.Count} segments differ.");
    }

    /// <summary>
    /// Runs the saved transcript's words through the app's own attribution with each
    /// diarizer's turns, and reports what the steps downstream of the diarizer did with them.
    /// <para>
    /// Those steps were tuned while pyannote supplied the turns, and two of their rules are
    /// about pyannote's habits rather than about speech: turns under half a second are ignored
    /// as window-vote flicker, and a word no turn covers is handed to a turn chosen by list
    /// order. Nemotron's turns are sharper and stop at pauses, so both rules may now fire on
    /// real speech. This counts how often they do, on real word timings, so whether to retune
    /// them is a measurement rather than a guess.
    /// </para>
    /// </summary>
    private static void CompareWords(
        Outcome current,
        Outcome nemotron,
        PcmAudio audio,
        IReadOnlyList<TranscriptSegment> saved,
        string modelRoot,
        ExecutionPlan? plan,
        bool printTranscripts)
    {
        if (ForcedAligner.Find(modelRoot) is not { } directory)
        {
            Console.WriteLine();
            Console.WriteLine("  (no alignment model, so the word-level comparison was skipped)");
            return;
        }

        Heading("Word-level attribution, as the app does it");

        using var aligner = ForcedAligner.Load(directory, plan);
        var scores = aligner.Scan(audio);

        if (scores is null)
        {
            Console.WriteLine("  The recording could not be scanned.");
            return;
        }

        // The saved segments without their old labels or crosstalk marks, so that neither
        // engine starts from the answer the old pipeline gave.
        var words = aligner.AlignAll(scores, saved, CancellationToken.None);
        var timed = saved
            .Select((segment, i) => new TimedSegment(
                segment with { Speaker = null, Overlapped = false },
                words[i] ?? []))
            .ToList();

        var totalWords = timed.Sum(t => t.Words.Count(w => w.EndSeconds > w.StartSeconds));
        Console.WriteLine($"  Words      {totalWords} measured, in {timed.Count} segments");
        Console.WriteLine();
        Console.WriteLine("                                      current   nemotron");

        var a = Attribute(timed, current);
        var b = Attribute(timed, nemotron);

        Row("Turns ignored as flicker (< 0.5s)", Short(current.Turns).Count, Short(nemotron.Turns).Count);
        Row("  ... seconds of speech in them", Short(current.Turns).Sum(t => t.DurationSeconds), Short(nemotron.Turns).Sum(t => t.DurationSeconds));
        Row("Words no kept turn covers", Orphans(timed, current.Turns), Orphans(timed, nemotron.Turns));
        Row("Speaker changes in the transcript", Changes(a), Changes(b));
        Row("Pieces marked as crosstalk", a.Count(p => p.Segment.Overlapped), b.Count(p => p.Segment.Overlapped));

        var ours = Labels(a);
        var theirs = Labels(b);

        if (ours.Count != theirs.Count)
        {
            Console.WriteLine($"  The two attributions kept different numbers of words ({ours.Count} and {theirs.Count}).");
            return;
        }

        // Nemotron's names translated into the current pipeline's, by the words they share.
        var map = theirs
            .Select((pair, i) => (Theirs: pair.Label, Ours: ours[i].Label))
            .GroupBy(pair => pair.Theirs)
            .ToDictionary(
                group => group.Key,
                group => group.GroupBy(pair => pair.Ours).MaxBy(g => g.Count())!.Key);

        var agree = ours.Where((word, i) => map[theirs[i].Label] == word.Label).Count();
        Console.WriteLine();
        Console.WriteLine($"  Words given to the same person: {agree} of {ours.Count} ({agree / (double)Math.Max(1, ours.Count):P1})");

        Heading("Runs of words the two give to different people");

        var listed = 0;
        for (var i = 0; i < ours.Count;)
        {
            if (map[theirs[i].Label] == ours[i].Label)
            {
                i++;
                continue;
            }

            var start = i;
            while (i < ours.Count && map[theirs[i].Label] != ours[i].Label)
            {
                i++;
            }

            listed++;
            var text = string.Join(" ", ours.Skip(start).Take(i - start).Select(w => w.Word.Text.Trim()));
            Console.WriteLine(
                $"  {Clock(ours[start].Word.StartSeconds)}  current: {ours[start].Label,-10} nemotron: {map[theirs[start].Label],-10} "
                + $"({i - start} word{(i - start == 1 ? "" : "s")})");
            Console.WriteLine($"      \"{Shorten(text, 140)}\"");
        }

        if (listed == 0)
        {
            Console.WriteLine("  (none)");
        }

        if (printTranscripts)
        {
            foreach (var (name, pieces) in new[] { ("current", a), ("nemotron", b) })
            {
                Heading($"Transcript as attributed - {name}");

                string? speaker = null;
                foreach (var piece in pieces.Where(p => p.Segment.Text.Any(char.IsLetter)))
                {
                    if (piece.Segment.Speaker != speaker)
                    {
                        speaker = piece.Segment.Speaker;
                        Console.WriteLine();
                        Console.Write($"  {Clock(piece.Segment.StartSeconds)}  {speaker}:");
                    }

                    Console.Write(piece.Segment.Overlapped
                        ? $" [crosstalk] {piece.Segment.Text.Trim()}"
                        : $" {piece.Segment.Text.Trim()}");
                }

                Console.WriteLine();
            }
        }

        static void Row(string what, double ours, double theirs) =>
            Console.WriteLine($"  {what,-34} {ours,8:0.#}   {theirs,8:0.#}");

        static List<SpeakerTurn> Short(IReadOnlyList<SpeakerTurn> turns) =>
            [.. turns.Where(t => t.DurationSeconds < WordLevelAttribution.ShortestTurnSeconds)];

        // A word whose end falls in no kept turn: the case the attribution resolves by list order.
        static int Orphans(IReadOnlyList<TimedSegment> timed, IReadOnlyList<SpeakerTurn> turns)
        {
            var kept = turns.Where(t => t.DurationSeconds >= WordLevelAttribution.ShortestTurnSeconds).ToList();

            return timed
                .SelectMany(t => t.Words)
                .Where(w => w.EndSeconds > w.StartSeconds)
                .Count(w =>
                {
                    var from = Math.Max(w.StartSeconds, w.EndSeconds - WordLevelAttribution.LongestWordSeconds);
                    return kept.All(t => t.OverlapWith(from, w.EndSeconds) <= 0);
                });
        }

        static int Changes(IReadOnlyList<TimedSegment> pieces)
        {
            var spoken = pieces.Where(p => p.Segment.Text.Any(char.IsLetter)).ToList();
            return spoken.Skip(1).Where((p, i) => p.Segment.Speaker != spoken[i].Segment.Speaker).Count();
        }

        static List<(WordTimings.Word Word, string Label)> Labels(IReadOnlyList<TimedSegment> pieces) =>
            [.. pieces.SelectMany(p => p.Words
                .Where(w => w.EndSeconds > w.StartSeconds)
                .Select(w => (w, p.Segment.Speaker ?? "-")))];
    }

    /// <summary>The app's attribution: words to turns, crosstalk marks, numbering by appearance.</summary>
    private static IReadOnlyList<TimedSegment> Attribute(IReadOnlyList<TimedSegment> timed, Outcome outcome)
    {
        var pieces = WordLevelAttribution.Apply(timed, outcome.Turns);
        pieces = CrosstalkMarks.Apply(pieces, outcome.Overlaps);

        var relabelled = SpeakerLabels.RenumberByAppearance([.. pieces.Select(p => p.Segment)]);
        return [.. pieces.Select((p, i) => p with { Segment = relabelled[i] })];
    }

    private static int Dominant(IReadOnlyList<SpeakerTurn> turns, TranscriptSegment segment)
    {
        var best = -1;
        double most = 0;

        foreach (var group in turns.GroupBy(t => t.Speaker))
        {
            var seconds = group.Sum(t => t.OverlapWith(segment.StartSeconds, segment.EndSeconds));
            if (seconds > most)
            {
                (best, most) = (group.Key, seconds);
            }
        }

        return best;
    }

    private const int Quiet = -1;
    private const int Several = -2;

    /// <summary>Who is talking in each frame: one speaker, nobody, or more than one.</summary>
    private static int[] Paint(IReadOnlyList<SpeakerTurn> turns, int frames)
    {
        var painted = Enumerable.Repeat(Quiet, frames).ToArray();

        foreach (var turn in turns)
        {
            var from = Math.Max(0, (int)Math.Round(turn.StartSeconds / Frame));
            var to = Math.Min(frames, (int)Math.Round(turn.EndSeconds / Frame));

            for (var frame = from; frame < to; frame++)
            {
                painted[frame] = painted[frame] == Quiet || painted[frame] == turn.Speaker
                    ? turn.Speaker
                    : Several;
            }
        }

        return painted;
    }

    private static string Said(IReadOnlyList<TranscriptSegment> saved, double start, double end)
    {
        var text = string.Join(
            " ",
            saved
                .Where(s => Math.Min(s.EndSeconds, end) - Math.Max(s.StartSeconds, start) > 0.3)
                .Select(s => s.Text.Trim()));

        return Shorten(text, 150);
    }

    private static double Level(double power, int frames) =>
        frames == 0 ? double.NaN : 10 * Math.Log10(Math.Max(1e-12, power / frames));

    private static string Shorten(string text, int length) =>
        text.Length <= length ? text : text[..length] + "…";

    private static string Clock(double seconds) =>
        $"{(int)(seconds / 60),2}:{seconds % 60:00.0}";

    private static void Heading(string title)
    {
        Console.WriteLine();
        Console.WriteLine(title);
        Console.WriteLine(new string('-', title.Length));
    }
}
