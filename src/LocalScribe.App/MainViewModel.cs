using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text;
using LocalScribe.Core.Archive;
using LocalScribe.Core.Audio;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Models;
using LocalScribe.Core.Pipeline;
using LocalScribe.Core.Refinement;
using LocalScribe.Core.Transcription;
using LocalScribe.Onnx;

namespace LocalScribe.App;

/// <summary>
/// Drives the window. Holds no UI types, so the interesting behaviour can be reasoned about
/// (and later tested) without standing up a WinUI host.
/// </summary>
public sealed class MainViewModel : INotifyPropertyChanged, IDisposable
{
    private readonly string _modelRoot;
    private ExecutionPlan? _plan;
    private DeviceCapabilities? _capabilities;
    private CancellationTokenSource? _cancellation;
    private MicrophoneCapture? _microphone;
    private LiveTranscriptionSession? _liveSession;

    /// <summary>
    /// The live transcriber is kept between recordings rather than rebuilt. Opening the QNN
    /// sessions takes about five seconds, and the session does not own it, so re-creating it per
    /// recording both leaked the old one and paid that cost again every time.
    /// </summary>
    private ITranscriber? _transcriber;
    private string? _transcriberKey;
    private readonly SemaphoreSlim _transcriberLock = new(1, 1);
    private bool _modelReady;
    private SessionDiagnostics? _diagnostics;

    /// <summary>
    /// The segments behind the displayed text. Kept because the transcript is no longer just a
    /// string: paragraphs, export formats and clicking a line to hear it all need the timings.
    /// </summary>
    private IReadOnlyList<TranscriptSegment> _segments = [];

    /// <summary>
    /// Everything the microphone delivered this session, so a finished recording can be played
    /// back and its transcript clicked through. A recording has no file to re-open, so if this
    /// is not kept the audio is gone the moment capture stops.
    /// </summary>
    private List<float>? _liveCapture;

    /// <summary>
    /// The audio behind the current transcript, and the segments as the transcriber returned
    /// them, before any speaker was attached. Kept so diarization can be run again with a
    /// different answer without transcribing anything twice — and run from the original
    /// segments, since attribution splits them and splitting an already-split transcript
    /// compounds whatever the first attempt got wrong.
    /// </summary>
    private PcmAudio? _audio;
    private IReadOnlyList<TranscriptSegment> _segmentsBeforeSpeakers = [];

    private string _status = "Starting up…";
    private string _hardwareSummary = string.Empty;
    private string _transcript = string.Empty;
    private string _provisionalText = string.Empty;
    private string _summary = string.Empty;
    private double _progress;
    private bool _busy;
    private bool _recording;
    private bool _preparing;
    private bool _warming;

    public MainViewModel(string? modelRoot = null)
    {
        _modelRoot = modelRoot ?? Path.Combine(AppContext.BaseDirectory, "models");
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>
    /// True while the model is loading at startup and nothing has asked for it yet.
    /// <para>
    /// Its own state rather than a flavour of <see cref="IsPreparing"/>. Preparing means the user
    /// pressed record and must not speak; warming up means the app is getting itself ready and
    /// the user can do as they please. Both wait for the same load, and telling someone to hold
    /// still when they have not asked for anything is a different and worse message.
    /// </para>
    /// </summary>
    public bool IsWarmingUp
    {
        get => _warming;
        private set => Set(ref _warming, value);
    }

    /// <summary>True once the model is loaded and a recording can start immediately.</summary>
    public bool IsModelReady
    {
        get => _modelReady;
        private set => Set(ref _modelReady, value);
    }

    /// <summary>Plays back whatever was last transcribed, so a line can be clicked and heard.</summary>
    public TranscriptPlayer Player { get; } = new();

    /// <summary>
    /// The loaded audio reduced to peaks for drawing. Computed here because the view model owns
    /// the samples; the window owns only how wide to draw them.
    /// </summary>
    public float[] WaveformPeaks(int buckets) => Player.Peaks(buckets);

    /// <summary>The transcript grouped into paragraphs, which is what the window shows.</summary>
    public IReadOnlyList<TranscriptParagraph> Paragraphs { get; private set; } = [];

    /// <summary>True when there is a transcript to copy, save or discard.</summary>
    public bool HasTranscript => _segments.Count > 0;

    /// <summary>Where the transcript came from, used to name an export.</summary>
    public string SourceName { get; private set; } = "Transcript";

    /// <summary>
    /// Replaces the transcript and everything derived from it. One place, so the paragraphs, the
    /// flat text and the export never disagree about what the transcript currently is.
    /// </summary>
    private void SetTranscript(IReadOnlyList<TranscriptSegment> segments)
    {
        _segments = segments;
        Paragraphs = TranscriptFormatter.Paragraphs(segments);
        Transcript = TranscriptFormatter.ToPlainText(Paragraphs);

        // Counted here rather than where diarization finishes, so that renaming keeps it honest:
        // merging two labels that were the same person really is one speaker fewer, and the
        // badge should say so the moment it happens.
        SpeakerCount = DistinctSpeakers(segments);

        Raise(nameof(Paragraphs));
        Raise(nameof(HasTranscript));
    }

    /// <summary>
    /// Saves the transcript and the recording it came from as one file.
    /// <para>
    /// The only format that keeps what the app is for. Text alone loses the half that makes it
    /// useful — clicking a line to hear it, dragging the waveform and watching the words follow,
    /// correcting a speaker by listening again — because all of that needs the audio and the
    /// timings together. And a recording made in the app existed nowhere but memory until now,
    /// so closing the window destroyed the one thing the user could not recreate.
    /// </para>
    /// </summary>
    public void SaveArchive(string path)
    {
        if (_audio is not { } audio)
        {
            throw new InvalidOperationException("There is no recording to save.");
        }

        TranscriptArchive.Save(path, audio, _segments, SourceName);
    }

    /// <summary>Opens a saved transcript, audio and all.</summary>
    public void OpenArchive(string path)
    {
        var contents = TranscriptArchive.Load(path);

        Discard();

        _audio = contents.Audio;
        _segmentsBeforeSpeakers = contents.Segments;

        SourceName = contents.Manifest.SourceName is { Length: > 0 } name
            ? name
            : Path.GetFileNameWithoutExtension(path);

        SetTranscript(contents.Segments);
        Player.Load(contents.Audio);

        Status = $"Opened {SourceName}. {contents.Segments.Count} segment(s), "
            + $"{TimeSpan.FromSeconds(contents.Audio.DurationSeconds):mm\\:ss} of audio.";
    }

    /// <summary>True when there is something worth saving as an archive.</summary>
    public bool CanSaveArchive => _audio is not null && _segments.Count > 0;

    /// <summary>
    /// When each word of a paragraph was said, so one can be clicked and heard.
    /// <para>
    /// Estimated rather than measured. Whisper times whole segments, and the proper way to get
    /// words — aligning them against the decoder's cross-attention — needs weights these
    /// exported graphs do not emit. What is available is the loudness of the recording, which
    /// shows where the pauses are, so the words are shared out across the speech rather than
    /// across the clock.
    /// </para>
    /// </summary>
    public IReadOnlyList<WordTimings.Word> WordsIn(IReadOnlyList<TranscriptSegment> segments)
    {
        ArgumentNullException.ThrowIfNull(segments);

        if (_audio is not { } audio)
        {
            return [];
        }

        var words = new List<WordTimings.Word>();

        foreach (var segment in segments)
        {
            // Measured where the aligner has been over this segment, estimated where it has not.
            // Read from a table rather than run here: this is called for every paragraph that
            // scrolls into view, and a third of a second of model time per paragraph would make
            // the transcript unusable to scroll.
            words.AddRange(Aligned(segment) ?? WordTimings.For(audio, segment));
        }

        return words;
    }

    /// <summary>
    /// The measured times for a segment, or null when there are none to be had.
    /// <para>
    /// Checked against the text as well as the clock. Cleanup rewrites a segment after alignment
    /// has already run over it, and although it keeps the timings and nearly all the words, a
    /// removed filler would leave the measured list one word short of the displayed one — after
    /// which every word in the paragraph would carry its neighbour's time.
    /// </para>
    /// </summary>
    private IReadOnlyList<WordTimings.Word>? Aligned(TranscriptSegment segment)
    {
        lock (_alignedGate)
        {
            if (!_aligned.TryGetValue(Key(segment), out var found))
            {
                return null;
            }

            return string.Equals(found.Text, segment.Text, StringComparison.Ordinal) ? found.Words : null;
        }
    }

    private static (long Start, long End) Key(TranscriptSegment segment) =>
        ((long)Math.Round(segment.StartSeconds * 1000), (long)Math.Round(segment.EndSeconds * 1000));

    private readonly object _alignedGate = new();

    private readonly Dictionary<(long Start, long End), (string Text, IReadOnlyList<WordTimings.Word> Words)>
        _aligned = [];

    /// <summary>
    /// Measures when every word was said, one segment at a time.
    /// <para>
    /// Runs alongside cleanup and speaker detection rather than after them: it needs the audio
    /// and the words, neither of which the other two change in a way that matters here — cleanup
    /// keeps the timings, and speakers are a label. Anything it cannot align keeps the estimate,
    /// so a failure costs precision on one segment and nothing else.
    /// </para>
    /// </summary>
    private async Task AlignWordsAsync(
        IReadOnlyList<TranscriptSegment> segments,
        PcmAudio audio,
        IProgress<double>? progress,
        CancellationToken cancellationToken)
    {
        if (ForcedAligner.Find(_modelRoot) is not { } directory)
        {
            return;
        }

        try
        {
            await Task.Run(() =>
            {
                using var aligner = ForcedAligner.Load(directory, _plan);

                for (var i = 0; i < segments.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    progress?.Report((i + 1) / (double)segments.Count);

                    if (aligner.Align(audio, segments[i], cancellationToken) is not { } words)
                    {
                        continue;
                    }

                    lock (_alignedGate)
                    {
                        _aligned[Key(segments[i])] = (segments[i].Text, words);
                    }
                }
            }, cancellationToken).ConfigureAwait(true);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception exception)
        {
            // The estimate is still there, so this is a loss of precision rather than of the
            // transcript.
            Status = $"Transcribed, but the words could not be timed exactly: {exception.Message}";
        }
    }

    /// <summary>The transcript in one of the formats the save dialog offers.</summary>
    public string Export(TranscriptFormat format) => format switch
    {
        TranscriptFormat.Markdown => TranscriptFormatter.ToMarkdown(Paragraphs, SourceName),
        TranscriptFormat.SubRip => TranscriptFormatter.ToSubRip(_segments),
        _ => TranscriptFormatter.ToPlainText(Paragraphs),
    };

    /// <summary>
    /// Works out who spoke when, if the speaker models are installed.
    /// <para>
    /// Optional on purpose. Whisper does not separate voices and the models that do are a
    /// separate download, so a machine without them transcribes exactly as before rather than
    /// failing. A failure here costs the labels, not the transcript.
    /// </para>
    /// </summary>
    private async Task<IReadOnlyList<TranscriptSegment>> AttributeSpeakersAsync(
        IReadOnlyList<TranscriptSegment> segments,
        PcmAudio audio,
        int? speakers = null)
    {
        Status = "Working out who spoke…";
        Progress = 0;

        var found = new Progress<double>(fraction =>
        {
            Progress = fraction;
            Status = $"Working out who spoke… {(int)(fraction * 100)}%";
        });

        var turns = await FindTurnsAsync(audio, speakers, found).ConfigureAwait(true);

        return turns is null ? segments : SpeakerDiarizer.Attribute(segments, turns);
    }

    /// <summary>
    /// Works out who spoke when, from the audio alone. Null when the models are missing or the
    /// run failed.
    /// <para>
    /// Separated from attribution because the two have completely different dependencies, and
    /// keeping them together hid that. Finding the turns needs the recording and nothing else —
    /// not the words, not their timings — so it can run at the same time as the cleanup model is
    /// rewriting the text. Attaching those turns to segments is pure bookkeeping and costs
    /// nothing once both have finished.
    /// </para>
    /// </summary>
    private async Task<IReadOnlyList<SpeakerTurn>?> FindTurnsAsync(
        PcmAudio audio,
        int? speakers,
        IProgress<double>? progress)
    {
        var directory = Path.Combine(_modelRoot, "diarization");

        if (!File.Exists(Path.Combine(directory, "segmentation.onnx")))
        {
            return null;
        }

        try
        {
            return await Task.Run(() =>
            {
                using var diarizer = SpeakerDiarizer.Load(directory);

                // Speakers are followed through the audio rather than told apart by their
                // voices, whether or not a count was given. Two local speakers in one window
                // are two different people — the segmentation model reports that directly, and
                // it survives recordings the embedding model cannot read at all.
                //
                // With no count, the same facts put a floor under it: colouring the graph of
                // who talked over whom says how few people the evidence can be explained by.
                // On a phone-quality interview that finds three, where clustering voices found
                // nineteen with one of them holding 85% of the speech.
                //
                // Measured against the old path on everything to hand: identical on the two
                // samples — 52/48 against 51/49, and ten rapid turns alternating cleanly in
                // both — and the difference between a usable transcript and an unusable one on
                // the real recording.
                return diarizer.DiarizeByTracking(
                    audio,
                    maxSpeakers: speakers,
                    exactSpeakers: speakers,
                    progress: progress,
                    cancellationToken: _cancellation?.Token ?? default);
            }).ConfigureAwait(true);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception exception)
        {
            Status = $"Transcribed, but speakers could not be identified: {exception.Message}";
            return null;
        }
    }

    /// <summary>How many distinct people the transcript is currently attributed to.</summary>
    public int SpeakerCount
    {
        get;
        private set
        {
            if (field == value)
            {
                return;
            }

            field = value;
            Raise(nameof(SpeakerCount));
        }
    }

    private static int DistinctSpeakers(IReadOnlyList<TranscriptSegment> segments) =>
        segments
            .Select(segment => segment.Speaker)
            .Where(speaker => !string.IsNullOrWhiteSpace(speaker))
            .Distinct(StringComparer.Ordinal)
            .Count();

    /// <summary>True when there is audio to work out speakers from.</summary>
    public bool CanFindSpeakers => _audio is not null && _segmentsBeforeSpeakers.Count > 0;

    /// <summary>
    /// Works out the speakers again, told how many there are.
    /// <para>
    /// The count is the one thing a person always knows and the algorithm never does. Without it
    /// the number of speakers has to be inferred from a distance threshold, and a threshold that
    /// is slightly wrong for a recording fails in both directions at once — merging two people
    /// who sound alike while splitting one who moved closer to the microphone. Being told there
    /// are three removes the guess entirely.
    /// </para>
    /// </summary>
    /// <param name="speakers">How many people are talking, or null to infer it.</param>
    public async Task FindSpeakersAsync(int? speakers)
    {
        if (_audio is not { } audio || _segmentsBeforeSpeakers.Count == 0)
        {
            return;
        }

        Status = speakers is { } n ? $"Finding {n} speakers…" : "Working out who spoke…";

        IsBusy = true;

        IReadOnlyList<TranscriptSegment> attributed;

        try
        {
            attributed = await AttributeSpeakersAsync(_segmentsBeforeSpeakers, audio, speakers);
        }
        finally
        {
            IsBusy = false;
            Progress = 0;
        }

        SetTranscript(attributed);

        Status = SpeakerCount switch
        {
            0 => "No speakers were found.",
            1 => "Found one speaker.",
            var many => $"Found {many} speakers.",
        };
    }

    /// <summary>
    /// Renames a speaker, either for one paragraph or throughout.
    /// <para>
    /// Both are needed and they are not the same operation. Diarization splits one person into
    /// two often enough that "this part is also Kim" has to be possible, and having named
    /// someone once, naming every other thing they said has to be one action rather than
    /// thirty.
    /// </para>
    /// </summary>
    /// <param name="startSeconds">Start of the paragraph being renamed.</param>
    /// <param name="endSeconds">End of the paragraph being renamed.</param>
    /// <param name="currentName">The label as it stands, used to find the rest of them.</param>
    /// <param name="newName">What to call them.</param>
    /// <param name="everywhere">True to rename every paragraph currently sharing the label.</param>
    public void RenameSpeaker(
        double startSeconds,
        double endSeconds,
        string? currentName,
        string newName,
        bool everywhere)
    {
        if (string.IsNullOrWhiteSpace(newName) || _segments.Count == 0)
        {
            return;
        }

        var name = newName.Trim();

        var updated = _segments.Select(segment =>
        {
            var inScope = everywhere
                ? string.Equals(segment.Speaker, currentName, StringComparison.Ordinal)

                // Otherwise only the segments inside the paragraph that was clicked. Compared on
                // midpoint so a segment straddling a boundary belongs to one paragraph rather
                // than to both or to neither.
                : Midpoint(segment) >= startSeconds && Midpoint(segment) <= endSeconds;

            return inScope ? segment with { Speaker = name } : segment;
        }).ToList();

        SetTranscript(updated);
        Status = everywhere ? $"Renamed to {name} throughout." : $"Renamed to {name} for that part.";
    }

    /// <summary>
    /// Renames one paragraph and every other one that sounds like the same person.
    /// <para>
    /// The answer to a merge, which renaming cannot fix on its own. "Rename everywhere" moves
    /// every paragraph carrying a label, which is right when two labels are one person and
    /// exactly wrong when one label is two: after peeling off the first paragraph by hand, the
    /// remaining label still means both people, so there is nothing left to point at. This
    /// takes the paragraph the user identified as an example and sorts the others against it by
    /// voice.
    /// </para>
    /// <para>
    /// It can decline. If the paragraphs do not divide into two voices, nothing is renamed
    /// beyond the one the user clicked, and the caller is told — a user can be wrong about a
    /// paragraph, and forcing a split on a single voice would scatter a correctly-labelled
    /// speaker across two names.
    /// </para>
    /// </summary>
    /// <returns>How many other paragraphs were moved, or -1 when there was no split to make.</returns>
    public async Task<int> RenameSpeakerByVoiceAsync(
        double startSeconds,
        double endSeconds,
        string? currentName,
        string newName)
    {
        if (string.IsNullOrWhiteSpace(newName) || _segments.Count == 0 || _audio is not { } audio)
        {
            return -1;
        }

        var directory = Path.Combine(_modelRoot, "diarization");
        if (!File.Exists(Path.Combine(directory, "embedding.onnx")))
        {
            return -1;
        }

        var name = newName.Trim();

        // The paragraph the user pointed at, and the others still carrying the old label.
        var example = Paragraphs.FirstOrDefault(paragraph =>
            paragraph.StartSeconds <= startSeconds && paragraph.EndSeconds >= endSeconds);

        if (example is null)
        {
            return -1;
        }

        var others = Paragraphs
            .Where(paragraph => !ReferenceEquals(paragraph, example))
            .Where(paragraph => string.Equals(paragraph.Speaker, currentName, StringComparison.Ordinal))
            .Where(paragraph => paragraph.EndSeconds - paragraph.StartSeconds >= ShortestComparableParagraph)
            .ToList();

        if (others.Count == 0)
        {
            RenameSpeaker(startSeconds, endSeconds, currentName, name, everywhere: false);
            return 0;
        }

        IsBusy = true;
        Status = "Listening to the other parts…";
        Progress = 0;

        SpeakerSplit.Result split;

        try
        {
            split = await Task.Run(() =>
            {
                using var diarizer = SpeakerDiarizer.Load(directory);

                if (diarizer.EmbedSpan(audio, example.StartSeconds, example.EndSeconds)
                    is not { } exampleVoice)
                {
                    return new SpeakerSplit.Result([], 0, false);
                }

                var voices = new List<float[]>(others.Count);
                var kept = new List<int>(others.Count);

                for (var i = 0; i < others.Count; i++)
                {
                    Progress = (i + 1) / (double)others.Count;

                    if (diarizer.EmbedSpan(audio, others[i].StartSeconds, others[i].EndSeconds)
                        is { } voice)
                    {
                        voices.Add(voice);
                        kept.Add(i);
                    }
                }

                var found = SpeakerSplit.ByExample(exampleVoice, voices);

                // Indexes come back against the embeddings that succeeded, which is a shorter
                // list than the paragraphs whenever one was too quiet to measure.
                return found with
                {
                    JoinsExample = found.JoinsExample.Select(index => kept[index]).ToList(),
                };
            }).ConfigureAwait(true);
        }
        catch (Exception exception)
        {
            Status = $"Could not compare the voices: {exception.Message}";
            return -1;
        }
        finally
        {
            IsBusy = false;
            Progress = 0;
        }

        if (!split.Split)
        {
            RenameSpeaker(startSeconds, endSeconds, currentName, name, everywhere: false);
            Status = $"Renamed that part to {name}. The other parts sound like the same voice, "
                + "so they were left alone.";
            return -1;
        }

        // Applied in one pass. Paragraphs are grouped by speaker, so renaming them one at a time
        // would regroup the transcript underneath the ranges still waiting to be renamed.
        var moving = split.JoinsExample.Select(index => others[index]).Append(example).ToList();

        var updated = _segments.Select(segment =>
        {
            var midpoint = Midpoint(segment);

            var inScope = moving.Any(paragraph =>
                midpoint >= paragraph.StartSeconds && midpoint <= paragraph.EndSeconds);

            return inScope ? segment with { Speaker = name } : segment;
        }).ToList();

        SetTranscript(updated);

        var moved = split.JoinsExample.Count;
        Status = moved == 0
            ? $"Renamed that part to {name}. No other part sounded like them."
            : $"Renamed that part and {moved} other{(moved == 1 ? "" : "s")} to {name}.";

        return moved;
    }

    /// <summary>
    /// Below this a paragraph identifies a voice too poorly to sort, so it keeps its label
    /// rather than being assigned on a guess.
    /// </summary>
    private const double ShortestComparableParagraph = 1.0;

    /// <summary>
    /// Everything that happens once the words exist: cleanup and speaker detection, run side by
    /// side, then combined.
    /// <para>
    /// Shared by both routes in. A recording used to stop here with neither — the live session
    /// committed its segments and that was the end of it, so a transcript made in the app was
    /// never punctuated at all, whatever cleanup model was running. That was the same complaint
    /// the file path had, in the one path that had never been wired up.
    /// </para>
    /// <para>
    /// The two stages share no data in either direction: the cleanup model rewrites text and
    /// never looks at the audio, the diarizer reads the audio and never looks at the words.
    /// Running them in sequence cost the shorter of the two for nothing.
    /// </para>
    /// </summary>
    /// <param name="spoken">The words as transcribed, before cleanup.</param>
    /// <param name="audio">The recording, or null when there is none to listen to again.</param>
    private async Task FinishTranscriptAsync(
        IReadOnlyList<TranscriptSegment> spoken,
        PcmAudio? audio,
        CancellationToken cancellationToken)
    {
        var refiner = BuildRefiner();

        if (refiner is null && audio is null)
        {
            _segmentsBeforeSpeakers = spoken;
            SetTranscript(spoken);
            return;
        }

        var stages = new SharedBar(this);
        var transcript = new Transcript(spoken);

        var cleaning = refiner is null
            ? Task.FromResult<RefinementResult?>(null)
            : CleanAsync(refiner, transcript, stages, cancellationToken);

        var listening = audio is null
            ? Task.FromResult<IReadOnlyList<SpeakerTurn>?>(null)
            : FindTurnsAsync(audio, speakers: null, new Progress<double>(stages.Speakers));

        // A third stage beside the other two. It reads the audio and the words and writes
        // neither, so it collides with nothing.
        var timing = audio is null
            ? Task.CompletedTask
            : AlignWordsAsync(spoken, audio, new Progress<double>(stages.Words), cancellationToken);

        await Task.WhenAll(cleaning, listening, timing);

        var refinement = await cleaning;
        var turns = await listening;

        // The cleaned segments when cleanup ran, the raw ones when it did not. This is the line
        // that was missing for a long time: the cleaned text used to be computed and dropped,
        // and what the user read was always the raw transcript.
        var cleaned = refinement?.CleanedSegments ?? spoken;

        _segmentsBeforeSpeakers = cleaned;

        // Attribution last, on the cleaned text rather than the raw. Segments spanning a speaker
        // change are divided at sentence boundaries, and by this point there are real sentence
        // boundaries to divide at.
        SetTranscript(turns is null ? cleaned : SpeakerDiarizer.Attribute(cleaned, turns));

    }

    private async Task<RefinementResult?> CleanAsync(
        TranscriptRefiner refiner,
        Transcript transcript,
        SharedBar stages,
        CancellationToken cancellationToken) =>
        await refiner.RefineAsync(
            transcript,
            Glossary,
            RefinementOutputs.Default,
            new Progress<double>(stages.Cleanup),

            // Shown as it lands, window by window, so the transcript is visibly gaining
            // punctuation rather than sitting there until the last one returns. Safe to write
            // over the top of: speaker labels are attached after both stages finish, so there is
            // nothing here yet for a redraw to lose.
            new Progress<IReadOnlyList<TranscriptSegment>>(SetTranscript),
            cancellationToken).ConfigureAwait(true);

    /// <summary>
    /// One progress bar shared by two stages running at once.
    /// <para>
    /// Reported as the mean of the two, which is a fair account of the wait: they start together
    /// and the run is over when the slower finishes. Showing whichever happened to report last
    /// would make the bar jump backwards, and showing only one would leave it parked while the
    /// other was still working.
    /// </para>
    /// <para>
    /// A stage that is not running contributes nothing and is not counted, so a machine with no
    /// cleanup model still gets a bar that means something.
    /// </para>
    /// </summary>
    private sealed class SharedBar(MainViewModel owner)
    {
        private readonly object _gate = new();

        private double _cleanup;
        private double _speakers;
        private double _words;
        private bool _cleaningUp;
        private bool _findingSpeakers;
        private bool _timingWords;

        public void Cleanup(double fraction)
        {
            lock (_gate)
            {
                _cleaningUp = true;
                _cleanup = fraction;
            }

            Publish();
        }

        public void Speakers(double fraction)
        {
            lock (_gate)
            {
                _findingSpeakers = true;
                _speakers = fraction;
            }

            Publish();
        }

        public void Words(double fraction)
        {
            lock (_gate)
            {
                _timingWords = true;
                _words = fraction;
            }

            Publish();
        }

        private void Publish()
        {
            double fraction;
            string what;

            lock (_gate)
            {
                var running = (_cleaningUp ? 1 : 0) + (_findingSpeakers ? 1 : 0) + (_timingWords ? 1 : 0);
                if (running == 0)
                {
                    return;
                }

                fraction = ((_cleaningUp ? _cleanup : 0)
                    + (_findingSpeakers ? _speakers : 0)
                    + (_timingWords ? _words : 0)) / running;

                what = (_cleaningUp, _findingSpeakers, _timingWords) switch
                {
                    (true, true, true) => "Cleaning up, working out who spoke, and timing the words",
                    (true, true, false) => "Cleaning up and working out who spoke",
                    (true, false, true) => "Cleaning up and timing the words",
                    (false, true, true) => "Working out who spoke and timing the words",
                    (true, false, false) => "Cleaning up the transcript",
                    (false, true, false) => "Working out who spoke",
                    _ => "Timing the words",
                };
            }

            owner.Progress = fraction;
            owner.Status = $"{what}… {(int)(fraction * 100)}%";
        }
    }

    private static double Midpoint(TranscriptSegment segment) =>
        segment.StartSeconds + ((segment.EndSeconds - segment.StartSeconds) / 2);

    /// <summary>Throws the transcript away and releases the audio behind it.</summary>
    public void Discard()
    {
        Player.Clear();

        _audio = null;
        _segmentsBeforeSpeakers = [];

        lock (_alignedGate)
        {
            _aligned.Clear();
        }

        SetTranscript([]);
        ProvisionalText = string.Empty;
        Progress = 0;
        SourceName = "Transcript";
        Status = "Discarded.";
    }

    /// <summary>What the app is doing right now, shown in the status bar.</summary>
    public string Status
    {
        get => _status;
        private set => Set(ref _status, value);
    }

    /// <summary>Where the models are running, so the user can see the NPU is actually in use.</summary>
    public string HardwareSummary
    {
        get => _hardwareSummary;
        private set => Set(ref _hardwareSummary, value);
    }

    /// <summary>Committed transcript text.</summary>
    public string Transcript
    {
        get => _transcript;
        private set => Set(ref _transcript, value);
    }

    /// <summary>
    /// Words the model may still revise. Shown in a lighter style so the user can tell settled
    /// text from a guess that is about to change.
    /// </summary>
    public string ProvisionalText
    {
        get => _provisionalText;
        private set => Set(ref _provisionalText, value);
    }

    public double Progress
    {
        get => _progress;
        private set => Set(ref _progress, value);
    }

    public bool IsBusy
    {
        get => _busy;
        private set => Set(ref _busy, value);
    }

    public bool IsRecording
    {
        get => _recording;
        private set => Set(ref _recording, value);
    }

    /// <summary>
    /// True while the model is loading and the microphone is not yet running.
    /// <para>
    /// This is its own state rather than a flavour of <see cref="IsBusy"/> because it is the one
    /// moment the user must not speak. Loading the QNN sessions takes seconds, and anything said
    /// before the microphone starts is gone — not delayed, gone.
    /// </para>
    /// </summary>
    public bool IsPreparing
    {
        get => _preparing;
        private set => Set(ref _preparing, value);
    }

    /// <summary>
    /// What is doing the cleanup, or null when nothing is. The glossary, the summary and the
    /// punctuation repair all depend on this, and all three silently do nothing without it —
    /// which is worth saying where the user is looking rather than leaving them to notice.
    /// </summary>
    public string? CleanupModel => _languageModel?.Description;

    /// <summary>Terms the cleanup model should spell correctly. Edited by the user in the UI.</summary>
    public List<string> Glossary { get; } = [];

    /// <summary>
    /// Probes the machine and works out the plan. Runs once at startup; the result is shown in
    /// the UI rather than hidden, because "is it using the NPU?" is the question everyone asks.
    /// </summary>
    public async Task InitialiseAsync()
    {
        Status = "Checking hardware…";

        // Both at once. Finding a cleanup backend can mean starting a process and waiting on it,
        // and that has no business being added to the delay before the microphone works.
        var probing = Task.Run(() => DeviceProbe.Probe(_modelRoot));
        var resolving = LocalLanguageModel.ResolveAsync();

        var capabilities = await probing;
        _languageModel = await resolving;

        capabilities = capabilities with { LocalLanguageModelPresent = _languageModel is not null };

        _capabilities = capabilities;
        _plan = AcceleratorPlanner.Plan(capabilities);

        // Where the work will run, but not which weights: that is not known until they are
        // opened, and naming the size the planner asked for is how the window spent a session
        // claiming medium.en while running large-v3-turbo.
        // Naming the cleanup backend, not just its device. Which model is doing the punctuation
        // decides how good the punctuation is, and this is the line people read when the answer
        // is "not very".
        var cleanup = _languageModel is { } model
            ? model.Description
            : "no cleanup model found";

        HardwareSummary = $"encoder on {_plan.Encoder.Device}, decoder on {_plan.Decoder.Device}, "
            + $"cleanup: {cleanup}";
        Status = _plan.Warnings.Count == 0
            ? "Ready."
            : $"Ready, with {_plan.Warnings.Count} warning(s). Run localscribe-doctor for detail.";
    }

    /// <summary>Transcribes a file from disk.</summary>
    public async Task TranscribeFileAsync(string path)
    {
        if (_plan is null)
        {
            Status = "Still checking hardware.";
            return;
        }

        IsBusy = true;
        Progress = 0;
        SetTranscript([]);
        _cancellation = new CancellationTokenSource();

        try
        {
            Status = AudioFileLoader.IsVideo(path) ? "Extracting audio…" : "Loading audio…";
            var audio = await Task.Run(() => AudioFileLoader.Load(path), _cancellation.Token);

            SourceName = Path.GetFileNameWithoutExtension(path);
            _audio = audio;

            // Held so a line in the transcript can be clicked and heard. The timings refer to
            // these samples, not to the file, which is why the decoded audio is what is kept.
            Player.Load(audio);

            Status = IsModelReady ? "Preparing…" : "Loading the model…";
            var transcriber = await Task.Run(() => TranscriberFor(_plan), _cancellation.Token);
            var pipeline = new TranscriptionPipeline(transcriber, BuildRefiner());

            Status = $"Transcribing with {transcriber.Description}…";

            // Each window's text as it lands, rather than a bare percentage. A long file is
            // otherwise several minutes of a progress bar and nothing to read.
            var streamed = new List<TranscriptSegment>();

            var progress = new Progress<TranscriptionProgress>(update =>
            {
                Progress = update.Fraction;

                Status = $"Transcribing… {update.ChunksCompleted} of {update.ChunksTotal} windows";

                if (update.Phase == TranscriptionPhase.Transcribing && update.LatestText.Length > 0)
                {
                    streamed.Add(new TranscriptSegment(
                        update.LatestText,
                        update.ChunksCompleted * AudioChunker.WindowSeconds,
                        (update.ChunksCompleted + 1) * AudioChunker.WindowSeconds));

                    SetTranscript(streamed.ToList());
                }
            });

            var transcript = await pipeline.TranscribeAsync(audio, progress, _cancellation.Token);

            await FinishTranscriptAsync(transcript.Segments, audio, _cancellation.Token);

            Status = "Done.";
        }
        catch (OperationCanceledException)
        {
            Status = "Cancelled.";
        }
        catch (Exception exception)
        {
            // Surfacing the message beats a silent failure: most problems here are a missing
            // model file or an export whose signature does not match, and both say so plainly.
            Status = $"Failed: {exception.Message}";
        }
        finally
        {
            IsBusy = false;
            _cancellation?.Dispose();
            _cancellation = null;
        }
    }

    /// <summary>
    /// Starts live transcription from the microphone.
    /// <para>
    /// The model is opened before the microphone, off the UI thread, and the user is told to
    /// wait until it is. This used to run the load synchronously on the click, which froze the
    /// window for about five seconds and started capturing only afterwards — so the opening
    /// words of every recording were lost, with nothing on screen to suggest they would be.
    /// </para>
    /// </summary>
    public async Task StartRecordingAsync()
    {
        if (_plan is null || IsRecording || IsPreparing)
        {
            return;
        }

        // Live work wants a smaller model than batch, so the plan is recomputed for this mode.
        var livePlan = AcceleratorPlanner.Plan(
            _capabilities ?? DeviceCapabilities.Unknown,
            PerformanceProfile.Considerate,
            WorkloadMode.Live);

        IsPreparing = true;
        Status = "Loading the model — wait for the go-ahead before speaking…";

        ITranscriber transcriber;

        try
        {
            transcriber = await Task.Run(() => TranscriberFor(livePlan));
        }
        catch (Exception exception)
        {
            IsPreparing = false;
            Status = $"Could not start listening: {exception.Message}";
            return;
        }

        _liveSession = new LiveTranscriptionSession(transcriber);
        _cancellation = new CancellationTokenSource();
        _diagnostics = SessionDiagnostics.StartIfEnabled(livePlan.Summary, transcriber.Description);
        _liveCapture = [];

        // The previous recording goes now, not when this one produces its own. Everything that
        // reads _audio — playback, the speakers button, and now the diarizer that runs when this
        // recording stops — would otherwise be working from the last file opened, and a
        // recording that captures nothing would have its speakers taken from that instead.
        _audio = null;
        _segmentsBeforeSpeakers = [];

        lock (_alignedGate)
        {
            _aligned.Clear();
        }

        Player.Clear();
        SourceName = $"Recording {DateTime.Now:yyyy-MM-dd HH.mm}";

        _microphone = new MicrophoneCapture();
        _microphone.SamplesAvailable += OnSamplesAvailable;
        _microphone.Start();

        // Only now is anything the user says actually being captured, so this is the first
        // moment it is honest to say so.
        IsPreparing = false;
        IsRecording = true;
        Status = $"Listening — go ahead. ({transcriber.Description})";
    }

    /// <summary>
    /// The cached live transcriber, rebuilt only when the placement or model size changes.
    /// Called on a worker thread; opening ONNX sessions is slow and blocking.
    /// </summary>
    /// <summary>
    /// The loaded transcriber for a plan, opening it only if what is already loaded will not do.
    /// <para>
    /// One slot rather than a cache. Holding a second model would double the memory a QNN
    /// context binary occupies, which is over a gigabyte, to save a load that only happens when
    /// the user switches between recording and transcribing a file.
    /// </para>
    /// <para>
    /// Keyed on the resolved model directory rather than the plan's requested size. The plan
    /// asks for small.en when listening and medium.en for a file, and both may well resolve to
    /// the same weights on disk — reloading a gigabyte because a string differed would be a poor
    /// reason.
    /// </para>
    /// </summary>
    private ITranscriber TranscriberFor(ExecutionPlan plan)
    {
        var directory = ModelDirectoryFor(plan);
        var key = $"{directory}|{plan.Encoder.Device}|{plan.Decoder.Device}|"
            + $"{plan.CpuBudget.IntraOpThreads}|{plan.StrictProviderCheck}";

        // Two callers can arrive at once — a preload running while the user clicks record — and
        // opening the same model twice wastes both the time and the memory.
        _transcriberLock.Wait();

        try
        {
            if (_transcriber is not null && _transcriberKey == key)
            {
                return _transcriber;
            }

            _transcriber?.Dispose();
            _transcriber = null;
            _transcriberKey = null;

            var opened = WhisperOnnxTranscriber.Load(directory, plan);

            _transcriber = opened;
            _transcriberKey = key;

            // Said here rather than only after a preload, so the line is right however the model
            // came to be loaded.
            HardwareSummary = opened.Description;

            return opened;
        }
        finally
        {
            _transcriberLock.Release();
        }
    }

    /// <summary>
    /// Opens the model this machine will use before anything asks for it.
    /// <para>
    /// A QNN context binary takes several seconds to load, and doing it on the click meant the
    /// first recording of a session began with a wait, and the first words after it were spoken
    /// into a microphone that was not running yet. Startup has that time going spare.
    /// </para>
    /// <para>
    /// The cost is memory: the model stays resident from launch rather than from first use. Set
    /// LOCALSCRIBE_NO_PRELOAD to trade the seconds back.
    /// </para>
    /// </summary>
    public async Task PreloadAsync()
    {
        if (_plan is null || Environment.GetEnvironmentVariable("LOCALSCRIBE_NO_PRELOAD") is { Length: > 0 })
        {
            return;
        }

        // The listening plan, because that is the path where waiting costs words rather than
        // patience. A file transcription reuses it when the weights resolve the same.
        var livePlan = AcceleratorPlanner.Plan(
            _capabilities ?? DeviceCapabilities.Unknown,
            PerformanceProfile.Considerate,
            WorkloadMode.Live);

        IsWarmingUp = true;
        Status = "Warming up…";

        try
        {
            var transcriber = await Task.Run(() => TranscriberFor(livePlan));

            IsModelReady = true;

            // The hardware line now names what was loaded rather than what was asked for. The
            // two differ whenever the installed weights are not the size the planner chose, and
            // reporting the request as though it were the result is how the window ended up
            // claiming medium.en while running large-v3-turbo.
            Status = "Ready.";
        }
        catch (Exception exception)
        {
            // Not fatal. The model will be opened on demand, slowly, and the failure will be
            // reported then with the context of what the user was trying to do.
            Status = $"Ready, but the model could not be preloaded: {exception.Message}";
        }
        finally
        {
            IsWarmingUp = false;
        }
    }

    /// <summary>
    /// Where this plan's weights live.
    /// <para>
    /// Keyed on the planned device, not the chip: a Snapdragon that ended up on the CPU needs
    /// the portable export, not the chipset binaries it does not have. And located rather than
    /// resolved, because the plan names a Whisper size while what is installed is whatever the
    /// user obtained — NPU builds arrive under their own names, and demanding an exact match
    /// hides a working model and sends the work to the CPU.
    /// </para>
    /// </summary>
    private string ModelDirectoryFor(ExecutionPlan plan)
    {
        var family = _capabilities?.Family ?? SocFamily.Unknown;

        return ModelLayout.Locate(_modelRoot, family, plan.Encoder.Device, plan.WhisperModel)
            ?? throw new DirectoryNotFoundException(
                $"No Whisper weights under {Path.Combine(_modelRoot, ModelLayout.FolderFor(family, plan.Encoder.Device))}. "
                + "Run 'localscribe-doctor --fetch-models' to download a portable set.");
    }

    /// <summary>Stops listening and commits the trailing words.</summary>
    public async Task StopRecordingAsync()
    {
        if (!IsRecording || _liveSession is null)
        {
            return;
        }

        IsRecording = false;

        // Unsubscribe before stopping. StopRecording does not promise that no further buffers
        // arrive — ones already captured still raise the event — and a handler that is mid-await
        // is running regardless.
        _microphone!.SamplesAvailable -= OnSamplesAvailable;
        _microphone.Stop();
        _microphone.Dispose();
        _microphone = null;

        var session = _liveSession;

        // Cleared first so any callback still in flight sees no session and returns rather than
        // pushing audio into one that is being torn down.
        _liveSession = null;

        try
        {
            var committed = await session.FinishAsync();

            // Shown straight away, unpunctuated. Cleanup takes a while and the words the user
            // just spoke should not wait behind it.
            SetTranscript(committed);
            ProvisionalText = string.Empty;

            // The recording becomes playable the moment it stops, on the same path a
            // transcribed file takes.
            if (_liveCapture is { Count: > 0 } captured)
            {
                _audio = new PcmAudio([.. captured]);
                _segmentsBeforeSpeakers = committed;

                Player.Load(_audio);
            }

            _liveCapture = null;

            // Diagnostics record what the model emitted, so they are written before anything
            // rewrites it.
            if (_diagnostics is not null)
            {
                _diagnostics.Finished(committed);
                Status = $"Stopped. Diagnostics written to {_diagnostics.Directory}";
                _diagnostics = null;
            }
            else
            {
                Status = "Stopped.";
            }

            // The same finishing a transcribed file gets. Without this a recording made in the
            // app was never cleaned up and never had its speakers worked out.
            IsBusy = true;

            try
            {
                await FinishTranscriptAsync(committed, _audio, _cancellation?.Token ?? default);
                Status = "Done.";
            }
            finally
            {
                IsBusy = false;
                Progress = 0;
            }
        }
        catch (OperationCanceledException)
        {
            Status = "Stopped.";
        }
        finally
        {
            await session.DisposeAsync();

            _cancellation?.Dispose();
            _cancellation = null;
        }
    }

    private async void OnSamplesAvailable(float[] samples)
    {
        var session = _liveSession;
        if (session is null)
        {
            return;
        }

        try
        {
            _diagnostics?.Captured(samples);
            _liveCapture?.AddRange(samples);

            var update = await session.PushAsync(samples, _cancellation?.Token ?? default);
            if (update is not null)
            {
                ProvisionalText = update.Text;
                SetTranscript(session.CommittedSegments);

                _diagnostics?.Pass(update.Text, session.CommittedSegments);
            }
        }
        catch (OperationCanceledException)
        {
            // Expected when the user stops recording mid-pass.
        }
        catch (Exception exception)
        {
            Status = $"Live transcription failed: {exception.Message}";
        }
    }

    /// <summary>
    /// The backend found at startup, reused rather than reconnected per pass. Null when none is
    /// running, which disables the cleanup stage without disabling transcription.
    /// </summary>
    private ILanguageModel? _languageModel;

    private TranscriptRefiner? BuildRefiner() =>
        _languageModel is null ? null : new TranscriptRefiner(_languageModel);

    /// <summary>Cancels whatever is running.</summary>
    public void Cancel() => _cancellation?.Cancel();

    /// <summary>
    /// Releases the cached transcriber and anything still capturing. The transcriber holds ONNX
    /// sessions and, on the NPU path, a context binary well over a gigabyte, so it is worth
    /// letting go of rather than leaving to process exit.
    /// </summary>
    public void Dispose()
    {
        _microphone?.Dispose();
        _microphone = null;

        _transcriber?.Dispose();
        _transcriber = null;
        _transcriberKey = null;
        _transcriberLock.Dispose();

        _cancellation?.Dispose();
        _cancellation = null;

        (_languageModel as IDisposable)?.Dispose();
        _languageModel = null;
    }

    private void Raise(string name) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));

    private void Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return;
        }

        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
