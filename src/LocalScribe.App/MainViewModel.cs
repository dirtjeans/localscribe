using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text;
using LocalScribe.Core.Archive;
using LocalScribe.Core.Alignment;
using LocalScribe.Core.Audio;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Models;
using LocalScribe.Core.Pipeline;
using LocalScribe.Core.Provisioning;
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

    public MainViewModel(
        string? modelRoot = null,
        Func<ExecutionPlan, string?, ITranscriber>? openTranscriber = null)
    {
        _modelRoot = modelRoot ?? Path.Combine(AppContext.BaseDirectory, "models");
        _openTranscriber = openTranscriber;
    }

    /// <summary>
    /// How a transcriber is opened for a plan, given the resolved ONNX model directory (null
    /// when none is installed). Injected so the macOS app can choose whisper.cpp; left null,
    /// the ONNX engine loads exactly as it always has. This is the one seam the view model
    /// offers a platform, and it is a constructor argument rather than a virtual because
    /// everything else in here is deliberately platform-blind.
    /// </summary>
    private readonly Func<ExecutionPlan, string?, ITranscriber>? _openTranscriber;

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
        // Anything that changes the transcript makes the copy on disk stale — a fresh
        // transcription, a renamed speaker, a rerun cleanup all arrive here. Opening and
        // saving mark it clean again themselves.
        _savedToDisk = false;

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
        _savedToDisk = true;
    }

    /// <summary>Opens a saved transcript, audio and all.</summary>
    public void OpenArchive(string path)
    {
        // Whatever was waiting for the model is not wanted now. A saved transcript needs no model
        // at all, so it opens straight away — and having it replaced moments later by a file the
        // reader had given up on would be its own kind of wrong.
        _waitingFor = null;

        var contents = TranscriptArchive.Load(path);

        Discard();

        _audio = contents.Audio;
        _segmentsBeforeSpeakers = contents.Segments;

        SourceName = contents.Manifest.SourceName is { Length: > 0 } name
            ? name
            : Path.GetFileNameWithoutExtension(path);

        SetTranscript(contents.Segments);
        RecordSpans(contents.Segments);
        Player.Load(contents.Audio);

        Status = $"Opened {SourceName}. {contents.Segments.Count} segment(s), "
            + $"{TimeSpan.FromSeconds(contents.Audio.DurationSeconds):mm\\:ss} of audio.";

        // What was just loaded is exactly what is on disk. Editing it makes it unsaved again.
        _savedToDisk = true;
    }

    /// <summary>True when there is something worth saving as an archive.</summary>
    public bool CanSaveArchive => _audio is not null && _segments.Count > 0;

    /// <summary>
    /// True when closing the window would lose something. A transcript that was opened from a
    /// file and left alone is already safe; one that was transcribed, recorded, or edited here
    /// exists nowhere else until it is saved.
    /// </summary>
    public bool HasUnsavedWork => CanSaveArchive && !_savedToDisk;

    /// <summary>Whether the transcript as it stands matches a file on disk.</summary>
    private bool _savedToDisk;

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
    /// Found by when the words were said rather than by which segment they came from, because by
    /// the time a segment reaches the reader it is often not the segment that was aligned.
    /// Attaching speakers divides any segment that spans a turn, so thirty transcribed segments
    /// became thirty-eight displayed ones — and the ten that changed were the long early ones,
    /// exactly where an estimate is at its worst. Keyed by identity they all missed, and the
    /// transcript quietly followed the estimate instead.
    /// </para>
    /// <para>
    /// The words themselves come from the segment, not from the alignment. Cleanup rewrites
    /// punctuation after alignment has run, and it is the cleaned text that should be read; only
    /// the times are borrowed. If the two disagree about how many words there are — a filler
    /// removed, a contraction split — they cannot be paired at all and the estimate stands.
    /// </para>
    /// </summary>
    private IReadOnlyList<WordTimings.Word>? Aligned(TranscriptSegment segment)
    {
        var own = Split(segment.Text);
        if (own.Count == 0)
        {
            return null;
        }

        lock (_alignedGate)
        {
            return _alignedFor.TryGetValue(segment, out var measured)
                ? MeasuredWords.Pair(measured, own)
                : null;
        }
    }

    /// <summary>
    /// Writes what the marker will actually follow, one line per segment. The aligner's times
    /// have been verified against the audio and the playback clock against a stopwatch, and the
    /// highlight still lags — so the link left unproven is whether the words on screen carry the
    /// measured times at all. Where the table lookup misses, the marker follows a loudness
    /// estimate instead, and nothing anywhere reported which of the two it was doing.
    /// </summary>
    private void RecordSpans(IReadOnlyList<TranscriptSegment> segments)
    {
        if (_audio is not { } audio)
        {
            return;
        }

        try
        {
            var text = new StringBuilder();
            var measured = 0;

            foreach (var segment in segments)
            {
                var aligned = Aligned(segment);
                var words = aligned ?? WordTimings.For(audio, segment);

                if (aligned is not null)
                {
                    measured++;
                }

                var span = words.Count > 0
                    ? FormattableString.Invariant(
                        $"words {words[0].StartSeconds,7:F2}-{words[^1].EndSeconds,-7:F2}")
                    : "words none           ";

                var head = segment.Text.Length <= 40 ? segment.Text : segment.Text[..37] + "…";

                text.AppendLine(FormattableString.Invariant(
                    $"{(aligned is not null ? "measured" : "ESTIMATE")} {segment.StartSeconds,7:F2}-{segment.EndSeconds,-7:F2} {span} \"{head}\""));
            }

            var header = FormattableString.Invariant(
                $"Displayed word times — {SourceName}{Environment.NewLine}measured {measured} of {segments.Count} segments{Environment.NewLine}{Environment.NewLine}");

            File.WriteAllText(
                Path.Combine(Path.GetTempPath(), "localscribe-spans.txt"),
                header + text);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            // A diagnostic that cannot be written is not worth failing the transcript over.
        }
    }

    /// <summary>Whether this segment's words carry measured times, for drawing the boundary.</summary>
    public bool IsTimed(TranscriptSegment segment)
    {
        lock (_alignedGate)
        {
            return _alignedFor.ContainsKey(segment);
        }
    }

    /// <summary>True once any segment on screen carries measured word times.</summary>
    public bool HasMeasuredWords
    {
        get
        {
            lock (_alignedGate)
            {
                return _alignedFor.Count > 0;
            }
        }
    }

    /// <summary>The words of a segment, with where each begins in its text.</summary>
    private static List<WordTimings.Word> Split(string text)
    {
        var words = new List<WordTimings.Word>();
        var at = 0;

        while (at < text.Length)
        {
            while (at < text.Length && char.IsWhiteSpace(text[at]))
            {
                at++;
            }

            if (at >= text.Length)
            {
                break;
            }

            var from = at;
            while (at < text.Length && !char.IsWhiteSpace(text[at]))
            {
                at++;
            }

            words.Add(new WordTimings.Word(text[from..at], 0, 0) { Offset = from });
        }

        return words;
    }

    private readonly object _alignedGate = new();

    /// <summary>What the aligner measured, kept with the segment it was measured for.</summary>
    private readonly Dictionary<TranscriptSegment, IReadOnlyList<WordTimings.Word>> _alignedFor = [];

    /// <summary>
    /// Reads the whole recording through the alignment model, keeping what it made of it.
    /// <para>
    /// This is nearly all the cost of timing words and it needs only the audio, so it runs beside
    /// cleanup and speaker detection. What it cannot do this early is decide which words go where:
    /// cleanup rewrites the text and attaching a speaker divides segments, so the transcript it
    /// would be aligning against does not exist yet. Only the scores are taken now.
    /// </para>
    /// <para>
    /// The model stays loaded until the words are placed, and is let go immediately afterwards —
    /// it is six hundred megabytes and there is nothing to keep it resident for.
    /// </para>
    /// </summary>
    /// <summary>The growing scan grid; rows before <see cref="_scanFrontierSeconds"/> are final.</summary>
    private AlignmentScores? _scanPartial;

    private double _scanFrontierSeconds;

    private readonly object _frontierGate = new();

    /// <summary>The raw transcript as it streams in, for the progressive pass to time.</summary>
    private IReadOnlyList<TranscriptSegment> _rawSoFar = [];

    /// <summary>
    /// The timed head as last published, and how many raw segments it replaced. Counts, not
    /// clocks: timed bounds come from words and are not strictly monotone, and composing the
    /// view with a time comparison once cut the head short at the first out-of-order end —
    /// stretches that were clickable went grey. Order is law here as everywhere else.
    /// </summary>
    private IReadOnlyList<TranscriptSegment> _timedHead = [];

    private int _timedHeadCovers;

    /// <summary>
    /// How far into the recording clicks already work, or zero when they do not yet. The status
    /// line reads this, and the window uses per-segment timing to draw the same boundary.
    /// </summary>
    public double UsableThroughSeconds { get; private set; }

    /// <summary>
    /// The streamed windows carry honest coarse anchors — window k genuinely covers its thirty
    /// seconds — but the scan's frontier is the hard edge. Text is only timed once its stamps
    /// end this far short of it, so speech spilling a little past its window still has its
    /// frames scanned.
    /// </summary>
    private const double ProgressiveMarginSeconds = 45;

    /// <summary>
    /// Times the head of the transcript while the scan is still working on the tail.
    /// <para>
    /// The scan fills its grid front to back and the streamed windows arrive front to back, so
    /// every stride of scan progress makes another stretch of transcript alignable. Each pass
    /// re-times the whole eligible prefix — a pass is a sub-second Viterbi walk, so re-running
    /// beats bookkeeping — and publishes a transcript whose head carries measured times while
    /// its tail is still raw. The full preview replaces all of this the moment the scan ends.
    /// </para>
    /// </summary>
    private async Task RunProgressiveTimingAsync(CancellationToken cancellationToken)
    {
        var timedThrough = 0.0;
        var frontierUsed = 0.0;

        try
        {
            while (!cancellationToken.IsCancellationRequested && _scores is null)
            {
                await Task.Delay(TimeSpan.FromSeconds(8), cancellationToken).ConfigureAwait(true);

                AlignmentScores? grid;
                double frontier;

                lock (_frontierGate)
                {
                    grid = _scanPartial;
                    frontier = _scanFrontierSeconds;
                }

                if (_aligner is not { } aligner || grid is null || frontier < frontierUsed + 30)
                {
                    continue;
                }

                var raws = _rawSoFar;
                var eligible = raws
                    .TakeWhile(segment => segment.EndSeconds <= frontier - ProgressiveMarginSeconds)
                    .ToList();

                if (eligible.Count == 0 || eligible[^1].EndSeconds <= timedThrough)
                {
                    frontierUsed = frontier;
                    continue;
                }

                var prefix = grid.Prefix(grid.FrameAt(frontier));

                var placed = await Task.Run(
                    () => aligner.AlignAll(prefix, eligible, cancellationToken), cancellationToken)
                    .ConfigureAwait(true);

                cancellationToken.ThrowIfCancellationRequested();

                if (_scores is not null)
                {
                    // The scan finished while this pass ran; the full preview owns it now.
                    break;
                }

                if (placed.All(words => words is null))
                {
                    // A refused pass places nothing. Keeping the previous head is strictly
                    // better than wiping it; the next stride, or the full preview, tries
                    // again with more scan behind it.
                    frontierUsed = frontier;
                    continue;
                }

                var timed = new List<TranscriptSegment>(eligible.Count);

                lock (_alignedGate)
                {
                    _alignedFor.Clear();

                    for (var i = 0; i < eligible.Count; i++)
                    {
                        if (placed[i] is not { Count: > 0 } words
                            || words.Where(w => w.EndSeconds > w.StartSeconds).ToList()
                                is not { Count: > 0 } sounded)
                        {
                            timed.Add(eligible[i]);
                            continue;
                        }

                        var moved = eligible[i] with
                        {
                            StartSeconds = sounded[0].StartSeconds,
                            EndSeconds = Math.Max(sounded[^1].EndSeconds, sounded[0].StartSeconds),
                        };

                        _alignedFor[moved] = words;
                        timed.Add(moved);
                    }
                }

                _timedHead = timed;
                _timedHeadCovers = eligible.Count;

                SetTranscript([.. timed, .. raws.Skip(eligible.Count)]);

                timedThrough = timed[^1].EndSeconds;
                frontierUsed = frontier;
                UsableThroughSeconds = timedThrough;
            }
        }
        catch (OperationCanceledException)
        {
            // The run was cancelled or superseded; whoever did that owns the transcript now.
        }
        catch (Exception exception)
        {
            // Progressive timing is a convenience over the streamed text; losing it costs
            // nothing the final pass does not restore.
            LogError(exception);
        }
    }

    private async Task ScanForWordsAsync(
        PcmAudio audio,
        IProgress<double>? progress,
        CancellationToken cancellationToken)
    {
        if (ForcedAligner.Find(_modelRoot) is not { } directory)
        {
            return;
        }

        ForgetAlignment();

        try
        {
            await Task.Run(() =>
            {
                var aligner = ForcedAligner.Load(directory, _plan);

                try
                {
                    // Handed over before the scan rather than after, so the progressive pass
                    // can place words against the growing grid. Placing needs no model — it
                    // is a Viterbi walk over scores already written — so it never contends
                    // with the scan for the session.
                    _aligner = aligner;

                    _scores = aligner.Scan(audio, progress, cancellationToken, (grid, seconds) =>
                    {
                        lock (_frontierGate)
                        {
                            _scanPartial = grid;
                            _scanFrontierSeconds = seconds;
                        }
                    });
                }
                catch
                {
                    _aligner = null;
                    aligner.Dispose();
                    throw;
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
            // transcript. The stack is kept: a whole recording quietly falling back to
            // estimates presents as sync drift, and the status line alone was missable.
            Status = $"Transcribed, but the words could not be timed exactly: {exception.Message}";
            LogError(exception);
        }
    }

    /// <summary>
    /// Places the words of the finished transcript onto the frames the scan produced.
    /// <para>
    /// Cheap, because the model has already run: this is a Viterbi pass over a few hundred frames
    /// per segment. Anything it cannot place keeps the estimate, so a failure costs precision on
    /// one segment and nothing else.
    /// </para>
    /// </summary>
    /// <param name="keepAligner">
    /// Holds on to the model afterwards. The preview pass runs while cleanup is still
    /// rewriting the text, and the final pass over the cleaned text would otherwise pay a
    /// full reload for the release in between.
    /// </param>
    private async Task<IReadOnlyList<TimedSegment>> AlignWordsAsync(
        IReadOnlyList<TranscriptSegment> segments,
        IProgress<double>? progress,
        CancellationToken cancellationToken,
        bool keepAligner = false)
    {
        var untimed = () => (IReadOnlyList<TimedSegment>)
            [.. segments.Select(segment => new TimedSegment(segment, []))];

        _unheardTail = 0;

        if (_aligner is not { } aligner || _scores is not { } scores)
        {
            return untimed();
        }

        RecordAlignmentInput(segments);

        var placed = new List<TimedSegment>(segments.Count);
        var unheard = 0;
        var doubled = 0;

        try
        {
            await Task.Run(() =>
            {
                var limit = scores.SecondsAt(scores.Frames);

                // The whole transcript in one pass, so text order and time order are jointly
                // monotonic by construction. Per-segment windows were tried in every shape a
                // window has and each failed somewhere; the pass makes those failures
                // unrepresentable rather than tuned against.
                var all = aligner.AlignAll(scores, segments, cancellationToken);

                // A segment stamped past the end of the recording is almost always real speech
                // whose stamps the final padded window inflated — reading the last seconds of
                // one such recording back proved an outro convicted as invented was spoken,
                // word for word. The global pass places inflated stamps correctly on its own,
                // so this is only a safety net for text that truly has no audio, and conviction
                // demands both failures at once: words that neither occupy a plausible share of
                // the time the text needs nor read as themselves where they landed. Dropping
                // real speech is the worse lie, and this trial has told it once already.
                var dropped = new bool[segments.Count];

                for (var i = 0; i < segments.Count; i++)
                {
                    if (segments[i].StartSeconds < limit)
                    {
                        continue;
                    }

                    var claimed = Math.Max(
                        1.0, segments[i].EndSeconds - segments[i].StartSeconds);

                    var proof = all[i]?.Where(w => w.EndSeconds > w.StartSeconds).ToList();

                    var span = proof is { Count: > 0 }
                        ? proof[^1].EndSeconds - proof[0].StartSeconds
                        : 0;

                    var heard = proof is { Count: > 0 }
                        ? aligner.Read(scores, proof[0].StartSeconds, proof[^1].EndSeconds)
                        : string.Empty;

                    if (span < claimed * 0.3
                        && TextLikeness.Share(segments[i].Text, heard) < 0.45)
                    {
                        dropped[i] = true;
                        unheard++;
                    }
                }

                // Once judged, the pass runs again without the condemned lines. Under a global
                // path the invented tokens do not just cram — the path has to reserve real
                // frames for them, squeezing the words that actually own that audio, and one
                // invented line can even fake a passing grade with the room another's tokens
                // freed up. Dropping them and realigning returns the audio to its owners. The
                // pass costs well under a second, so a second one is free.
                if (unheard > 0)
                {
                    Repass();
                }

                // A line the transcriber wrote twice is the one failure the global pass cannot
                // absorb. Trimming catches the copies it can see at a seam, but a copy landing
                // a few words inside a segment survives it — and under a single monotone path a
                // surviving twin is no longer cosmetic: every letter of it must be funded with
                // real frames, so the path parks the copy on top of whatever is actually said
                // there and shoves the sentences around it seconds off. On one podcast a
                // duplicated "They cannot be victims to this." put the marker a few words
                // behind where the copy sat and a sentence or two behind by the end.
                //
                // Conviction takes two proofs, because reading badly alone is how mumbled and
                // crosstalked lines read too. The text must be a verbatim copy of a neighbour's
                // — folded, so hyphens and case cannot hide it — and the audio under its
                // placement must not read as it. One conviction per round, then the pass runs
                // again: a real line merely displaced by the twin is re-placed, not condemned
                // with it.
                for (var round = 0; round < 3; round++)
                {
                    var worst = -1;
                    // The bar is "reads as itself", not "reads as nothing". A twin parked on its
                    // neighbour's sentence still shares letters with it by coincidence — LCS is
                    // generous over short strings — and 0.35 let this one pass. A real verbatim
                    // repeat sits on its own audio and clears 0.5 easily.
                    var worstShare = 0.5;

                    for (var i = 0; i < segments.Count; i++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        if (dropped[i])
                        {
                            continue;
                        }

                        var own = TextLikeness.Fold(segments[i].Text);

                        if (own.Length < 12 || !CopiedNearby(i, own))
                        {
                            continue;
                        }

                        var proof = all[i]?.Where(w => w.EndSeconds > w.StartSeconds).ToList();

                        if (proof is not { Count: > 0 })
                        {
                            continue;
                        }

                        var heard = aligner.Read(
                            scores, proof[0].StartSeconds, proof[^1].EndSeconds);
                        var asItself = TextLikeness.Share(segments[i].Text, heard);

                        if (asItself < worstShare)
                        {
                            worstShare = asItself;
                            worst = i;
                        }
                    }

                    if (worst < 0)
                    {
                        break;
                    }

                    dropped[worst] = true;
                    doubled++;
                    Repass();
                }

                progress?.Report(1);

                bool CopiedNearby(int i, string own)
                {
                    for (var n = Math.Max(0, i - 2); n <= Math.Min(segments.Count - 1, i + 2); n++)
                    {
                        if (n != i && TextLikeness.Fold(segments[n].Text).Contains(own, StringComparison.Ordinal))
                        {
                            return true;
                        }
                    }

                    return false;
                }

                void Repass()
                {
                    var survivors = new List<TranscriptSegment>(segments.Count);
                    var back = new List<int>(segments.Count);

                    for (var i = 0; i < segments.Count; i++)
                    {
                        if (!dropped[i])
                        {
                            survivors.Add(segments[i]);
                            back.Add(i);
                        }
                    }

                    var again = aligner.AlignAll(scores, survivors, cancellationToken);
                    var remapped = new IReadOnlyList<WordTimings.Word>?[segments.Count];

                    for (var k = 0; k < back.Count; k++)
                    {
                        remapped[back[k]] = again[k];
                    }

                    all = remapped;
                }

                for (var i = 0; i < segments.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    if (dropped[i])
                    {
                        continue;
                    }

                    var words = all[i];

                    if (words is null)
                    {
                        placed.Add(new TimedSegment(segments[i], []));
                        continue;
                    }

                    var sounded = words.Where(w => w.EndSeconds > w.StartSeconds).ToList();

                    var moved = sounded.Count > 0
                        ? segments[i] with
                        {
                            StartSeconds = sounded[0].StartSeconds,
                            EndSeconds = Math.Max(sounded[^1].EndSeconds, sounded[0].StartSeconds),
                        }
                        : segments[i];

                    placed.Add(new TimedSegment(moved, words));
                }
            }, cancellationToken).ConfigureAwait(true);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception exception)
        {
            Status = $"Transcribed, but the words could not be timed exactly: {exception.Message}";
            LogError(exception);
            return untimed();
        }
        finally
        {
            if (!keepAligner)
            {
                ReleaseAlignmentModel();
            }
        }

        _unheardTail = unheard;
        _doubledLines = doubled;

        if (placed.Count + unheard + doubled != segments.Count)
        {
            // Never silent: this books every word as an estimate, and an estimated transcript
            // presents as sync drift with nothing anywhere saying why.
            Status = $"Transcribed, but the word timings did not balance "
                + $"({placed.Count} placed, {unheard} unheard, {doubled} doubled of {segments.Count}); "
                + "word times are estimates.";
            LogError(new InvalidOperationException(
                $"Alignment count mismatch: {placed.Count}+{unheard}+{doubled} != {segments.Count}"));
            return untimed();
        }

        return placed;
    }

    /// <summary>
    /// Writes down what aligning and attribution did to the segment boundaries.
    /// <para>
    /// A file rather than the status line: it is a page of numbers nobody wants during a
    /// transcription, and it needs to survive the run to be read afterwards.
    /// </para>
    /// </summary>
    private void Record(IReadOnlyList<TranscriptSegment> before, IReadOnlyList<TimedSegment> after)
    {
        try
        {
            var report = AlignmentCrowding.Describe(before, after);

            File.WriteAllText(
                CrowdingReportPath,
                AlignmentCrowding.Format(report, SourceName));

            if (report.OverlappedAfter > report.OverlappedBefore)
            {
                Status = $"Transcribed. {report.OverlappedAfter} of {Math.Max(1, report.Segments - 1)} "
                    + $"segment boundaries overlap — see {Path.GetFileName(CrowdingReportPath)}.";
            }
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            // A diagnostic that cannot be written is not worth failing a transcription over.
        }
    }

    /// <summary>Where the crowding report is written.</summary>
    public static string CrowdingReportPath { get; } =
        Path.Combine(Path.GetTempPath(), "localscribe-alignment.txt");

    /// <summary>
    /// Trailing lines dropped because the recording ends before they are spoken.
    /// </summary>
    private int _unheardTail;

    /// <summary>Lines dropped because the transcriber wrote them twice.</summary>
    private int _doubledLines;

    /// <summary>The last cleaned transcript as assembly received it, for targeted retries.</summary>
    private IReadOnlyList<TranscriptSegment>? _cleanedBase;

    /// <summary>The kept-raw instances inside <see cref="_cleanedBase"/>, by reference.</summary>
    private List<TranscriptSegment> _retryTargets = [];

    /// <summary>
    /// Writes the aligner's exact input — the transcriber's stamps as the corridor will anchor
    /// on them. The app's placements and the checker's diverge on the same audio, and the one
    /// input the checker cannot reproduce is these stamps: a saved archive's bounds have been
    /// through an alignment already. This makes the app's run repeatable outside the app.
    /// </summary>
    private static void RecordAlignmentInput(IReadOnlyList<TranscriptSegment> segments)
    {
        try
        {
            var text = new StringBuilder();

            foreach (var segment in segments)
            {
                text.AppendLine(FormattableString.Invariant(
                    $"{segment.StartSeconds:F2}\t{segment.EndSeconds:F2}\t{segment.Text}"));
            }

            File.WriteAllText(
                Path.Combine(Path.GetTempPath(), "localscribe-input.txt"), text.ToString());
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            // A diagnostic that cannot be written is not worth failing the transcript over.
        }
    }

    /// <summary>What "finished" should say, given what had to be left out along the way.</summary>
    private string DoneStatus()
    {
        var text = "Done.";

        if (_unheardTail > 0)
        {
            text += $" The recording ends before its last {_unheardTail} line(s) are spoken, "
                + "so they were left out.";
        }

        if (_doubledLines > 0)
        {
            text += $" {_doubledLines} line(s) the transcriber wrote twice were dropped.";
        }

        return text;
    }

    /// <summary>
    /// Lets go of the network but keeps what a second attempt would need.
    /// <para>
    /// Scanning the recording is most of the cost of timing words and depends only on the audio,
    /// so its result stays: about eight megabytes an hour, against a scan that takes roughly half
    /// the length of the recording to repeat. Cleaning up again then costs only the cleanup.
    /// </para>
    /// </summary>
    private void ReleaseAlignmentModel() => _aligner?.ReleaseModel();

    /// <summary>Lets go of the scores as well, when there is no transcript they belong to.</summary>
    private void ForgetAlignment()
    {
        _aligner?.Dispose();
        _aligner = null;
        _scores = null;
    }

    /// <summary>
    /// A scan started beside transcription, waiting for the finish stages to collect it.
    /// Nulled where it is consumed; a failed transcription leaves it to finish on its own —
    /// ScanForWordsAsync reports its own failures, and the next run's scan forgets its state.
    /// </summary>
    private Task? _scanInFlight;

    /// <summary>The alignment model, held only between the scan and the placing.</summary>
    private ForcedAligner? _aligner;

    /// <summary>What the model made of the recording, frame by frame.</summary>
    private AlignmentScores? _scores;

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

        if (turns is null)
        {
            return segments;
        }

        _lastTurns = turns;

        // Timed again first, because this runs over the segments as they were before speakers
        // were attached and the words held now belong to the pieces they were divided into. The
        // scan is kept, so this is the cheap half of alignment rather than a second pass over
        // the audio.
        var timed = await AlignWordsAsync(segments, found, _cancellation?.Token ?? default)
            .ConfigureAwait(true);

        return Attribute(timed, turns);
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
                // The thread budget is passed only where a --diarize turn diff has proven it moves
                // no boundary — macOS so far. Windows keeps its historical all-cores sessions
                // until the laptop runs the same measurement; the tuning is frozen and thread
                // count can reorder float summation.
                using var diarizer = SpeakerDiarizer.Load(
                    directory, _capabilities?.Platform == DevicePlatform.MacOS ? _plan : null);

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
                // Which of the two, as recorded beside the models. Neither wins everywhere:
                // tracking rescued a phone recording of an argument that clustering split into
                // nineteen speakers, and clustering found five voices in a studio podcast where
                // tracking found three and 22 turns in seven minutes.
                var method = DiarizationChoice.Read(
                    Path.Combine(_modelRoot, "diarization"));

                var found = method == DiarizationMethod.Voices
                    ? diarizer.Diarize(
                        audio,
                        maxSpeakers: speakers,
                        exactSpeakers: speakers,
                        progress: progress,
                        cancellationToken: _cancellation?.Token ?? default)
                    : diarizer.DiarizeByTracking(
                        audio,
                        maxSpeakers: speakers,
                        exactSpeakers: speakers,
                        progress: progress,
                        cancellationToken: _cancellation?.Token ?? default);

                // Both paths report crosstalk now: tracking measures it between windows, and
                // the voices path reads the segmentation model's own overlap classes — the
                // frames on which it heard two speakers at once.
                _overlaps = diarizer.LastOverlaps;

                return found;
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
        get => _speakerCount;
        private set
        {
            if (_speakerCount == value)
            {
                return;
            }

            _speakerCount = value;
            Raise(nameof(SpeakerCount));
        }
    }

    // A named backing field rather than the `field` keyword, which the macOS build's compiler
    // does not speak yet: this file is compiled into both apps, so it holds to the older of
    // the two languages.
    private int _speakerCount;

    private static int DistinctSpeakers(IReadOnlyList<TranscriptSegment> segments) =>
        segments
            .Select(segment => segment.Speaker)
            .Where(speaker => !string.IsNullOrWhiteSpace(speaker))
            .Distinct(StringComparer.Ordinal)
            .Count();

    /// <summary>
    /// Whether to render speech as English rather than write it down as spoken.
    /// <para>
    /// Off, and it stays off unless asked. A transcript that quietly changes language is not a
    /// transcript of the recording — which is how this was found: a Portuguese recording came
    /// back in English because the language had been inherited from an earlier file, and that
    /// read as a feature nobody had chosen.
    /// </para>
    /// </summary>
    public bool TranslateToEnglish
    {
        get => _translateToEnglish;
        set
        {
            if (_translateToEnglish == value)
            {
                return;
            }

            _translateToEnglish = value;
            Raise(nameof(TranslateToEnglish));
        }
    }

    private bool _translateToEnglish;

    /// <summary>
    /// The language of the transcript on screen, or null when nothing has been transcribed.
    /// </summary>
    public string? SpokenLanguage
    {
        get => _spokenLanguage;
        private set
        {
            if (_spokenLanguage == value)
            {
                return;
            }

            _spokenLanguage = value;
            Raise(nameof(SpokenLanguage));
            Raise(nameof(CanOfferTranslation));
        }
    }

    private string? _spokenLanguage;

    /// <summary>
    /// True when rendering this recording in English would change anything.
    /// <para>
    /// English speech is the case where the offer is noise, and no detected language at all is
    /// the case where it would be a guess. Both hide it.
    /// </para>
    /// </summary>
    public bool CanOfferTranslation =>
        _sourcePath is { Length: > 0 }
        && SpokenLanguage is { Length: > 0 } language
        && !language.Equals("en", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// What to ask the transcriber for.
    /// <para>
    /// Not named Task. A member called that shadows <see cref="System.Threading.Tasks.Task"/>
    /// throughout the class, and every <c>Task.Run</c> in the file stops compiling.
    /// </para>
    /// </summary>
    private SpeechTask RequestedTask =>
        TranslateToEnglish ? SpeechTask.TranslateToEnglish : SpeechTask.Transcribe;

    /// <summary>Where people talked over each other, from the last run.</summary>
    private IReadOnlyList<(double Start, double End)> _overlaps = [];

    /// <summary>The transcript as it came from the recogniser, before anything was done to it.</summary>
    private IReadOnlyList<TranscriptSegment> _rawSegments = [];

    /// <summary>Who spoke when, from the last run. Depends on the audio alone, so a retry keeps it.</summary>
    private IReadOnlyList<SpeakerTurn>? _lastTurns;

    /// <summary>
    /// What cleanup could not do, or null when it did everything asked of it.
    /// <para>
    /// Worth saying out loud. A cleanup model that gives up leaves the transcript readable but
    /// unpunctuated in places, which looks like the recording was unclear rather than like
    /// something went wrong and could be tried again.
    /// </para>
    /// </summary>
    public string? CleanupNotice
    {
        get => _cleanupNotice;
        private set
        {
            if (_cleanupNotice == value)
            {
                return;
            }

            _cleanupNotice = value;
            Raise(nameof(CleanupNotice));
            Raise(nameof(CanRetryCleanup));
        }
    }

    private string? _cleanupNotice;

    /// <summary>True when there is something to try again and something to try it on.</summary>
    public bool CanRetryCleanup => _cleanupNotice is not null && _rawSegments.Count > 0 && !IsBusy;

    /// <summary>
    /// Attaches speakers, cutting segments at the word the voice changed on where the words were
    /// measured and at a sentence end where they were not.
    /// <para>
    /// The sentence rule stays for the machine with no aligner installed, and keeps its repair
    /// for a speaker change that lands mid-sentence — which, without word times, is far more
    /// often a wrong turn than a real interruption. With word times the cut is evidence rather
    /// than a guess, so it is left alone.
    /// </para>
    /// </summary>
    private IReadOnlyList<TranscriptSegment> Attribute(
        IReadOnlyList<TimedSegment> timed,
        IReadOnlyList<SpeakerTurn> turns)
    {
        if (timed.Any(t => t.Words.Count > 0))
        {
            var pieces = WordLevelAttribution.Apply(timed, turns);

            // Before the aligned-times table is keyed, because the mark rewrites the segment
            // records and the table is keyed by value — marking afterwards would quietly hand
            // every crosstalk paragraph a loudness estimate instead of its measured words.
            pieces = CrosstalkMarks.Apply(pieces, _overlaps);

            // Renumbering rewrites the records too, so it obeys the same law: before the
            // table is keyed. Cluster order is not speaking order, and a transcript that
            // opens with "Speaker 2" reads as a bug even when the separation is right.
            var relabelled = SpeakerLabels.RenumberByAppearance([.. pieces.Select(p => p.Segment)]);
            pieces = [.. pieces.Select((p, i) => p with { Segment = relabelled[i] })];

            // Here rather than after aligning. Aligning is not the last thing that moves a
            // boundary — dividing segments between speakers sets new ones from the words, and
            // the tidying that stops two segments claiming the same moment runs after that. A
            // report taken before either could not see whether they worked, and did not: it came
            // back byte-identical across a build that changed both.
            Record([.. timed.Select(t => t.Segment)], pieces);

            lock (_alignedGate)
            {
                _alignedFor.Clear();

                foreach (var piece in pieces.Where(p => p.Words.Count > 0))
                {
                    _alignedFor[piece.Segment] = piece.Words;
                }
            }

            return [.. pieces.Select(p => p.Segment)];
        }

        var attributed = SpeakerDiarizer.Attribute([.. timed.Select(t => t.Segment)], turns);

        // No measured words on this path, so the bounds-based mark is the honest one.
        return SpeakerLabels.RenumberByAppearance(SpeakerAttribution.MarkOverlaps(
            SpeakerAttribution.KeepSentencesWhole(attributed), _overlaps));
    }

    /// <summary>Says what cleanup left undone, or takes the notice away when it did not.</summary>
    private void ReportCleanup(TranscriptRefiner? refiner) =>
        CleanupNotice = refiner is null
            ? null
            : CleanupReport.Describe(refiner.Failed, refiner.Rejected, refiner.LastError);


    /// <summary>
    /// Runs cleanup again over the original transcript, keeping everything that does not depend
    /// on it.
    /// <para>
    /// Speaker turns come from the audio and the alignment scan comes from the audio, so neither
    /// is redone: only the stage that failed is repeated, and the stages downstream of it that
    /// read its text.
    /// </para>
    /// </summary>
    public async Task RetryCleanupAsync()
    {
        if (_rawSegments.Count == 0 || IsBusy)
        {
            return;
        }

        // Asked again rather than trusted — which is what made Try again once fail exactly
        // the way the run had, against a port the restarted service no longer held.
        await EnsureCleanupBackendAnswersAsync();

        if (BuildRefiner() is not { } refiner)
        {
            Status = "No cleanup model is running any more, so there is nothing to retry with.";
            return;
        }

        _cancellation?.Dispose();
        _cancellation = new CancellationTokenSource();
        var cancellationToken = _cancellation.Token;

        IsBusy = true;
        CleanupNotice = null;
        Progress = 0;
        Status = "Cleaning up again…";

        try
        {
            var stages = new SharedBar(this);

            // Only the passages that failed, when the last run recorded which ones those
            // were: re-cleaning eight windows because one timed out spent minutes to redo
            // work that was already right. The failed instances sit verbatim in the last
            // cleaned list, so each consecutive run of them is one small transcript to
            // re-clean and splice back.
            if (_cleanedBase is { Count: > 0 } baseline && _retryTargets is { Count: > 0 } targets)
            {
                var wanted = new HashSet<TranscriptSegment>(
                    targets, ReferenceEqualityComparer.Instance);

                var repaired = new List<TranscriptSegment>(baseline.Count);
                var at = 0;

                while (at < baseline.Count)
                {
                    if (!wanted.Contains(baseline[at]))
                    {
                        repaired.Add(baseline[at]);
                        at++;
                        continue;
                    }

                    var run = new List<TranscriptSegment>();

                    while (at < baseline.Count && wanted.Contains(baseline[at]))
                    {
                        run.Add(baseline[at]);
                        at++;
                    }

                    var part = await refiner.RefineAsync(
                        new Transcript(run),
                        Glossary,
                        RefinementOutputs.Default,
                        new Progress<double>(stages.Cleanup),
                        cancellationToken: cancellationToken).ConfigureAwait(true);

                    repaired.AddRange(part.CleanedSegments ?? run);
                }

                await AssembleAsync(repaired, refiner, stages, cancellationToken)
                    .ConfigureAwait(true);
            }
            else
            {
                var refinement = await CleanWhenAwakeAsync(
                    refiner, new Transcript(_rawSegments), stages, cancellationToken).ConfigureAwait(true);

                await AssembleAsync(
                    refinement?.CleanedSegments ?? _rawSegments, refiner, stages, cancellationToken)
                    .ConfigureAwait(true);
            }

            Status = CleanupNotice is null
                ? "Cleaned up."
                : "Tried again, and some passages were still left as transcribed.";
        }
        catch (OperationCanceledException)
        {
            Status = "Stopped.";
        }
        catch (Exception exception)
        {
            Status = $"Could not clean up again: {exception.Message}";
        }
        finally
        {
            IsBusy = false;
            Progress = 0;
            Raise(nameof(CanRetryCleanup));
        }
    }

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
        catch (OperationCanceledException)
        {
            Status = "Stopped.";
            return;
        }
        catch (Exception exception)
        {
            // Escaping here would take the whole app down with it — this runs from a click.
            Status = $"Could not work out the speakers: {exception.Message}";
            LogError(exception);
            return;
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
                // The thread budget is passed only where a --diarize turn diff has proven it moves
                // no boundary — macOS so far. Windows keeps its historical all-cores sessions
                // until the laptop runs the same measurement; the tuning is frozen and thread
                // count can reorder float summation.
                using var diarizer = SpeakerDiarizer.Load(
                    directory, _capabilities?.Platform == DevicePlatform.MacOS ? _plan : null);

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
        CancellationToken cancellationToken,
        double progressFloor = 0)
    {
        // Before the refiner is built from it, so a backend that restarted since discovery is
        // rediscovered instead of failing every cleanup window one notice deep.
        await EnsureCleanupBackendAnswersAsync();

        var refiner = BuildRefiner();

        _rawSegments = spoken;

        if (refiner is null && audio is null)
        {
            _segmentsBeforeSpeakers = spoken;
            SetTranscript(spoken);
            return;
        }

        var stages = new SharedBar(this, progressFloor);
        var transcript = new Transcript(spoken);

        // Cleanup waits for warm weights inside its own lane, so the diarizer and the scan —
        // which need no language model — never wait behind a cold Foundry.
        var cleaning = refiner is null
            ? Task.FromResult<RefinementResult?>(null)
            : CleanWhenAwakeAsync(refiner, transcript, stages, cancellationToken);

        var listening = audio is null
            ? Task.FromResult<IReadOnlyList<SpeakerTurn>?>(null)
            : FindTurnsAsync(audio, speakers: null, new Progress<double>(stages.Speakers));

        // A third stage beside the other two, and this time one that can genuinely run there. It
        // reads the audio and writes nothing the others touch.
        // A scan already started beside transcription is collected rather than restarted;
        // the recording path, which has no audio until the microphone stops, starts one here.
        var scanning = _scanInFlight
            ?? (audio is null
                ? Task.CompletedTask
                : ScanForWordsAsync(audio, new Progress<double>(stages.Words), cancellationToken));

        _scanInFlight = null;

        // The fourth lane: the moment the scan lands, the raw transcript gets its word times
        // and goes on screen — clickable, playable, highlight following the voice — while
        // cleanup and diarization are still minutes from done. Placing words is the cheap half
        // of alignment once the scan exists, and neither of the slow stages is needed for it:
        // diarization only attaches names to words already timed, and cleanup only rewrites
        // text. The missing names and raw punctuation are themselves the honest sign that the
        // transcript is still being finished; the final assembly replaces it in place.
        var previewing = PreviewTimedAsync(spoken, scanning, stages, cancellationToken);

        await Task.WhenAll(cleaning, listening, scanning, previewing);

        var refinement = await cleaning;
        var turns = await listening;

        _lastTurns = turns;

        // The cleaned segments when cleanup ran, the raw ones when it did not. This is the line
        // that was missing for a long time: the cleaned text used to be computed and dropped,
        // and what the user read was always the raw transcript.
        await AssembleAsync(refinement?.CleanedSegments ?? spoken, refiner, stages, cancellationToken);
    }

    /// <summary>
    /// Puts the raw transcript on screen with its words timed, as soon as the scan allows.
    /// <para>
    /// A failure here costs only the preview — the final assembly neither knows nor cares —
    /// so nothing but a real cancellation is allowed out.
    /// </para>
    /// </summary>
    private async Task PreviewTimedAsync(
        IReadOnlyList<TranscriptSegment> spoken,
        Task scanning,
        SharedBar stages,
        CancellationToken cancellationToken)
    {
        try
        {
            await scanning.ConfigureAwait(true);

            if (_scores is null || spoken.Count == 0)
            {
                return;
            }

            var timed = await AlignWordsAsync(
                spoken, progress: null, cancellationToken, keepAligner: true).ConfigureAwait(true);

            cancellationToken.ThrowIfCancellationRequested();

            // The times table is keyed here the way Attribute keys it, because the window
            // looks words up by segment value; the final assembly clears and rekeys it.
            lock (_alignedGate)
            {
                _alignedFor.Clear();

                foreach (var piece in timed.Where(t => t.Words.Count > 0))
                {
                    _alignedFor[piece.Segment] = piece.Words;
                }
            }

            SetTranscript([.. timed.Select(t => t.Segment)]);
            UsableThroughSeconds = 0;
            stages.PreviewReady();
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception exception)
        {
            LogError(exception);
        }
    }

    /// <summary>
    /// Everything that follows from the cleaned text: repeats trimmed, speakers attached, words
    /// timed.
    /// <para>
    /// Separate from the stages above it because cleanup can be run again on its own. What comes
    /// out of the audio — who spoke when, and the scan the word times are placed against — does
    /// not change when the text does, so a second attempt repeats only the stage that failed and
    /// the ones that read its output.
    /// </para>
    /// </summary>
    private async Task AssembleAsync(
        IReadOnlyList<TranscriptSegment> cleaned,
        TranscriptRefiner? refiner,
        SharedBar stages,
        CancellationToken cancellationToken)
    {
        // Read across the segments, not just inside each one. The stitcher repairs what it can
        // see at the seam between two windows, but the two copies of a looped sentence usually
        // land in two different segments — the transcriber breaks where a sentence ends, which
        // is exactly between them — and cleanup can rewrite the text after the stitcher has
        // finished with it. This is the last point before the reader.
        // Remembered before trimming, because the retry re-cleans against this exact list:
        // the kept-raw instances the refiner recorded are found in it by reference.
        _cleanedBase = cleaned;
        _retryTargets = refiner?.KeptRaw.ToList() ?? [];

        cleaned = RepeatedPhrase.TrimAcross(cleaned);

        // A segment with no letters or digits in it was never speech — the transcriber pads
        // music and silence with ellipses. It cannot be aligned, so it would keep its raw
        // stamp and land between neighbours placed by measurement, out of order: one such "…"
        // printed itself thirty seconds deep into the middle of a conversation. A reader
        // loses nothing a reader ever had.
        cleaned = [.. cleaned.Where(segment => segment.Text.Any(char.IsLetterOrDigit))];

        _segmentsBeforeSpeakers = cleaned;

        // Words first, then speakers. The other way round for a long time, and that was what
        // lost most of the diarization: a segment could only be cut where a sentence ended, so a
        // speaker change with no sentence end near it had nowhere to go and the whole segment
        // went to whoever held most of it. On the debate recording 26 of 31 changes fell inside
        // a segment. Measured word times give a boundary to cut on.
        var timed = _scores is not null
            ? await AlignWordsAsync(cleaned, new Progress<double>(stages.Words), cancellationToken)
            : [.. cleaned.Select(segment => new TimedSegment(segment, []))];

        // Crosstalk is marked inside Attribute, not here: marking rewrites the segment records,
        // and the aligned-times table Attribute builds is keyed by value — a record rewritten
        // after keying quietly trades its measured words for a loudness estimate.
        var finished = _lastTurns is null
            ? [.. timed.Select(t => t.Segment)]
            : Attribute(timed, _lastTurns);

        SetTranscript(finished);
        RecordSpans(finished);

        ReportCleanup(refiner);
    }

    /// <summary>
    /// The cleanup stage, behind the wake: waits (cancellably) for the background weight load
    /// that startup began, then cleans. A wake that failed is not a reason to skip trying —
    /// the refiner degrades per window and reports what it could not do.
    /// </summary>
    private async Task<RefinementResult?> CleanWhenAwakeAsync(
        TranscriptRefiner refiner,
        Transcript transcript,
        SharedBar stages,
        CancellationToken cancellationToken)
    {
        await EnsureCleanupAwakeAsync().WaitAsync(cancellationToken).ConfigureAwait(true);

        return await CleanAsync(refiner, transcript, stages, cancellationToken).ConfigureAwait(true);
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

            // Not shown as it lands any more. Streaming the half-cleaned text over the
            // screen was safe when nothing on it could be lost; now the progressive preview
            // has the head timed and clickable, and each cleanup window's publish wiped that
            // back to grey — reported as clickable text reverting, timed to the exact moment
            // the finish stages began. The cleaned text arrives once, at final assembly,
            // timed and whole.
            cleanedSoFar: null,
            cancellationToken: cancellationToken).ConfigureAwait(true);

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
    private sealed class SharedBar(MainViewModel owner, double floor = 0)
    {
        private readonly object _gate = new();

        private double _cleanup;
        private double _speakers;
        private double _words;
        private bool _cleaningUp;
        private bool _findingSpeakers;
        private bool _timingWords;
        private bool _usable;

        /// <summary>
        /// The timed preview is on screen: from here on, every progress line leads with the
        /// part the user can act on. "Working out who spoke… 43%" reads as "wait"; the same
        /// words after "you can use this now" read as what they are — background work.
        /// </summary>
        public void PreviewReady()
        {
            lock (_gate)
            {
                _usable = true;
            }

            Publish();
        }

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

                if (_usable)
                {
                    // Compact on purpose: this shares one status line with a percentage, and
                    // the first wording taught the interaction so thoroughly it ran off the
                    // end of the bar.
                    what = $"Transcript ready — click any line. {what}";
                }
                else if (owner.UsableThroughSeconds > 0)
                {
                    what = $"Clickable through {TranscriptFormatter.Clock(owner.UsableThroughSeconds)} "
                        + $"(grey still timing). {what}";
                }
            }

            // Mapped above the floor the earlier phase already earned: the bar once jumped
            // from full back to nearly empty when the finish stages began, which read as the
            // run going backwards — and was the landmark users learned to dread.
            owner.Progress = floor + (fraction * (1 - floor));
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
            _alignedFor.Clear();
        }

        // A run abandoned between the scan and the placing would otherwise leave six hundred
        // megabytes of model loaded with nothing left to do with it.
        ForgetAlignment();

        _rawSegments = [];
        _lastTurns = null;
        CleanupNotice = null;

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
        private set
        {
            Set(ref _busy, value);

            // Whether a second attempt can be made turns on this, and a button that stays greyed
            // out after the work finishes is a button nobody can use.
            Raise(nameof(CanRetryCleanup));
        }
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
        _cleanupAwake = false;
        _cleanupWake = null;

        if (_languageModel is null)
        {
            await TryResumeCleanupBackendAsync().ConfigureAwait(true);
        }

        capabilities = capabilities with { LocalLanguageModelPresent = _languageModel is not null };

        _capabilities = capabilities;
        _plan = AcceleratorPlanner.Plan(capabilities);

        // Where the work will run, but not which weights: that is not known until they are
        // opened, and naming the size the planner asked for is how the window spent a session
        // claiming medium.en while running large-v3-turbo.
        AnnounceHardware();
        Status = _plan.Warnings.Count == 0
            ? "Ready."
            : $"Ready, with {_plan.Warnings.Count} warning(s). Run localscribe-doctor for detail.";

        // Fire-and-forget on purpose: the weights load while the user opens a file or the
        // first transcription runs. Only the cleanup stage ever waits on this task.
        _ = EnsureCleanupAwakeAsync();

        if (_waitingFor is { Length: > 0 } held)
        {
            _waitingFor = null;
            await TranscribeFileAsync(held).ConfigureAwait(true);
        }
    }

    /// <summary>
    /// Starts an already-installed Foundry Local, in case its service simply is not running —
    /// which it is not after every reboot, and the banner used to ask for a click each launch
    /// just to start it. Resuming a tool the user already set up is not an install: nothing
    /// is downloaded, nothing is added, and if the service starts but no model answers, the
    /// banner stays and its button still owns the download.
    /// </summary>
    private async Task TryResumeCleanupBackendAsync()
    {
        try
        {
            var foundry = new FoundryLocalManager();

            if (!await foundry.IsInstalledAsync())
            {
                return;
            }

            Status = "Starting Foundry Local…";

            var started = await foundry.StartServiceAsync();

            if (!started.Succeeded)
            {
                Status = started.Message;
                return;
            }

            _languageModel = await LocalLanguageModel.ResolveAsync();

            // A service asked its models the instant it starts can answer with none while it
            // is still taking inventory — and that momentary blank once put the download
            // banner in front of a machine that had a model all along. One short second look
            // costs two seconds; the wrongly-offered gigabyte cost rather more.
            if (_languageModel is null)
            {
                await Task.Delay(TimeSpan.FromSeconds(2));
                _languageModel = await LocalLanguageModel.ResolveAsync();
            }

            _cleanupAwake = false;
            _cleanupWake = null;
        }
        catch (Exception exception)
        {
            // Optional convenience; failing it quietly leaves exactly the launch we had before.
            LogError(exception);
        }
    }

    /// <summary>
    /// Transcription's share of the one continuous progress bar on a file run; the finish
    /// stages own the rest. Roughly proportional on the reference recordings, and the exact
    /// split matters less than the bar never running backwards.
    /// </summary>
    private const double TranscriptionShare = 0.6;

    /// <summary>
    /// A file opened before there was anything to transcribe it with, kept until there is.
    /// <para>
    /// One, not a queue: opening a second file before the first has started means the second is
    /// the one wanted.
    /// </para>
    /// </summary>
    private string? _waitingFor;

    /// <summary>
    /// Transcribes the open recording again, in English this time, or back as spoken.
    /// <para>
    /// A second pass over the audio rather than a translation of the text. Whisper renders
    /// English in the same pass that recognises speech, so there is nothing to translate
    /// afterwards — and going back is the same work in the other direction.
    /// </para>
    /// </summary>
    public async Task TranslateAgainAsync(bool intoEnglish)
    {
        if (_sourcePath is not { Length: > 0 } path || IsBusy)
        {
            return;
        }

        TranslateToEnglish = intoEnglish;

        await TranscribeFileAsync(path).ConfigureAwait(true);
    }

    /// <summary>Where the open recording was read from, so it can be transcribed again.</summary>
    private string? _sourcePath;

    /// <summary>Transcribes a file from disk.</summary>
    public async Task TranscribeFileAsync(string path)
    {
        if (_plan is null)
        {
            // Held rather than refused. A file is still there in a few seconds' time, and the
            // alternative is telling somebody who has just opened one to open it again — which
            // reads as the app having ignored them, because from where they sit it did.
            //
            // The microphone is the opposite case and stays refused: audio arriving before there
            // is anything to transcribe it with is audio lost, and a recording that silently
            // began late would be worse than one that plainly did not start.
            _waitingFor = path;
            Status = $"{Path.GetFileNameWithoutExtension(path)} is queued — still checking hardware.";
            return;
        }

        IsBusy = true;
        Progress = 0;
        SetTranscript([]);
        _rawSoFar = [];
        _timedHead = [];
        _timedHeadCovers = 0;
        UsableThroughSeconds = 0;

        lock (_frontierGate)
        {
            _scanPartial = null;
            _scanFrontierSeconds = 0;
        }

        _cancellation = new CancellationTokenSource();

        try
        {
            Status = AudioFileLoader.IsVideo(path) ? "Extracting audio…" : "Loading audio…";
            var audio = await Task.Run(() => AudioFileLoader.Load(path), _cancellation.Token);

            SourceName = Path.GetFileNameWithoutExtension(path);
            _audio = audio;
            _sourcePath = path;

            // Held so a line in the transcript can be clicked and heard. The timings refer to
            // these samples, not to the file, which is why the decoded audio is what is kept.
            Player.Load(audio);

            // The scan is the pipeline's long pole and needs only the audio, while the
            // transcriber's heavy half runs on the accelerator with the CPU near idle — so
            // the two overlap from here, and the finish stages collect a scan that is
            // usually already done. Measured on a 63-second file: the scan alone was 41
            // seconds, transcription 12, and running them in sequence spent both.
            _scanInFlight = ScanForWordsAsync(audio, progress: null, _cancellation.Token);

            // Beside the scan, the lane that makes its progress usable: the head of the
            // transcript gains real word times while the tail is still being scanned.
            _ = RunProgressiveTimingAsync(_cancellation.Token);

            Status = IsModelReady ? "Preparing…" : "Loading the model…";
            var transcriber = await Task.Run(() => TranscriberFor(_plan), _cancellation.Token);
            var pipeline = new TranscriptionPipeline(transcriber, BuildRefiner());

            Status = $"Transcribing with {transcriber.Description}…";

            // Each window's text as it lands, rather than a bare percentage. A long file is
            // otherwise several minutes of a progress bar and nothing to read.
            var streamed = new List<TranscriptSegment>();

            var progress = new Progress<TranscriptionProgress>(update =>
            {
                Progress = update.Fraction * TranscriptionShare;

                Status = $"Transcribing… {update.ChunksCompleted} of {update.ChunksTotal} windows";

                if (update.Phase == TranscriptionPhase.Transcribing && update.LatestText.Length > 0
                    && update.LatestEndSeconds > update.LatestStartSeconds)
                {
                    // The window's real range, not window-number-times-thirty: windows overlap
                    // by two seconds, and the arithmetic stamp ran ninety seconds late by the
                    // end of a long recording — every streamed anchor beyond the aligner's
                    // reach, which is why the in-progress sync was so much worse than the
                    // finished one.
                    streamed.Add(new TranscriptSegment(
                        update.LatestText,
                        update.LatestStartSeconds,
                        update.LatestEndSeconds));

                    // The progressive pass times a prefix of exactly this list, so the raw
                    // tail must be these segments, not a rebuilt approximation — and the
                    // timed head it has already published must survive this update, or the
                    // transcript would visibly un-time itself once a window.
                    var raws = streamed.ToList();
                    _rawSoFar = raws;

                    SetTranscript([.. _timedHead, .. raws.Skip(_timedHeadCovers)]);
                }
            });

            var transcript = await pipeline.TranscribeAsync(audio, progress, _cancellation.Token, RequestedTask);

            // Now, rather than before: what language this is could not be known until it had been
            // listened to, and offering to translate before that would have been a guess.
            SpokenLanguage = transcriber.DetectedLanguage;

            // The speech model's gigabytes go back before the finish stages: the scan owns
            // the CPU for minutes while the engine sits idle, and on a 16 GB machine
            // idle-but-resident is what pushed a long recording into swap, starved the scan,
            // and put every word on an estimate.
            ReleaseTranscriberIfCheap();

            await FinishTranscriptAsync(
                transcript.Segments, audio, _cancellation.Token, TranscriptionShare);

            Status = DoneStatus();
        }
        catch (OperationCanceledException exception)
            when (_cancellation?.IsCancellationRequested != true)
        {
            // A cancellation nobody asked for is an engine failure wearing a costume —
            // HttpClient reports a timeout this way, and so can a binding tearing down.
            // Calling it "Cancelled." blamed the user for something they did not do and
            // threw away the evidence; the error file keeps the stack.
            Status = "Failed: the engine stopped mid-run without being asked to. "
                + $"Details in {Path.GetFileName(ErrorLogPath)}.";
            LogError(exception);
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
            LogError(exception);
        }
        finally
        {
            // Cancelled before disposal so a scan left in flight by a failed run stops now,
            // rather than finishing later and writing another recording's scores over the
            // state a new operation is building. After a successful run the scan has already
            // been collected and the cancel touches nothing.
            _cancellation?.Cancel();
            _scanInFlight = null;

            IsBusy = false;
            _cancellation?.Dispose();
            _cancellation = null;
        }
    }

    /// <summary>Where failures keep their stacks; the status line only has room for a verdict.</summary>
    public static string ErrorLogPath { get; } =
        Path.Combine(Path.GetTempPath(), "localscribe-errors.txt");

    private static void LogError(Exception exception)
    {
        try
        {
            File.AppendAllText(
                ErrorLogPath,
                $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} {exception}{Environment.NewLine}{Environment.NewLine}");
        }
        catch (Exception failure) when (failure is IOException or UnauthorizedAccessException)
        {
            // A diagnostic that cannot be written is not worth failing anything over.
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
        // Recording is a decision made now, so it displaces a file that was only ever waiting.
        _waitingFor = null;

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

        _liveSession = new LiveTranscriptionSession(transcriber, task: RequestedTask);
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
            _alignedFor.Clear();
        }

        // A run abandoned between the scan and the placing would otherwise leave six hundred
        // megabytes of model loaded with nothing left to do with it.
        ForgetAlignment();

        _rawSegments = [];
        _lastTurns = null;
        CleanupNotice = null;

        Player.Clear();
        SourceName = $"Recording {DateTime.Now:yyyy-MM-dd HH.mm}";

        // Guarded, because this is where capture genuinely fails: a device in use, no input
        // device at all, or — on macOS — microphone permission refused. Unguarded, the
        // exception escaped through the click handler and took the process with it, which
        // turns "the app needs a permission" into "the app crashed".
        try
        {
            _microphone = new MicrophoneCapture();
            _microphone.SamplesAvailable += OnSamplesAvailable;
            _microphone.Start();
        }
        catch (Exception exception)
        {
            _microphone?.Dispose();
            _microphone = null;

            var session = _liveSession;
            _liveSession = null;

            if (session is not null)
            {
                await session.DisposeAsync();
            }

            _cancellation?.Dispose();
            _cancellation = null;
            _diagnostics = null;
            _liveCapture = null;

            IsPreparing = false;
            Status = $"Could not start the microphone: {exception.Message}";
            return;
        }

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
        // With an injected engine the ONNX directory is advice, not a requirement — a Mac with
        // only a whisper.cpp model on disk must not fail here for want of weights it will
        // never open.
        var directory = _openTranscriber is null
            ? ModelDirectoryFor(plan)
            : ModelLayout.Locate(
                _modelRoot, _capabilities?.Family ?? SocFamily.Unknown,
                plan.Encoder.Device, plan.WhisperModel);

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

            var opened = _openTranscriber is not null
                ? _openTranscriber(plan, directory)
                : WhisperOnnxTranscriber.Load(directory!, plan);

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
    /// Lets go of the cached transcriber, but only where reopening is nearly free. The
    /// injected engines measured sub-second loads from a warm cache; the QNN engine keeps its
    /// slot, because its five-second open is the reason the cache exists.
    /// </summary>
    private void ReleaseTranscriberIfCheap()
    {
        if (_openTranscriber is null)
        {
            return;
        }

        _transcriberLock.Wait();

        try
        {
            _transcriber?.Dispose();
            _transcriber = null;
            _transcriberKey = null;
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
        // An injected engine is not preloaded: the reason this method exists is QNN's
        // five-second session open, and whisper.cpp measured 0.9 s from a warm cache —
        // preloading there spends gigabytes of residency from launch to save less than a
        // second, which on a 16 GB machine helped push a long transcription into swap.
        if (_plan is null
            || _openTranscriber is not null
            || Environment.GetEnvironmentVariable("LOCALSCRIBE_NO_PRELOAD") is { Length: > 0 })
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
                Status = DoneStatus();
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

    /// <summary>A health check that treats any failure as "not answering".</summary>
    private static async Task<bool> AnswersAsync(ILanguageModel model)
    {
        try
        {
            return await model.IsAvailableAsync();
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Makes sure the cached cleanup backend still answers before anything leans on it.
    /// Foundry binds a dynamic port, and a service restarted since discovery leaves the
    /// cached client talking to an address nobody holds. A ping costs milliseconds and runs
    /// once per run; rediscovery — asking the CLI where the service lives now — runs only
    /// when the ping fails.
    /// </summary>
    private async Task EnsureCleanupBackendAnswersAsync()
    {
        if (_languageModel is { } cached && !await AnswersAsync(cached))
        {
            (cached as IDisposable)?.Dispose();
            _languageModel = await LocalLanguageModel.ResolveAsync();
            AnnounceHardware();
            Raise(nameof(CleanupModel));
        }
    }

    private void AnnounceHardware()
    {
        if (_plan is not { } plan)
        {
            return;
        }

        // Naming the cleanup backend, not just its device. Which model is doing the punctuation
        // decides how good the punctuation is, and this is the line people read when the answer
        // is "not very".
        var cleanup = _languageModel is { } model
            ? model.Description
            : "no cleanup model found";

        HardwareSummary = $"encoder on {plan.Encoder.Device}, decoder on {plan.Decoder.Device}, "
            + $"cleanup: {cleanup}";
    }

    /// <summary>
    /// Gets a cleanup model running, from wherever the machine currently is: installs Foundry
    /// Local if its CLI is missing, starts the service, downloads the default model, and
    /// reconnects. Everything is user-initiated — this runs from a button that says what it
    /// does — and everything it downloads runs on this machine.
    /// <para>
    /// If a transcript is already open, cleanup runs on it once the model is up, because
    /// enabling cleanup was the whole point of the click.
    /// </para>
    /// </summary>
    public async Task<bool> ProvisionCleanupAsync()
    {
        if (_languageModel is not null)
        {
            return true;
        }

        if (_provisioningCleanup)
        {
            return false;
        }

        _provisioningCleanup = true;

        try
        {
            var foundry = new FoundryLocalManager();

            // The fraction reaches the progress bar when a stage reports one — the model
            // download does — so a multi-hundred-megabyte fetch reads as movement rather than
            // as a stuck status line.
            var progress = new Progress<InstallProgress>(p =>
            {
                Status = p.Message;

                if (p.Fraction is { } fraction)
                {
                    Progress = fraction;
                }
            });

            if (!await foundry.IsInstalledAsync())
            {
                var installed = await foundry.InstallAsync(progress);

                if (!installed.Succeeded)
                {
                    Status = installed.Message;
                    return false;
                }
            }

            var started = await foundry.StartServiceAsync(progress);

            if (!started.Succeeded)
            {
                Status = started.Message;
                return false;
            }

            // What the service already holds comes first — the same rule the resolver lives
            // by. This button once named the default alias unconditionally, and on a machine
            // whose cache held a perfectly good model under another name, one click bought a
            // gigabyte of second model nobody needed.
            Status = "Checking what the service already has…";
            _languageModel = await LocalLanguageModel.ResolveAsync();

            if (_languageModel is not null)
            {
                _cleanupAwake = false;
                _cleanupWake = null;
                AnnounceHardware();
                Raise(nameof(CleanupModel));
                Status = $"Cleanup ready on {_languageModel.Description}.";

                if (_rawSegments.Count > 0 && !IsBusy)
                {
                    await RetryCleanupAsync();
                }

                return true;
            }

            // Idempotent: a model already in the cache reports success without moving data.
            var downloaded = await foundry.DownloadModelAsync(progress: progress);

            if (!downloaded.Succeeded)
            {
                Status = downloaded.Message;
                return false;
            }

            Status = "Connecting to the cleanup model…";
            _languageModel = await LocalLanguageModel.ResolveAsync();
            _cleanupAwake = false;
            _cleanupWake = null;

            if (_languageModel is null)
            {
                Status = "Foundry Local is running, but no model answered. "
                    + "Try 'foundry service status' in a terminal.";
                return false;
            }

            AnnounceHardware();
            Raise(nameof(CleanupModel));
            Status = $"Cleanup ready on {_languageModel.Description}.";

            if (_rawSegments.Count > 0 && !IsBusy)
            {
                await RetryCleanupAsync();
            }

            return true;
        }
        catch (Exception exception)
        {
            Status = $"Could not set up the cleanup model: {exception.Message}";
            return false;
        }
        finally
        {
            _provisioningCleanup = false;
        }
    }

    private bool _provisioningCleanup;

    private TranscriptRefiner? BuildRefiner() =>
        _languageModel is null ? null : new TranscriptRefiner(_languageModel);

    /// <summary>
    /// True once the cleanup backend has answered a completion this session.
    /// </summary>
    private bool _cleanupAwake;

    /// <summary>The wake in flight, so every path waits on one load rather than starting its own.</summary>
    private Task? _cleanupWake;

    /// <summary>
    /// Starts waking the cleanup backend if nothing has, and returns the task to wait on.
    /// <para>
    /// Called fire-and-forget at startup so the weights load while the user opens a file or
    /// the first transcription runs — only the cleanup stage actually waits for them, in its
    /// own parallel lane. Deliberately not cancellable: the load benefits the whole session,
    /// and a cancelled transcription abandoning it would just move the wait to the next one.
    /// </para>
    /// </summary>
    private Task EnsureCleanupAwakeAsync()
    {
        if (_cleanupAwake || _languageModel is null)
        {
            return Task.CompletedTask;
        }

        // A failed wake is retried by whoever asks next; the backend may simply need longer.
        if (_cleanupWake is { IsCompleted: false } or { IsCompletedSuccessfully: true })
        {
            return _cleanupWake;
        }

        _cleanupWake = WakeCleanupAsync();
        return _cleanupWake;
    }

    /// <summary>
    /// First contact with the cleanup backend loads its weights — Foundry Local keeps the
    /// model cold until asked, and the load can take a minute or more. Left unnarrated, that
    /// minute is a cleanup bar parked at 0% and, before the refiner learned better, a timeout
    /// dressed as a cancellation. One tiny completion pays the cost, said out loud — except
    /// while recording, whose own status ("Listening — go ahead") must not be stomped by a
    /// background chore.
    /// </summary>
    private async Task WakeCleanupAsync()
    {
        if (_cleanupAwake || _languageModel is not { } model)
        {
            return;
        }

        Narrate($"Waking the cleanup model ({model.Description}) — the first load takes a while…");

        try
        {
            await model.CompleteAsync(
                    "Answer with the single word OK.", "OK?", maxTokens: 8, CancellationToken.None)
                .ConfigureAwait(true);

            _cleanupAwake = true;
            Narrate("Cleanup model awake.");
        }
        catch (Exception exception)
        {
            // Not fatal: cleanup already degrades per window. But say so, because "the
            // punctuation never improved" otherwise looks like the recording's fault.
            Narrate($"The cleanup model is not answering yet: {exception.Message}");
            LogError(exception);
        }

        void Narrate(string message)
        {
            if (!IsRecording && !IsPreparing)
            {
                Status = message;
            }
        }
    }

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
