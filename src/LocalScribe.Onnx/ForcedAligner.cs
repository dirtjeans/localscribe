using LocalScribe.Core.Alignment;
using LocalScribe.Core.Audio;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Transcription;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalScribe.Onnx;

/// <summary>
/// Times the words of a segment by aligning them against the audio.
/// <para>
/// Whisper times whole segments and its exported graphs emit no cross-attention, so the usual
/// way of recovering word times from Whisper is unavailable and they have been estimated from
/// loudness — good to about half a second. This measures them instead, to about the length of
/// one twenty-millisecond frame.
/// </para>
/// <para>
/// The model is a multilingual CTC recogniser, used here for the far easier job of alignment:
/// the words are already known, so all that is wanted is which frames go with which letters.
/// One model covers a thousand languages because it works in a romanised alphabet, which is why
/// <see cref="AlignmentAlphabet"/> folds every word down to plain Latin letters first.
/// </para>
/// <para>
/// Optional, like the speaker models. A machine without it transcribes exactly as before and
/// falls back to estimating.
/// </para>
/// </summary>
public sealed class ForcedAligner : IDisposable
{
    private InferenceSession? _session;
    private readonly AlignmentAlphabet _alphabet;
    private readonly string _input;

    private ForcedAligner(InferenceSession session, AlignmentAlphabet alphabet)
    {
        _session = session;
        _alphabet = alphabet;
        _input = session.InputMetadata.Keys.First();
    }

    /// <summary>What the aligner needs on disk, or null when it is not installed.</summary>
    public static string? Find(string modelRoot)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelRoot);

        var directory = Path.Combine(modelRoot, "alignment");

        return File.Exists(Path.Combine(directory, "vocab.json")) && ModelIn(directory) is not null
            ? directory
            : null;
    }

    /// <summary>Loads the aligner from a directory, or throws saying what is missing.</summary>
    public static ForcedAligner Load(string directory, ExecutionPlan? plan = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);

        var model = ModelIn(directory)
            ?? throw new FileNotFoundException($"No alignment model under {directory}.");

        var vocabulary = Path.Combine(directory, "vocab.json");
        if (!File.Exists(vocabulary))
        {
            throw new FileNotFoundException($"No alignment vocabulary at {vocabulary}.");
        }

        var options = new SessionOptions
        {
            // The same restraint the rest of the app runs under. Alignment is a second pass over
            // the audio and there is no version of this worth making the machine feel busy for.
            IntraOpNumThreads = plan?.CpuBudget.IntraOpThreads ?? 4,
            InterOpNumThreads = plan?.CpuBudget.InterOpThreads ?? 1,
        };

        return new ForcedAligner(new InferenceSession(model, options), AlignmentAlphabet.Load(vocabulary));
    }

    /// <summary>
    /// Runs the recogniser over the whole recording once and keeps what it made of every frame.
    /// <para>
    /// This is nearly all the cost of alignment and it depends on nothing but the audio, so it
    /// can run while the text is still being cleaned up and the speakers worked out. Placing
    /// words onto the frames afterwards is a Viterbi pass over a few hundred of them.
    /// </para>
    /// <para>
    /// Scored on a fixed grid rather than segment by segment. Where the segments will fall is
    /// not known this early, and cleanup and attribution would move them afterwards regardless.
    /// </para>
    /// </summary>
    /// <returns>The whole recording's scores, or null when it cannot be scored at all.</returns>
    public AlignmentScores? Scan(
        PcmAudio audio,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);

        if (audio.Samples.Length < ShortestAlignableSamples)
        {
            return null;
        }

        // How much audio a frame covers, asked rather than assumed, and asked twice.
        //
        // Dividing one window's samples by the frames it returned gives the wrong answer, and it
        // is wrong in a way that looks plausible: this model returns one frame per 320 samples
        // but needs 400 samples to return the first one, so a second of audio yields 49 frames
        // rather than 50 and the stride reads as 326. On a short recording that passes for right.
        // Over half an hour it is a two per cent drift, and inside each scored window it walks
        // the words along by half a second before starting over at the next one.
        //
        // The offset is constant, so the difference between two lengths cancels it exactly.
        var shortProbe = Math.Max(ShortestAlignableSamples, Math.Min(audio.Samples.Length, audio.SampleRate));
        var longProbe = Math.Min(audio.Samples.Length, shortProbe * 2);

        if (Score(audio.Samples.AsSpan(0, shortProbe)) is not { } near
            || Score(audio.Samples.AsSpan(0, longProbe)) is not { } far
            || far.Frames <= near.Frames)
        {
            return null;
        }

        var samplesPerFrame = (longProbe - shortProbe) / (far.Frames - near.Frames);
        var frames = samplesPerFrame > 0 ? audio.Samples.Length / samplesPerFrame : 0;

        if (frames <= 0)
        {
            return null;
        }

        var probe = far;

        var scores = new AlignmentScores(frames, probe.Alphabet, samplesPerFrame / (double)audio.SampleRate);

        // Windows are counted in frames so every boundary lands on one. Each is scored with a
        // margin of extra audio on both sides which is then thrown away: a window edge falling
        // mid-word would otherwise give the model half a word to recognise, and the frames either
        // side of the join are exactly the ones a listener would notice being wrong.
        var core = Math.Max(1, (int)(WindowSeconds * audio.SampleRate) / samplesPerFrame);
        var margin = (int)(MarginSeconds * audio.SampleRate) / samplesPerFrame;
        var shortest = (ShortestAlignableSamples + samplesPerFrame - 1) / samplesPerFrame;

        for (var at = 0; at < frames; at += core)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var until = Math.Min(frames, at + core);
            var readTo = Math.Min(frames, until + margin);
            var readFrom = Math.Max(0, at - margin);

            // The last window can be a sliver on its own. Reaching further back rather than
            // giving up keeps the tail of the recording timed.
            if (readTo - readFrom < shortest)
            {
                readFrom = Math.Max(0, readTo - shortest);
            }

            if (readTo - readFrom < shortest)
            {
                break;
            }

            if (Score(audio.Samples.AsSpan(readFrom * samplesPerFrame, (readTo - readFrom) * samplesPerFrame))
                is not { } window)
            {
                break;
            }

            // Only the middle is kept. The margins did their job by being there.
            var offset = at - readFrom;
            var rows = Math.Min(until - at, window.Frames - offset);

            if (rows > 0)
            {
                scores.Fill(at, window.Scores.AsSpan(offset * window.Alphabet, rows * window.Alphabet));
            }

            progress?.Report(until / (double)frames);
        }

        return scores;
    }

    /// <summary>
    /// Times every word of a segment against an existing scan, or returns null when it cannot.
    /// <para>
    /// Null rather than a guess: the caller has a perfectly good estimate to fall back on, and a
    /// bad alignment presented as a measurement is worse than an honest approximation.
    /// </para>
    /// </summary>
    /// <param name="allowWideRetry">
    /// Whether a placement that fails its grade may look again with a far wider window. Off for
    /// a segment already on trial for never having been spoken: extra audio is extra rope for
    /// invented text to fake a passing grade with, and on one archive it did.
    /// </param>
    public IReadOnlyList<WordTimings.Word>? Align(
        AlignmentScores scores,
        TranscriptSegment segment,
        bool allowWideRetry = true,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(scores);
        ArgumentNullException.ThrowIfNull(segment);

        var words = Words(segment.Text);
        if (words.Count == 0)
        {
            return null;
        }

        var (tokens, spellings) = _alphabet.Spell([.. words.Select(w => w.Text)]);
        if (tokens.Count == 0 || spellings.Count == 0)
        {
            return null;
        }

        cancellationToken.ThrowIfCancellationRequested();

        // Anchored to the segment's stated start, reaching back three seconds — and not chained
        // from where the previous segment's words ended, which was tried and measured. The two
        // schemes fail differently. Stamps are noisy but unbiased: each segment resets to the
        // audio, so the error never accumulates, and the median misplacement is zero in every
        // fifth of both test recordings. A chain is quiet but integrates: one late tail at a
        // window seam pushed everything after it, and the marker oscillated two to four seconds
        // out for the rest of the recording, with nothing ever pulling it back. Scattered
        // per-word noise loses to that on every count a listener cares about.
        //
        // Reaching back three seconds was once the thing that scrambled transcripts, and what
        // pardoned it is everything that changed since: segment order now comes from the decoder
        // and is never re-derived from placements, bounds are untangled after attribution, and a
        // word's speaker is judged on its last stretch rather than its whole span — so a first
        // word that pins a little early costs a slightly early marker on one word, not a
        // swallowed segment or a spliced paragraph.
        var duration = Math.Max(
            segment.EndSeconds - segment.StartSeconds, ShortestAlignableSeconds);

        // A narrow window first, and a wide one only on evidence. The stamps usually hold, but
        // their drift is a sawtooth — it grows through each of the transcriber's thirty-second
        // windows and resets at the seam, and on one recording it passed five seconds — so a
        // fixed reach fails inside the teeth, and a smoothed running correction chases the
        // resets and corrupts the segments after them; that was built, measured, and taken out.
        // What a drifted placement cannot do is read as itself: its words sit over some other
        // stretch of speech, and the decoded letters underneath do not resemble the text. So
        // every placement is graded, and only a failing grade pays for the wide window — with
        // its own risk of reaching into a neighbour — and then only if the wide attempt
        // actually grades better.
        var narrow = PlaceWindow(
            scores, segment, words, tokens, spellings,
            SearchBackSeconds, SearchForwardSeconds, duration);

        var grade = Grade(scores, segment, narrow);

        if (grade >= ReadsAsItself || !allowWideRetry)
        {
            return narrow;
        }

        var wide = PlaceWindow(
            scores, segment, words, tokens, spellings,
            DriftedReachSeconds, SearchForwardSeconds + 2, duration);

        return Grade(scores, segment, wide) > grade ? wide : narrow;
    }

    /// <summary>
    /// Places every segment's words in one pass over the whole recording.
    /// <para>
    /// This supersedes aligning segment by segment. A per-segment window is a local decision
    /// about a global constraint — text and time are jointly monotonic across the whole
    /// transcript — and every window shape tried failed somewhere: anchored windows could not
    /// recover stamps that drifted past their reach, chained ones integrated error, and any of
    /// them could lock a repeated phrase onto its twin. One path over everything makes those
    /// failures unrepresentable rather than tuned against.
    /// </para>
    /// <para>
    /// The stamps are demoted to what they are good for: centring the search corridor. Inside
    /// it, the audio decides everything.
    /// </para>
    /// </summary>
    /// <returns>One word list per segment, null where a segment could not be spelled at all.</returns>
    public IReadOnlyList<IReadOnlyList<WordTimings.Word>?> AlignAll(
        AlignmentScores scores,
        IReadOnlyList<TranscriptSegment> segments,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(scores);
        ArgumentNullException.ThrowIfNull(segments);

        var results = new IReadOnlyList<WordTimings.Word>?[segments.Count];

        var allTokens = new List<int>();
        var parts = new (List<WordTimings.Word> Words,
            IReadOnlyList<AlignmentAlphabet.Spelling> Spellings, int TokenStart)?[segments.Count];

        var limit = scores.SecondsAt(scores.Frames);
        var anchors = new List<(double Second, int Token)> { (0, 0) };

        // The drift accumulates: stamps describing more recording than exists mean every stamp
        // along the way is proportionally late, and the correction is one multiplication.
        var scale = CredibleAnchors.Scale(
            segments.Count == 0 ? 0 : segments.Max(s => s.EndSeconds), limit);

        for (var i = 0; i < segments.Count; i++)
        {
            var words = Words(segments[i].Text);

            if (words.Count == 0)
            {
                continue;
            }

            var (tokens, spellings) = _alphabet.Spell([.. words.Select(w => w.Text)]);

            if (tokens.Count == 0 || spellings.Count == 0)
            {
                continue;
            }

            var start = allTokens.Count;
            allTokens.AddRange(tokens);
            parts[i] = (words, spellings, start);

            var from = Math.Clamp(segments[i].StartSeconds * scale, 0, limit);

            anchors.Add((from, start));
            anchors.Add((Math.Clamp(segments[i].EndSeconds * scale, from, limit), start + tokens.Count));
        }

        if (allTokens.Count == 0)
        {
            return results;
        }

        anchors.Add((limit, allTokens.Count));

        // The stamps of the final padded window arrive up to eighteen seconds past the end of
        // the audio, and clamped to the end they become confident lies that collapse the tail
        // of the spine onto the last frames — far enough from the truth that the band cannot
        // reach back to it. Anchors that could not be spoken are dropped, and the spine
        // interpolates across the gap at the recording's own pace instead.
        anchors = CredibleAnchors.Prune(anchors, allTokens.Count, limit);

        // Wider than the worst lie the stamps have told. Fifteen seconds was "every drift ever
        // measured with an order of magnitude to spare" until a recording arrived whose stamps
        // ran seventeen and a half seconds late by its sixth minute — outside the band, so the
        // truth was unrepresentable and the pass parked the tail as close as the edge allowed,
        // which read as the marker falling a sentence behind. The corridor is here to bound the
        // compute, not to inform the placement; the audio decides inside it, so the only cost
        // of width is time, and doubling past the worst measurement is cheaper than meeting it
        // again. Thirty-six seconds of letters either side, never fewer than 900 states.
        var halfBand = Math.Max(
            900, (int)(allTokens.Count / Math.Max(1.0, limit) * 36) * 2);

        var placed = GlobalCtcAlignment.Align(
            scores,
            allTokens,
            _alphabet.Blank,
            Spine(scores, anchors, allTokens.Count),
            halfBand,
            cancellationToken);

        if (placed is null)
        {
            return results;
        }

        for (var i = 0; i < segments.Count; i++)
        {
            if (parts[i] is { } part)
            {
                results[i] = Assemble(scores, part.Words, part.Spellings, placed, part.TokenStart);
            }
        }

        return results;
    }

    /// <summary>
    /// The corridor's spine: for each frame, roughly which trellis state the stamps expect to
    /// be active. Monotone in both axes, because the corridor must never run backwards through
    /// the text — that guarantee is what kills the repeated-phrase double lock.
    /// </summary>
    private static int[] Spine(
        AlignmentScores scores, List<(double Second, int Token)> anchors, int tokens)
    {
        anchors.Sort((a, b) => a.Second.CompareTo(b.Second));

        var high = 0;

        for (var i = 0; i < anchors.Count; i++)
        {
            high = Math.Max(high, anchors[i].Token);
            anchors[i] = (anchors[i].Second, high);
        }

        var centers = new int[scores.Frames];
        var at = 0;

        for (var t = 0; t < scores.Frames; t++)
        {
            var second = scores.SecondsAt(t);

            while (at + 1 < anchors.Count && anchors[at + 1].Second <= second)
            {
                at++;
            }

            var (fromSecond, fromToken) = anchors[at];
            var (toSecond, toToken) = at + 1 < anchors.Count ? anchors[at + 1] : anchors[at];

            var share = toSecond > fromSecond
                ? (second - fromSecond) / (toSecond - fromSecond)
                : 0;

            centers[t] = Math.Clamp(
                (fromToken + (int)(share * (toToken - fromToken))) * 2, 0, tokens * 2);
        }

        return centers;
    }

    /// <summary>One segment's words, timed from the global placements.</summary>
    private static IReadOnlyList<WordTimings.Word> Assemble(
        AlignmentScores scores,
        List<WordTimings.Word> words,
        IReadOnlyList<AlignmentAlphabet.Spelling> spellings,
        IReadOnlyList<CtcForcedAlignment.Placement> placed,
        int tokenStart)
    {
        var spelled = spellings.ToDictionary(spelling => spelling.Index);
        var timed = new List<WordTimings.Word>(words.Count);
        var previous = scores.SecondsAt(placed[tokenStart].FirstFrame);

        for (var i = 0; i < words.Count; i++)
        {
            if (!spelled.TryGetValue(i, out var spelling))
            {
                timed.Add(new WordTimings.Word(words[i].Text, previous, previous)
                {
                    Offset = words[i].Offset,
                });

                continue;
            }

            var firstToken = tokenStart + spelling.First;
            var lastToken = firstToken + spelling.Count - 1;

            var from = scores.SecondsAt(placed[firstToken].FirstFrame);
            var to = scores.SecondsAt(placed[lastToken].LastFrame + 1);

            timed.Add(new WordTimings.Word(spelling.Word, from, Math.Max(to, from))
            {
                Offset = words[i].Offset,
            });

            previous = Math.Max(to, from);
        }

        return timed;
    }

    /// <summary>A placement below this does not read as its own text where it landed.</summary>
    private const double ReadsAsItself = 0.5;

    /// <summary>How far back a segment that failed its grade may look.</summary>
    private const double DriftedReachSeconds = 10;

    /// <summary>
    /// How much a placement's audio resembles the words placed on it, 0 to 1.
    /// <para>
    /// Judged by halves, taking the worse, because a whole-segment score is blind to a shift.
    /// A placement four seconds late still shares most of its text with the audio under it —
    /// the tail of the segment really is down there, one phrase along — and scored whole it
    /// passes while its first words sit on somebody else's speech. The first half of a shifted
    /// placement always fails, and the minimum is what lets that failure count.
    /// </para>
    /// </summary>
    private double Grade(
        AlignmentScores scores,
        TranscriptSegment segment,
        IReadOnlyList<WordTimings.Word>? placement)
    {
        var sounded = placement?.Where(w => w.EndSeconds > w.StartSeconds).ToList();

        if (sounded is not { Count: > 0 })
        {
            return 0;
        }

        if (sounded.Count < 6)
        {
            return TextLikeness.Share(
                segment.Text,
                Read(scores, sounded[0].StartSeconds, sounded[^1].EndSeconds));
        }

        var half = sounded.Count / 2;

        var first = TextLikeness.Share(
            string.Join(" ", sounded.Take(half).Select(w => w.Text)),
            Read(scores, sounded[0].StartSeconds, sounded[half - 1].EndSeconds));

        var second = TextLikeness.Share(
            string.Join(" ", sounded.Skip(half).Select(w => w.Text)),
            Read(scores, sounded[half].StartSeconds, sounded[^1].EndSeconds));

        return Math.Min(first, second);
    }

    private IReadOnlyList<WordTimings.Word>? PlaceWindow(
        AlignmentScores scores,
        TranscriptSegment segment,
        List<WordTimings.Word> words,
        IReadOnlyList<int> tokens,
        IReadOnlyList<AlignmentAlphabet.Spelling> spellings,
        double reachBack,
        double reachForward,
        double duration)
    {
        var first = scores.FrameAt(Math.Max(0, segment.StartSeconds - reachBack));
        var count = scores.FrameAt(
            Math.Max(segment.EndSeconds, segment.StartSeconds + duration)
                + reachForward) - first;

        if (count < ShortestAlignableSeconds / scores.FrameSeconds)
        {
            return null;
        }

        var placed = CtcForcedAlignment.Align(
            scores.Between(first, count), count, scores.Alphabet, tokens, _alphabet.Blank);

        if (placed is null)
        {
            return null;
        }

        var spelled = spellings.ToDictionary(spelling => spelling.Index);
        var timed = new List<WordTimings.Word>(words.Count);
        var previous = scores.SecondsAt(first);

        // Every word gets an entry, including the ones with no letters in them. A lone dash or
        // full stop cannot be aligned — there is no sound to align it to — but leaving it out
        // makes the returned list shorter than the segment's own words, and a caller pairing the
        // two positionally would then give every word after it its neighbour's time. On the
        // debate recording exactly one segment ended in a stray full stop, and that one segment
        // silently fell back to the estimate.
        for (var i = 0; i < words.Count; i++)
        {
            if (!spelled.TryGetValue(i, out var spelling))
            {
                timed.Add(new WordTimings.Word(words[i].Text, previous, previous)
                {
                    Offset = words[i].Offset,
                });

                continue;
            }

            var letters = placed.Skip(spelling.First).Take(spelling.Count).ToList();
            if (letters.Count == 0)
            {
                timed.Add(new WordTimings.Word(words[i].Text, previous, previous) { Offset = words[i].Offset });
                continue;
            }

            var to = scores.SecondsAt(first + letters[^1].LastFrame + 1);

            // Not reaching back further than a word can be. The search deliberately looks before
            // a segment's stated start, and the first word is where that room gets spent: on one
            // recording "Our top story this week…" began 2.5 seconds early, which put it ahead of
            // the "And I'm Ken Spencer-Brown." that was actually said first. The end of a word is
            // measured; the start of a long one is mostly whatever came before it.
            var from = Math.Max(
                scores.SecondsAt(first + letters[0].FirstFrame),
                to - WordTimings.LongestWordSeconds);

            timed.Add(new WordTimings.Word(spelling.Word, from, Math.Max(to, from))
            {
                Offset = words[i].Offset,
            });

            previous = Math.Max(to, from);
        }

        return timed.Count == 0 ? null : timed;
    }

    /// <summary>
    /// Reads a stretch of a scan back as letters, by taking the likeliest token in each frame.
    /// <para>
    /// Nothing in the app needs this: aligning already knows what the words are. It exists so a
    /// scan can be checked. If the letters coming out of the grid between two timestamps read
    /// like what was actually said then, the grid holds the right frames and maps onto the clock
    /// correctly — which is the part of this that no test without the model can reach.
    /// </para>
    /// <para>
    /// The letters are romanised and unpunctuated, because that is the alphabet the model works
    /// in. "sowhatdoyoumean" is a pass.
    /// </para>
    /// </summary>
    public string Read(AlignmentScores scores, double fromSeconds, double toSeconds)
    {
        ArgumentNullException.ThrowIfNull(scores);

        var first = scores.FrameAt(fromSeconds);
        var count = scores.FrameAt(toSeconds) - first;

        if (count <= 0)
        {
            return string.Empty;
        }

        var text = new System.Text.StringBuilder();
        var previous = -1;

        for (var frame = first; frame < first + count; frame++)
        {
            var best = Likeliest(scores, frame);

            // How CTC output is read: a repeated token is one letter held across several frames,
            // and a blank is what separates two of the same letter from one long one.
            if (best != previous && best != _alphabet.Blank)
            {
                text.Append(_alphabet.Letter(best));
            }

            previous = best;
        }

        return text.ToString();
    }

    /// <summary>Letters read out of a scan, and the frame each was heard in.</summary>
    /// <param name="Letters">The whole recording as romanised, unpunctuated letters.</param>
    /// <param name="Frames">Where each letter sits, one entry per letter.</param>
    public sealed record AudioReading(string Letters, IReadOnlyList<int> Frames);

    /// <summary>
    /// Reads the entire scan back as letters, remembering where each one was heard.
    /// <para>
    /// Searching this is how a word can be found in a recording without knowing roughly where it
    /// is already. Decoding a stretch at a time and sliding the window works only within a few
    /// seconds of the answer — which quietly means the badly-placed words, the ones most worth
    /// finding, are the ones it cannot find.
    /// </para>
    /// </summary>
    public AudioReading ReadAll(AlignmentScores scores)
    {
        ArgumentNullException.ThrowIfNull(scores);

        var letters = new System.Text.StringBuilder(scores.Frames / 4);
        var frames = new List<int>(scores.Frames / 4);
        var previous = -1;

        for (var frame = 0; frame < scores.Frames; frame++)
        {
            var best = Likeliest(scores, frame);

            if (best != previous && best != _alphabet.Blank)
            {
                letters.Append(_alphabet.Letter(best));
                frames.Add(frame);
            }

            previous = best;
        }

        return new AudioReading(letters.ToString(), frames);
    }

    /// <summary>The token the model thought likeliest in one frame.</summary>
    private static int Likeliest(AlignmentScores scores, int frame)
    {
        var row = scores.Between(frame, 1);
        var best = 0;

        for (var token = 1; token < scores.Alphabet; token++)
        {
            if (row[token] > row[best])
            {
                best = token;
            }
        }

        return best;
    }

    /// <summary>What the network made of one stretch of audio, as log probabilities.</summary>
    private (float[] Scores, int Frames, int Alphabet)? Score(ReadOnlySpan<float> samples)
    {
        var normalised = Normalise(samples);

        if (_session is not { } session)
        {
            return null;
        }

        using var outputs = session.Run(
        [
            NamedOnnxValue.CreateFromTensor(_input, new DenseTensor<float>(normalised, [1, normalised.Length])),
        ]);

        var logits = outputs.First().AsTensor<float>();
        var frames = logits.Dimensions[1];
        var alphabet = logits.Dimensions[2];

        return frames <= 0 || alphabet != _alphabet.Size
            ? null
            : (LogSoftmax(logits.ToArray(), frames, alphabet), frames, alphabet);
    }

    /// <summary>How much audio to score at a time.</summary>
    private const double WindowSeconds = 30;

    /// <summary>Extra audio scored on each side of a window and then discarded.</summary>
    private const double MarginSeconds = 2;

    /// <summary>Shorter than this and there is nothing to align a word to.</summary>
    private const double ShortestAlignableSeconds = 0.2;

    /// <summary>
    /// How far before a segment's stated start the words may actually turn out to be.
    /// <para>
    /// Generous, because this is the direction the error goes. Segment times come from the
    /// transcriber, which times whole segments and gets them late: time given to speech that was
    /// never said pushes everything after it back, and on the debate recording that left the
    /// second half of the transcript two to three seconds behind the voice.
    /// </para>
    /// <para>
    /// Deliberately not floored at where the previous segment ended, which was tried and was
    /// worse. A floor sounds like the obvious guard against two segments claiming the same
    /// speech, and it is a ratchet: one segment placed late forbids every later segment from
    /// reaching back past it, so a drift can be inherited but never corrected. With it in place,
    /// widening this made no difference at all — 37 of 66 words adrift at every width, because
    /// the room was never usable. Without it, 13.
    /// </para>
    /// <para>
    /// Segments are kept apart by the words themselves instead. Each is matched against the sound
    /// of its own text, and text that only occurs once has only one place it can go.
    /// </para>
    /// </summary>
    /// <para>
    /// Three rather than six. The sweep that chose this measured drift and not overlap, and six
    /// scored no better than three on drift while giving the first word of a segment six seconds
    /// of the previous speaker to stretch into. It took it: one recording had "Atheists" placed
    /// across 5.1 seconds reaching back over somebody else's sentence, which then attributed the
    /// word to them and dragged the segment's start time with it.
    /// </para>


    /// <summary>
    /// How far before a segment's stated start its words may really be. Segment stamps drift
    /// late routinely; three seconds covers every case measured, and the harms wide reach once
    /// caused are prevented downstream rather than here.
    /// </summary>
    private const double SearchBackSeconds = 3;

    /// <summary>
    /// How far past a segment's stated end the words may run.
    /// <para>
    /// Kept at a second, unlike the backward reach, because cutting it costs something the
    /// backward one did not: matched words more than doubled their failures, from 42 to 96, while
    /// drift stayed at zero. The last words of a segment genuinely need room to be found.
    /// </para>
    /// <para>
    /// The overlap it creates is dealt with where it does harm instead. Two segments sharing a
    /// second of the clock is only a problem for things that walk the transcript in order, so the
    /// bounds are tidied after the words are placed and the word times themselves are left alone.
    /// </para>
    /// </summary>
    private const double SearchForwardSeconds = 1;


    /// <summary>
    /// Zero mean and unit variance, which is what the model's own feature extractor does. It is
    /// declared in the model's preprocessor config and is not optional: the network was trained
    /// on normalised audio and a quiet recording fed in raw simply reads as silence.
    /// </summary>
    private static float[] Normalise(ReadOnlySpan<float> samples)
    {
        var mean = 0.0;
        foreach (var sample in samples)
        {
            mean += sample;
        }

        mean /= samples.Length;

        var variance = 0.0;
        foreach (var sample in samples)
        {
            variance += (sample - mean) * (sample - mean);
        }

        var deviation = Math.Sqrt((variance / samples.Length) + 1e-7);
        var scaled = new float[samples.Length];

        for (var i = 0; i < samples.Length; i++)
        {
            scaled[i] = (float)((samples[i] - mean) / deviation);
        }

        return scaled;
    }

    /// <summary>
    /// Turns the model's raw scores into log probabilities, one frame at a time.
    /// <para>
    /// Subtracting the largest score first is not a nicety: without it the exponentials overflow
    /// and every frame comes back as nothing at all.
    /// </para>
    /// </summary>
    private static float[] LogSoftmax(float[] logits, int frames, int alphabet)
    {
        var scores = new float[logits.Length];

        for (var t = 0; t < frames; t++)
        {
            var row = t * alphabet;

            var largest = float.NegativeInfinity;
            for (var k = 0; k < alphabet; k++)
            {
                if (logits[row + k] > largest)
                {
                    largest = logits[row + k];
                }
            }

            var total = 0.0;
            for (var k = 0; k < alphabet; k++)
            {
                total += Math.Exp(logits[row + k] - largest);
            }

            var offset = largest + Math.Log(total);
            for (var k = 0; k < alphabet; k++)
            {
                scores[row + k] = (float)(logits[row + k] - offset);
            }
        }

        return scores;
    }

    /// <summary>Words with where they start in the segment's text, for highlighting.</summary>
    private static List<WordTimings.Word> Words(string text)
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

    private static string? ModelIn(string directory)
    {
        if (!Directory.Exists(directory))
        {
            return null;
        }

        // Whichever export is present. The half-precision one is preferred: the quantised builds
        // use ConvInteger, which ONNX Runtime has no ARM64 implementation for, so a machine that
        // downloaded one would fail at load rather than run slowly.
        foreach (var name in new[] { "model_fp16.onnx", "model.onnx", "model_fp32.onnx" })
        {
            var path = Path.Combine(directory, name);
            if (File.Exists(path))
            {
                return path;
            }
        }

        return null;
    }

    /// <summary>Below about a fifth of a second there are too few frames to place anything.</summary>
    private const int ShortestAlignableSamples = 16000 / 5;

    /// <summary>
    /// Lets go of the network while keeping what is needed to place words.
    /// <para>
    /// Scanning needs the model; placing words on the frames it produced does not — that is a
    /// Viterbi pass over a table and a spelling. Six hundred megabytes is far too much to hold
    /// resident on the chance of a second attempt, and a scan is far too slow to repeat for one.
    /// Releasing the one and keeping the other makes a retry cost almost nothing.
    /// </para>
    /// </summary>
    public void ReleaseModel()
    {
        _session?.Dispose();
        _session = null;
    }

    /// <summary>True while the recording can still be scanned.</summary>
    public bool CanScan => _session is not null;

    public void Dispose() => ReleaseModel();
}
