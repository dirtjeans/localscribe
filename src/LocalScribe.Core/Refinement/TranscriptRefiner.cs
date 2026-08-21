using System.Text;
using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Refinement;

/// <summary>Optional extras the cleanup pass can produce beyond a tidied transcript.</summary>
[Flags]
public enum RefinementOutputs
{
    None = 0,

    /// <summary>Repair punctuation, casing, and sentence breaks.</summary>
    Punctuation = 1,

    /// <summary>Correct names and domain terms against a supplied glossary.</summary>
    Glossary = 2,

    /// <summary>Write a short summary of the recording.</summary>
    Summary = 4,

    /// <summary>Pull out action items.</summary>
    ActionItems = 8,

    Default = Punctuation | Glossary,
    Everything = Punctuation | Glossary | Summary | ActionItems,
}

/// <param name="CleanedText">The transcript after punctuation and glossary repair.</param>
/// <param name="Summary">Present only when <see cref="RefinementOutputs.Summary"/> was requested.</param>
/// <param name="ActionItems">Present only when <see cref="RefinementOutputs.ActionItems"/> was requested.</param>
/// <param name="CleanedSegments">
/// The same text, back in the segments it came from, with every timing intact.
/// <para>
/// The form the app actually needs, and for a long time the one it could not have. Cleanup
/// returned a flat string; the transcript view is built from timed segments, because that is
/// what click-to-play seeks with and what speaker labels attach to. With no way to get the
/// string back into the segments, the cleaned text was computed and then dropped — every
/// transcript shown, saved, and copied was the raw one, which is why parts of them were never
/// punctuated no matter which model ran.
/// </para>
/// </param>
public sealed record RefinementResult(
    string CleanedText,
    string? Summary = null,
    IReadOnlyList<string>? ActionItems = null,
    IReadOnlyList<TranscriptSegment>? CleanedSegments = null);

/// <summary>
/// Runs the transcript through a local language model to fix what Whisper reliably gets wrong:
/// punctuation, casing, and any name or acronym specific to the speaker's world.
/// <para>
/// The glossary is the part that earns its keep. Whisper has no idea how your colleagues or
/// your product are spelled, and no amount of a larger model fixes that. A short list of terms
/// fed to the cleanup model does.
/// </para>
/// </summary>
public sealed class TranscriptRefiner
{
    /// <summary>
    /// Cleanup runs a window at a time so a long recording does not overflow a small model's
    /// context. Roughly 400 words keeps well clear of a 2k-token limit.
    /// </summary>
    public const int WordsPerCleanupWindow = 400;

    private readonly ILanguageModel _model;

    public TranscriptRefiner(ILanguageModel model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <param name="progress">
    /// Fraction of the cleanup done, from 0 to 1. Worth reporting because this runs a language
    /// model over the whole transcript a window at a time, which on a long recording is minutes
    /// during which nothing else says anything is happening.
    /// </param>
    public async Task<RefinementResult> RefineAsync(
        Transcript transcript,
        IReadOnlyList<string>? glossary = null,
        RefinementOutputs outputs = RefinementOutputs.Default,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(transcript);

        var rawText = transcript.FullText;
        if (rawText.Length == 0 || outputs == RefinementOutputs.None)
        {
            return new RefinementResult(rawText, CleanedSegments: transcript.Segments);
        }

        var wantsCleanup = outputs.HasFlag(RefinementOutputs.Punctuation)
            || outputs.HasFlag(RefinementOutputs.Glossary);

        // Windows are groups of whole segments rather than slices of a string, so that what
        // comes back has somewhere to go. Every call to the model is one step, counted up front
        // so the fraction means something from the first one rather than jumping about as the
        // work reveals itself.
        var windows = wantsCleanup
            ? SplitSegmentsIntoWindows(transcript.Segments, WordsPerCleanupWindow)
            : [];

        var totalSteps = Math.Max(1,
            windows.Count
            + (outputs.HasFlag(RefinementOutputs.Summary) ? 1 : 0)
            + (outputs.HasFlag(RefinementOutputs.ActionItems) ? 1 : 0));

        var completed = 0;
        void Step() => progress?.Report(Math.Min(1.0, ++completed / (double)totalSteps));

        var cleanedSegments = windows.Count > 0
            ? await CleanAsync(windows, glossary, outputs, Step, cancellationToken).ConfigureAwait(false)
            : transcript.Segments;

        var cleaned = string.Join(
            " ",
            cleanedSegments.Select(segment => segment.Text.Trim()).Where(text => text.Length > 0));

        string? summary = null;
        if (outputs.HasFlag(RefinementOutputs.Summary))
        {
            summary = (await _model.CompleteAsync(
                SummarySystemPrompt,
                cleaned,
                maxTokens: 512,
                cancellationToken).ConfigureAwait(false)).Trim();

            Step();
        }

        IReadOnlyList<string>? actionItems = null;
        if (outputs.HasFlag(RefinementOutputs.ActionItems))
        {
            var raw = await _model.CompleteAsync(
                ActionItemsSystemPrompt,
                cleaned,
                maxTokens: 512,
                cancellationToken).ConfigureAwait(false);
            actionItems = ParseActionItems(raw);
        }

        return new RefinementResult(cleaned, summary, actionItems, cleanedSegments);
    }

    /// <summary>
    /// How many windows came back unfaithful and were kept raw. Worth surfacing: a handful is
    /// the model being a small model, and most of them means the backend is a poor fit for the
    /// job rather than that the recording was difficult.
    /// </summary>
    public int Rejected { get; private set; }

    private async Task<IReadOnlyList<TranscriptSegment>> CleanAsync(
        IReadOnlyList<IReadOnlyList<TranscriptSegment>> windows,
        IReadOnlyList<string>? glossary,
        RefinementOutputs outputs,
        Action onWindowDone,
        CancellationToken cancellationToken)
    {
        var systemPrompt = BuildCleanupSystemPrompt(glossary, outputs);
        var result = new List<TranscriptSegment>();

        foreach (var window in windows)
        {
            var spoken = TextOf(window);

            var cleaned = (await _model.CompleteAsync(
                systemPrompt,
                spoken,
                maxTokens: EstimateReplyTokens(spoken),
                cancellationToken).ConfigureAwait(false)).Trim();

            // Checked, not trusted. The instructions already say to keep every word and add no
            // notes, and small models disregard both often enough that a transcript cleaned
            // without verification is a transcript that has quietly lost sentences. A window
            // that fails goes through unchanged.
            if (TranscriptQuality.IsFaithfulCleanup(spoken, cleaned))
            {
                result.AddRange(CleanedTextAlignment.Apply(window, cleaned));
            }
            else
            {
                Rejected++;
                result.AddRange(window);
            }

            onWindowDone();
        }

        return result;
    }

    /// <summary>The words of a window, as the model should see them.</summary>
    private static string TextOf(IReadOnlyList<TranscriptSegment> window) =>
        string.Join(" ", window.Select(segment => segment.Text.Trim()).Where(text => text.Length > 0));

    /// <summary>
    /// Groups consecutive segments into windows of roughly <paramref name="wordsPerWindow"/>
    /// words.
    /// <para>
    /// Whole segments, never split, because a segment is the unit that carries a timing and half
    /// of one has nowhere to put its words back. A window runs a little over rather than cutting
    /// a segment in two.
    /// </para>
    /// </summary>
    internal static IReadOnlyList<IReadOnlyList<TranscriptSegment>> SplitSegmentsIntoWindows(
        IReadOnlyList<TranscriptSegment> segments,
        int wordsPerWindow)
    {
        var windows = new List<IReadOnlyList<TranscriptSegment>>();
        var current = new List<TranscriptSegment>();
        var words = 0;

        foreach (var segment in segments)
        {
            var count = segment.Text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length;

            if (words > 0 && words + count > wordsPerWindow)
            {
                windows.Add(current);
                current = [];
                words = 0;
            }

            current.Add(segment);
            words += count;
        }

        if (current.Count > 0)
        {
            windows.Add(current);
        }

        return windows;
    }

    /// <summary>
    /// Builds the cleanup instructions. The emphasis on not rewording is load-bearing: small
    /// models will happily "improve" a transcript into something the speaker never said, and
    /// a transcript that reads well but is wrong is worse than a scruffy accurate one.
    /// </summary>
    internal static string BuildCleanupSystemPrompt(IReadOnlyList<string>? glossary, RefinementOutputs outputs)
    {
        var builder = new StringBuilder();
        builder.AppendLine("You are cleaning up a raw speech-to-text transcript.");
        builder.AppendLine();
        builder.AppendLine("Rules:");
        builder.AppendLine("- Keep every word the speaker said. Do not reword, shorten, or improve their phrasing.");
        builder.AppendLine("- Do not add facts, opinions, or commentary of your own.");

        if (outputs.HasFlag(RefinementOutputs.Punctuation))
        {
            builder.AppendLine("- Add punctuation, capitalisation, and paragraph breaks where they belong.");
            builder.AppendLine("- Remove stutters and repeated filler words, but keep genuine repetition.");
        }

        if (outputs.HasFlag(RefinementOutputs.Glossary) && glossary is { Count: > 0 })
        {
            builder.AppendLine("- Correct misheard names and technical terms to match this list exactly:");
            foreach (var term in glossary)
            {
                builder.AppendLine($"    {term}");
            }

            builder.AppendLine("- Only apply a correction when the transcript clearly meant that term.");
        }

        builder.AppendLine();
        builder.AppendLine("Reply with the corrected transcript and nothing else. No preamble, no notes.");
        return builder.ToString();
    }

    private const string SummarySystemPrompt =
        """
        Summarise this transcript in three to five sentences.

        Cover what was discussed and what was decided. Use only what the transcript says;
        if something is unclear, leave it out rather than guessing.

        Reply with the summary and nothing else.
        """;

    private const string ActionItemsSystemPrompt =
        """
        List the action items in this transcript.

        Put one per line, starting each line with "- ". Name who owns the item when the
        transcript says. If there are no action items, reply with exactly: NONE

        Reply with the list and nothing else.
        """;


    /// <summary>
    /// Cleanup output is about the same length as its input, plus headroom for the punctuation
    /// being added. Four tokens per three words is a safe overestimate for English.
    /// </summary>
    private static int EstimateReplyTokens(string window)
    {
        var words = window.Count(c => c == ' ') + 1;
        return Math.Clamp((int)(words * 2.0), 256, 4096);
    }

    /// <summary>Parses the action-item reply, tolerating the bullet styles small models drift between.</summary>
    internal static IReadOnlyList<string> ParseActionItems(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw) || raw.Trim().Equals("NONE", StringComparison.OrdinalIgnoreCase))
        {
            return [];
        }

        var items = new List<string>();
        foreach (var line in raw.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var trimmed = line.TrimStart('-', '*', '•', ' ', '\t');

            // Numbered lists show up often enough to be worth handling.
            var dotIndex = trimmed.IndexOf('.');
            if (dotIndex is > 0 and <= 2 && trimmed[..dotIndex].All(char.IsDigit))
            {
                trimmed = trimmed[(dotIndex + 1)..].TrimStart();
            }

            if (trimmed.Length > 0 && !trimmed.Equals("NONE", StringComparison.OrdinalIgnoreCase))
            {
                items.Add(trimmed);
            }
        }

        return items;
    }
}
