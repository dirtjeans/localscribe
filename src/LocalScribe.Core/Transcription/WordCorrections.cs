namespace LocalScribe.Core.Transcription;

/// <summary>
/// Words the user corrected, held apart from what the transcriber heard.
/// <para>
/// Written straight onto a segment's text, a correction would be lost twice over. Every window
/// that lands, every stretch the scan times, and the final assembly rebuild the transcript from
/// what was heard, so the fix would vanish at the next of those. And word times are paired with
/// a segment's words by value and by count, so "Ragu Nanda Kumara" becoming "Raghu Nandakumara"
/// would drop the whole segment from measured times back to an estimate. So the transcript stays
/// as heard underneath, the corrections are applied last, each time, and the words they replace
/// lend their times to the words that replace them.
/// </para>
/// <para>
/// Matched by whole words and by case: correcting "Will" must not touch "will". A correction
/// typed in lower case does also reach the capitalised form, since that is the same word at the
/// start of a sentence. Punctuation and a possessive stay where they were, so "Ragu's," becomes
/// "Raghu's,".
/// </para>
/// </summary>
public sealed class WordCorrections
{
    /// <summary>How far outside a segment's stamps a correction made "here" can still find its words.</summary>
    public const double HereReachSeconds = 3.0;

    private readonly List<Correction> _corrections = [];

    private sealed record Correction(string[] Wrong, string[] Right, double? AtSeconds);

    /// <summary>True when nothing has been corrected.</summary>
    public bool IsEmpty => _corrections.Count == 0;

    /// <summary>Corrects a phrase wherever it is said, including in text still to come.</summary>
    /// <returns>False when there is nothing to change.</returns>
    public bool Everywhere(string wrong, string right) => Add(wrong, right, null);

    /// <summary>Corrects the one occurrence nearest a moment in the recording.</summary>
    /// <returns>False when there is nothing to change.</returns>
    public bool Here(string wrong, string right, double seconds) => Add(wrong, right, seconds);

    /// <summary>Forgets every correction: a new recording, whose words are not these.</summary>
    public void Clear() => _corrections.Clear();

    private bool Add(string wrong, string right, double? at)
    {
        ArgumentNullException.ThrowIfNull(wrong);
        ArgumentNullException.ThrowIfNull(right);

        var wanted = Wanted(wrong);
        var replacement = Tokens(right);

        if (wanted.Length == 0 || replacement.Length == 0
            || string.Join(' ', wanted) == string.Join(' ', replacement))
        {
            return false;
        }

        _corrections.Add(new Correction(wanted, replacement, at));
        return true;
    }

    /// <summary>
    /// Applies every correction, in the order made, so correcting a correction works. Called
    /// with the transcript as heard, every time, which is why a correction that contains its
    /// own wrong word ("Ken" to "Ken Brown") cannot grow on each rebuild.
    /// </summary>
    public Corrected Apply(IReadOnlyList<TranscriptSegment> segments)
    {
        ArgumentNullException.ThrowIfNull(segments);

        if (IsEmpty)
        {
            return new Corrected(segments, new Dictionary<TranscriptSegment, Line>());
        }

        var lines = Lines(segments);

        foreach (var correction in _corrections)
        {
            if (correction.AtSeconds is { } at)
            {
                ReplaceNearest(lines, correction, at);
            }
            else
            {
                ReplaceAll(lines, correction);
            }
        }

        var shown = new List<TranscriptSegment>(lines.Count);
        var table = new Dictionary<TranscriptSegment, Line>();

        foreach (var line in lines)
        {
            if (!line.Changed)
            {
                shown.Add(line.Heard);
            }
            else if (line.Pieces.Count > 0)
            {
                // Empty only when every word it held went into a correction that began in the
                // segment before, and an empty row would read as a gap in the transcript.
                var segment = line.Heard with { Text = string.Join(' ', line.Pieces.Select(piece => piece.Text)) };
                shown.Add(segment);
                table[segment] = line;
            }
        }

        return new Corrected(shown, table);
    }

    /// <summary>How many places a correction made everywhere would change.</summary>
    public static int Occurrences(string wrong, IReadOnlyList<TranscriptSegment> segments)
    {
        ArgumentNullException.ThrowIfNull(wrong);
        ArgumentNullException.ThrowIfNull(segments);

        var wanted = Wanted(wrong);

        if (wanted.Length == 0)
        {
            return 0;
        }

        var lines = Lines(segments);
        var count = 0;

        for (var l = 0; l < lines.Count; l++)
        {
            var p = 0;

            while (p < lines[l].Pieces.Count)
            {
                if (Find(lines, l, p, wanted) is { } end)
                {
                    count++;

                    if (end.Line != l)
                    {
                        // The rest of this line belongs to the match.
                        break;
                    }

                    p = end.Piece;
                }
                else
                {
                    p++;
                }
            }
        }

        return count;
    }

    /// <summary>
    /// The whole words a selection touches, so dragging from the middle of a word corrects the
    /// word, without the punctuation around them, which stays put when the words change.
    /// </summary>
    /// <param name="text">The text the selection is in.</param>
    /// <param name="from">First selected character.</param>
    /// <param name="to">One past the last selected character; equal to <paramref name="from"/> for a point.</param>
    /// <returns>Where the words begin in the text and the words themselves, or null when it touches none.</returns>
    public static (int Start, string Words)? WordsAround(string text, int from, int to)
    {
        ArgumentNullException.ThrowIfNull(text);

        from = Math.Clamp(from, 0, text.Length);
        to = Math.Clamp(Math.Max(from, to), from, text.Length);

        // A point on a word, or the last character of a selection that stopped on its edge.
        if (from == to && from == text.Length && from > 0)
        {
            from--;
            to = from;
        }

        var start = from;

        while (start > 0 && !char.IsWhiteSpace(text[start - 1]))
        {
            start--;
        }

        var end = Math.Max(to, from + (from == to ? 1 : 0));
        end = Math.Min(end, text.Length);

        while (end < text.Length && !char.IsWhiteSpace(text[end]))
        {
            end++;
        }

        var tokens = text[start..end].Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

        if (tokens.Length == 0)
        {
            return null;
        }

        var first = Split(tokens[0]);
        var firstAt = start + (text[start..end].Length - text[start..end].TrimStart().Length) + first.Prefix.Length;
        var words = Wanted(string.Join(' ', tokens));

        return words.Length == 0 ? null : (firstAt, string.Join(' ', words));
    }

    private static string[] Wanted(string wrong)
    {
        var tokens = Tokens(wrong);
        var wanted = new string[tokens.Length];

        for (var i = 0; i < tokens.Length; i++)
        {
            wanted[i] = Split(tokens[i]).Core;
        }

        return wanted.All(word => word.Length == 0) ? [] : wanted;
    }

    private static string[] Tokens(string text) =>
        text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

    private static List<Line> Lines(IReadOnlyList<TranscriptSegment> segments) =>
        [.. segments.Select(segment => new Line(segment))];

    private static void ReplaceAll(List<Line> lines, Correction correction)
    {
        for (var l = 0; l < lines.Count; l++)
        {
            var p = 0;

            while (p < lines[l].Pieces.Count)
            {
                p = Find(lines, l, p, correction.Wrong) is { } end
                    ? Replace(lines, l, p, end, correction)
                    : p + 1;
            }
        }
    }

    private static void ReplaceNearest(List<Line> lines, Correction correction, double at)
    {
        (int Line, int Piece, Position End)? best = null;
        var bestDistance = double.MaxValue;

        for (var l = 0; l < lines.Count; l++)
        {
            var line = lines[l];

            if (at < line.Heard.StartSeconds - HereReachSeconds || at > line.Heard.EndSeconds + HereReachSeconds)
            {
                continue;
            }

            for (var p = 0; p < line.Pieces.Count; p++)
            {
                if (Find(lines, l, p, correction.Wrong) is not { } end)
                {
                    continue;
                }

                // Only to choose between occurrences a few seconds apart, so an even share of
                // the segment's span is plenty.
                var share = line.Pieces[p].From / Math.Max(1, line.HeardWords);
                var distance = Math.Abs(line.Heard.StartSeconds + (line.Heard.DurationSeconds * share) - at);

                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    best = (l, p, end);
                }
            }
        }

        if (best is { } found)
        {
            Replace(lines, found.Line, found.Piece, found.End, correction);
        }
    }

    private readonly record struct Position(int Line, int Piece);

    /// <summary>
    /// Where a match beginning here ends, or null. Runs across segments, because a name does not
    /// know where a segment was cut: the transcriber's boundaries are not the sentence's.
    /// </summary>
    private static Position? Find(List<Line> lines, int line, int piece, string[] wanted)
    {
        var l = line;
        var p = piece;

        foreach (var word in wanted)
        {
            while (l < lines.Count && p >= lines[l].Pieces.Count)
            {
                l++;
                p = 0;
            }

            if (l == lines.Count || !Same(word, Split(lines[l].Pieces[p].Text).Core))
            {
                return null;
            }

            p++;
        }

        return new Position(l, p);
    }

    private static bool Same(string wanted, string actual)
    {
        if (wanted == actual)
        {
            return true;
        }

        return wanted.Length == actual.Length
            && wanted.Length > 0
            && IsCapitalOf(actual[0], wanted[0])
            && string.CompareOrdinal(wanted, 1, actual, 1, wanted.Length - 1) == 0;
    }

    private static bool IsCapitalOf(char capital, char lower) =>
        char.IsLower(lower) && char.IsUpper(capital) && char.ToLowerInvariant(capital) == lower;

    /// <summary>Puts the correction in where the match began; returns where to carry on looking.</summary>
    private static int Replace(List<Line> lines, int line, int piece, Position end, Correction correction)
    {
        var pieces = lines[line].Pieces;
        var first = pieces[piece];
        var lastHere = end.Line == line ? pieces[end.Piece - 1] : pieces[^1];
        var last = lines[end.Line].Pieces[end.Piece - 1];

        var (prefix, core, _) = Split(first.Text);
        var (_, _, suffix) = Split(last.Text);
        var right = correction.Right;
        var count = right.Length;
        var inserted = new List<Piece>(count);

        for (var i = 0; i < count; i++)
        {
            var text = right[i];

            if (i == 0)
            {
                // The start of a sentence stays one when the wrong word was typed in lower case.
                if (core.Length > 0 && text.Length > 0 && IsCapitalOf(core[0], correction.Wrong[0][0]))
                {
                    text = char.ToUpperInvariant(text[0]) + text[1..];
                }

                text = text.StartsWith(prefix, StringComparison.Ordinal) ? text : prefix + text;
            }

            if (i == count - 1 && !text.EndsWith(suffix, StringComparison.Ordinal))
            {
                text += suffix;
            }

            // Shared evenly over the words replaced: their times are the only ones there are.
            var span = lastHere.To - first.From;
            inserted.Add(new Piece(text, first.From + (span * i / count), first.From + (span * (i + 1) / count)));
        }

        if (end.Line == line)
        {
            pieces.RemoveRange(piece, end.Piece - piece);
        }
        else
        {
            pieces.RemoveRange(piece, pieces.Count - piece);

            for (var l = line + 1; l < end.Line; l++)
            {
                lines[l].Pieces.Clear();
                lines[l].Changed = true;
            }

            lines[end.Line].Pieces.RemoveRange(0, end.Piece);
            lines[end.Line].Changed = true;
        }

        pieces.InsertRange(piece, inserted);
        lines[line].Changed = true;

        return piece + count;
    }

    /// <summary>A word's leading punctuation, the word, and what trails it, possessive included.</summary>
    private static (string Prefix, string Core, string Suffix) Split(string token)
    {
        var start = 0;

        while (start < token.Length && !char.IsLetterOrDigit(token[start]))
        {
            start++;
        }

        if (start == token.Length)
        {
            return (token, string.Empty, string.Empty);
        }

        var end = token.Length;

        while (!char.IsLetterOrDigit(token[end - 1]))
        {
            end--;
        }

        if (end - start > 2 && token[end - 1] is 's' or 'S' && token[end - 2] is '\'' or '’')
        {
            end -= 2;
        }

        return (token[..start], token[start..end], token[end..]);
    }

    /// <summary>A word as shown, and the stretch of the heard words whose times it borrows.</summary>
    internal readonly record struct Piece(string Text, double From, double To);

    /// <summary>A segment as heard, and what corrections have made of it so far.</summary>
    internal sealed class Line
    {
        public Line(TranscriptSegment heard)
        {
            Heard = heard;

            var tokens = Tokens(heard.Text);
            HeardWords = tokens.Length;
            Pieces = [.. tokens.Select((token, i) => new Piece(token, i, i + 1))];
        }

        /// <summary>The segment before any correction.</summary>
        public TranscriptSegment Heard { get; }

        internal int HeardWords { get; }

        internal List<Piece> Pieces { get; }

        internal bool Changed { get; set; }

        /// <summary>The words as corrected, carrying times taken from the words as heard.</summary>
        internal IReadOnlyList<WordTimings.Word> Retimed(IReadOnlyList<WordTimings.Word> heard)
        {
            var words = new List<WordTimings.Word>(Pieces.Count);
            var offset = 0;

            foreach (var piece in Pieces)
            {
                var start = StartAt(heard, piece.From);
                var end = EndAt(heard, piece.To);

                words.Add(new WordTimings.Word(piece.Text, start, Math.Max(start, end)) { Offset = offset });
                offset += piece.Text.Length + 1;
            }

            return words;
        }

        private double StartAt(IReadOnlyList<WordTimings.Word> heard, double position)
        {
            if (heard.Count != HeardWords)
            {
                return Even(position);
            }

            var word = heard[Math.Min((int)position, heard.Count - 1)];
            return word.StartSeconds + ((position - (int)position) * (word.EndSeconds - word.StartSeconds));
        }

        private double EndAt(IReadOnlyList<WordTimings.Word> heard, double position)
        {
            if (heard.Count != HeardWords)
            {
                return Even(position);
            }

            var index = Math.Clamp((int)Math.Ceiling(position) - 1, 0, heard.Count - 1);
            var word = heard[index];
            return word.StartSeconds + ((position - index) * (word.EndSeconds - word.StartSeconds));
        }

        // Words and times that do not correspond have nothing to lend; the span is all there is.
        private double Even(double position) =>
            Heard.StartSeconds + (Heard.DurationSeconds * position / Math.Max(1, HeardWords));
    }

    /// <summary>A transcript with the corrections applied, and the way back to what was heard.</summary>
    public sealed class Corrected
    {
        private readonly IReadOnlyDictionary<TranscriptSegment, Line> _lines;

        internal Corrected(IReadOnlyList<TranscriptSegment> shown, IReadOnlyDictionary<TranscriptSegment, Line> lines)
        {
            Shown = shown;
            _lines = lines;
        }

        /// <summary>A transcript nobody has corrected, before there is one.</summary>
        public static Corrected None { get; } = new([], new Dictionary<TranscriptSegment, Line>());

        /// <summary>The segments as the reader sees, saves and exports them.</summary>
        public IReadOnlyList<TranscriptSegment> Shown { get; }

        /// <summary>The segment word times are kept against, given the one on screen.</summary>
        public TranscriptSegment HeardOf(TranscriptSegment shown) =>
            _lines.TryGetValue(shown, out var line) ? line.Heard : shown;

        /// <summary>A shown segment's words, timed from the words of the segment it was heard as.</summary>
        public IReadOnlyList<WordTimings.Word> Retimed(TranscriptSegment shown, IReadOnlyList<WordTimings.Word> heard) =>
            _lines.TryGetValue(shown, out var line) ? line.Retimed(heard) : heard;
    }
}
