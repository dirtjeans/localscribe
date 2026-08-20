namespace LocalScribe.Core.Transcription;

/// <summary>
/// Joins the per-window results back into one transcript.
/// <para>
/// Because the chunker overlaps its windows, the same words are transcribed twice at every
/// boundary. Dropping duplicates by timestamp alone does not work: the two passes rarely agree
/// on exact timings, and Whisper often shifts a word by a few hundred milliseconds between
/// windows. So this matches on the words themselves and uses timing only to decide where to
/// look for a match.
/// </para>
/// </summary>
public sealed class TranscriptStitcher
{
    private readonly double _boundaryToleranceSeconds;

    /// <param name="boundaryToleranceSeconds">
    /// How far either side of a chunk boundary a duplicate might have drifted. Should be at
    /// least as large as the chunker's overlap.
    /// </param>
    public TranscriptStitcher(double boundaryToleranceSeconds = 2.5)
    {
        _boundaryToleranceSeconds = boundaryToleranceSeconds;
    }

    /// <summary>
    /// Merges segments from overlapping windows into one ordered, duplicate-free transcript.
    /// Segments are expected in chunk order; within a chunk they must already be time-ordered.
    /// </summary>
    public IReadOnlyList<TranscriptSegment> Stitch(IEnumerable<IReadOnlyList<TranscriptSegment>> chunkResults)
    {
        ArgumentNullException.ThrowIfNull(chunkResults);

        var merged = new List<TranscriptSegment>();

        foreach (var chunk in chunkResults)
        {
            foreach (var segment in chunk)
            {
                if (segment.LooksHallucinated || string.IsNullOrWhiteSpace(segment.Text))
                {
                    continue;
                }

                if (IsDuplicateOfRecent(merged, segment))
                {
                    continue;
                }

                merged.Add(segment);
            }
        }

        merged.Sort((left, right) => left.StartSeconds.CompareTo(right.StartSeconds));
        return merged;
    }

    /// <summary>
    /// Looks back over the segments near this one and reports whether the same words already
    /// landed. Only recent segments are considered, so a genuinely repeated phrase later in the
    /// recording survives.
    /// </summary>
    private bool IsDuplicateOfRecent(List<TranscriptSegment> merged, TranscriptSegment candidate)
    {
        var normalisedCandidate = Normalise(candidate.Text);
        if (normalisedCandidate.Length == 0)
        {
            return false;
        }

        for (var i = merged.Count - 1; i >= 0; i--)
        {
            var existing = merged[i];

            // Segments are appended in roughly increasing time order, so once we are well
            // behind the candidate there is nothing left to compare against.
            if (candidate.StartSeconds - existing.EndSeconds > _boundaryToleranceSeconds)
            {
                break;
            }

            if (Normalise(existing.Text) == normalisedCandidate)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Strips punctuation, casing, and spacing so that "Right, so —" and "right so" compare
    /// equal. The two passes over an overlap often differ by exactly this much.
    /// </summary>
    internal static string Normalise(string text)
    {
        Span<char> buffer = text.Length <= 256 ? stackalloc char[text.Length] : new char[text.Length];
        var length = 0;
        var lastWasSpace = true;

        foreach (var character in text)
        {
            if (char.IsLetterOrDigit(character))
            {
                buffer[length++] = char.ToLowerInvariant(character);
                lastWasSpace = false;
            }
            else if (!lastWasSpace)
            {
                buffer[length++] = ' ';
                lastWasSpace = true;
            }
        }

        while (length > 0 && buffer[length - 1] == ' ')
        {
            length--;
        }

        return new string(buffer[..length]);
    }
}
