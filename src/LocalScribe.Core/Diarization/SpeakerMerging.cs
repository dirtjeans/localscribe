namespace LocalScribe.Core.Diarization;

/// <summary>
/// Brings an end-to-end diarizer's answer down to the number of people the user says were in
/// the room.
/// <para>
/// Nemotron has no setting for a speaker count; it decides. When the user knows better and it
/// has found more people than there were, the extras are folded into whoever they sound most
/// like — smallest first, because the speaker heard for four seconds is far likelier to be a
/// cough or a voice on a bad line than the one heard for ten minutes is.
/// </para>
/// <para>
/// It never splits. If the model heard fewer people than the user names, nothing in the
/// activity says where the missing person is, and inventing a boundary would be worse than
/// reporting the shortfall.
/// </para>
/// </summary>
public static class SpeakerMerging
{
    /// <summary>The turns with speakers merged until at most <paramref name="wanted"/> remain.</summary>
    /// <param name="turns">Turns numbered from zero.</param>
    /// <param name="voices">
    /// A voice print per speaker where one could be taken. A speaker without one — too little
    /// clean speech to measure — goes to whoever talks nearest to them in time.
    /// </param>
    /// <param name="wanted">How many people there were.</param>
    public static IReadOnlyList<SpeakerTurn> ToCount(
        IReadOnlyList<SpeakerTurn> turns,
        IReadOnlyDictionary<int, float[]> voices,
        int wanted)
    {
        ArgumentNullException.ThrowIfNull(turns);
        ArgumentNullException.ThrowIfNull(voices);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(wanted);

        var owner = turns.Select(t => t.Speaker).Distinct().ToDictionary(s => s, s => s);
        var heard = turns.GroupBy(t => t.Speaker).ToDictionary(g => g.Key, g => g.Sum(t => t.DurationSeconds));
        var prints = voices.ToDictionary(pair => pair.Key, pair => pair.Value);

        int Root(int speaker) => owner[speaker] == speaker ? speaker : owner[speaker] = Root(owner[speaker]);

        while (heard.Count > wanted)
        {
            var smallest = heard.MinBy(pair => pair.Value).Key;
            var others = heard.Keys.Where(s => s != smallest).ToList();

            var into = prints.TryGetValue(smallest, out var print) && others.Any(prints.ContainsKey)
                ? others
                    .Where(prints.ContainsKey)
                    .MinBy(other => SpeakerClustering.CosineDistance(print, prints[other]))
                : NearestInTime(turns, smallest, others, Root);

            owner[smallest] = into;
            heard[into] += heard[smallest];
            heard.Remove(smallest);

            // The merged speaker's print stays the larger speaker's. Averaging in a voice that
            // was only a guess would blur the one print that was measured well.
            prints.Remove(smallest);
        }

        var merged = turns.Select(t => t with { Speaker = Root(t.Speaker) }).ToList();

        // Numbered by first appearance again, so a merge never leaves a gap in the labels.
        var order = merged
            .OrderBy(t => t.StartSeconds)
            .Select(t => t.Speaker)
            .Distinct()
            .Select((speaker, index) => (speaker, index))
            .ToDictionary(pair => pair.speaker, pair => pair.index);

        return [.. merged
            .Select(t => t with { Speaker = order[t.Speaker] })
            .OrderBy(t => t.StartSeconds)
            .ThenBy(t => t.Speaker)];
    }

    /// <summary>
    /// Where two different speakers talk at once. After a merge this replaces the model's own
    /// overlap, which would otherwise call one person talking over themselves crosstalk.
    /// </summary>
    public static IReadOnlyList<(double Start, double End)> Overlaps(IReadOnlyList<SpeakerTurn> turns)
    {
        ArgumentNullException.ThrowIfNull(turns);

        var spans = new List<(double Start, double End)>();

        for (var i = 0; i < turns.Count; i++)
        {
            for (var j = i + 1; j < turns.Count; j++)
            {
                if (turns[i].Speaker == turns[j].Speaker)
                {
                    continue;
                }

                var start = Math.Max(turns[i].StartSeconds, turns[j].StartSeconds);
                var end = Math.Min(turns[i].EndSeconds, turns[j].EndSeconds);

                if (end > start)
                {
                    spans.Add((start, end));
                }
            }
        }

        var joined = new List<(double Start, double End)>();

        foreach (var span in spans.OrderBy(s => s.Start))
        {
            if (joined.Count > 0 && span.Start <= joined[^1].End)
            {
                joined[^1] = (joined[^1].Start, Math.Max(joined[^1].End, span.End));
            }
            else
            {
                joined.Add(span);
            }
        }

        return joined;
    }

    private static int NearestInTime(
        IReadOnlyList<SpeakerTurn> turns,
        int speaker,
        IReadOnlyList<int> candidates,
        Func<int, int> root)
    {
        var own = turns.Where(t => root(t.Speaker) == speaker).ToList();
        var best = candidates[0];
        var closest = double.MaxValue;

        foreach (var turn in turns)
        {
            var who = root(turn.Speaker);
            if (!candidates.Contains(who))
            {
                continue;
            }

            foreach (var mine in own)
            {
                var gap = Math.Max(0, Math.Max(turn.StartSeconds - mine.EndSeconds, mine.StartSeconds - turn.EndSeconds));
                if (gap < closest)
                {
                    (closest, best) = (gap, who);
                }
            }
        }

        return best;
    }
}
