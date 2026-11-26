namespace LocalScribe.Core.Audio;

/// <summary>What decided the order of a set of recordings.</summary>
public enum OrderBasis
{
    /// <summary>When each file says recording began.</summary>
    RecordedTimes,

    /// <summary>Track or part numbers written into the files.</summary>
    TrackNumbers,

    /// <summary>Numbers or part letters in the file names: part 2 before part 10, B after A.</summary>
    FileNames,

    /// <summary>When each file was last saved, the one clue every file has.</summary>
    FileTimes,
}

/// <summary>
/// Puts several recordings in the order they were made, so they can be transcribed as one.
/// <para>
/// The strongest clue every file shares decides: what the files say about when recording began,
/// then part numbers written into them, then numbers in their names, then when each was last
/// saved. "Every file" because a clue only some of them carry cannot place the rest, and
/// "distinct" because two files claiming the same moment say nothing about which came first.
/// A file's own recording time comes first because it is the only one set when the sound was
/// captured: names are whatever someone typed, and copying a file can change when it was saved.
/// </para>
/// <para>
/// The answer is a proposal, not a verdict. Whatever decided it is said in words, and where
/// another clue points another way, that is said too, so the person who made the recordings
/// checks the order before anything is transcribed.
/// </para>
/// </summary>
public static class RecordingOrder
{
    /// <summary>How far one file may begin before the last one ended and still follow it.</summary>
    public const double OverlapToleranceSeconds = 2;

    /// <summary>A silence longer than this between parts is worth asking about.</summary>
    public const double LongGapSeconds = 30 * 60;

    /// <summary>The order, why, and anything that argues against it.</summary>
    /// <param name="Ordered">The files, first to last.</param>
    /// <param name="Basis">The clue that decided.</param>
    /// <param name="Reason">That clue, in a sentence.</param>
    /// <param name="Doubts">Anything that should make a person look twice.</param>
    /// <param name="GapsBefore">
    /// For each file, the silence between the end of the one before and its start, in seconds,
    /// where both are known; negative when it starts before the other ended. Null for the first.
    /// </param>
    /// <param name="EndStamped">
    /// The paths of the files whose recorded time is read as when recording stopped rather than
    /// started; see <see cref="StartOf"/>.
    /// </param>
    public sealed record Proposal(
        IReadOnlyList<RecordingFacts> Ordered,
        OrderBasis Basis,
        string Reason,
        IReadOnlyList<string> Doubts,
        IReadOnlyList<double?> GapsBefore,
        IReadOnlySet<string> EndStamped)
    {
        /// <summary>True when any file's recorded time is read as when it stopped.</summary>
        public bool StampsMarkEnds => EndStamped.Count > 0;

        /// <summary>When a file's recording began, read the way this proposal reads its stamp.</summary>
        public DateTimeOffset? StartOf(RecordingFacts file) => RecordingOrder.StartOf(file, EndStamped);
    }

    /// <summary>Proposes an order for these recordings.</summary>
    public static Proposal Decide(IReadOnlyList<RecordingFacts> files)
    {
        ArgumentNullException.ThrowIfNull(files);

        var timed = files.Where(f => f.RecordedAt is not null).ToList();
        var ends = EndStamped(timed);
        var byTime = Distinct(files, f => StartOf(f, ends));
        var byTrack = Distinct(files, f => f.Track);
        // Sorted as the names were judged — words and numbers, not punctuation — so a clip whose
        // underscores became spaces in a copy still sorts by its time.
        var byName = NamesCarryOrder(files)
            ? [.. files.OrderBy(f => string.Join(' ', Words(f.Name)), NaturalOrder.Instance).ThenBy(f => f.Name, StringComparer.Ordinal)]
            : null as List<RecordingFacts>;

        var (ordered, basis, reason) =
            byTime is not null ? (byTime, OrderBasis.RecordedTimes,
                ends.Count == 0 ? "Ordered by when each file says it was recorded."
                : ends.Count == timed.Count ? "Ordered by the times stored in the files, which mark when each recording stopped."
                : "Ordered by when each file says it was recorded; for some, that is when recording stopped.")
            : byTrack is not null ? (byTrack, OrderBasis.TrackNumbers, "Ordered by the part numbers written into the files.")
            : byName is not null ? (byName, OrderBasis.FileNames, "Ordered by the file names.")
            : (Saved(files), OrderBasis.FileTimes, "Ordered by when each file was last saved.");

        var doubts = new List<string>();

        if (basis == OrderBasis.FileTimes)
        {
            doubts.Add(
                timed.Count == 0 ? "Nothing in these files says when they were recorded, and copying a file can change "
                    + "when it was saved. Check the order."
                : timed.Count == files.Count ? "Some of these say they were recorded at the same moment, so that cannot "
                    + "order them, and copying a file can change when it was saved. Check the order."
                : "Only some of these say when they were recorded, and copying a file can change when it was saved. "
                    + "Check the order.");
        }

        if (basis != OrderBasis.FileNames && byName is not null && !Same(byName, ordered))
        {
            doubts.Add("The file names suggest a different order.");
        }

        if (basis == OrderBasis.RecordedTimes && byTrack is not null && !Same(byTrack, ordered))
        {
            doubts.Add("The part numbers in the files suggest a different order.");
        }

        // Names decide over save times, but a copy usually keeps when a file was saved: where
        // two files' save times plainly disagree with the names, one of the two is wrong. Ties
        // say nothing either way, so only strict disagreements count.
        if (basis == OrderBasis.FileNames && SavedDisagrees(ordered))
        {
            doubts.Add("When the files were saved suggests a different order.");
        }

        // Only where recording times decided: from a mixture of clues, a gap between the two
        // files that happen to carry a time measures nothing about the order shown.
        var gaps = basis == OrderBasis.RecordedTimes ? Gaps(ordered, ends) : [.. ordered.Select(_ => (double?)null)];

        // Overlap is judged on the files' own times, among those that carry one, whatever
        // decided the order: two microphones on one session start at the same second, which
        // ties the times and hands the order to the names, and joining them would play the
        // conversation twice. A third file with no time does not excuse the other two.
        var sameMoment = timed.Select(f => f.RecordedAt).Distinct().Count() < timed.Count;

        if (sameMoment || Overlaps(timed, ends))
        {
            // Stamps and lengths alone cannot tell clips stamped as they stopped from a second
            // microphone that started later and stopped sooner: both overlap read one way and
            // follow on read the other. Without a name to say which, both are put to the reader.
            var allEnds = timed.Select(f => f.Path).ToHashSet(StringComparer.Ordinal);
            var couldFollowOn = !sameMoment && ends.Count == 0
                && timed.All(f => Length(f) is not null) && !Overlaps(timed, allEnds);

            doubts.Add(couldFollowOn
                ? "These overlap if the times stored in them are when each began. Some devices store when "
                    + "recording stopped instead, and then they follow on. Check they belong together."
                : "Some of these overlap in time, so they may be separate recordings of the same thing "
                    + "rather than parts of one.");
        }

        if (gaps.Any(gap => gap > LongGapSeconds))
        {
            doubts.Add("More than half an hour passed between some of these. Check they belong together.");
        }

        return new Proposal(ordered, basis, reason, doubts, gaps, ends);
    }

    /// <summary>How close a name's time must come to a stamp to vouch for it.</summary>
    private const double NameAgreesSeconds = 3;

    /// <summary>
    /// The files whose recorded time marks when recording stopped rather than started.
    /// <para>
    /// The standards say a file's creation time; what a writer puts there is when it created the
    /// file, and that is not always when the sound began. A Pixel writes its videos' movie header
    /// as it closes them: a clip named 18:06:04 that runs 7:44 says 18:13:48. Read as starts,
    /// consecutive clips like that overlap by most of their length, and nine clips of one meeting
    /// raised a warning that they were two recordings of the same thing.
    /// </para>
    /// <para>
    /// Stamps and lengths alone cannot settle it: a second microphone that started later and
    /// stopped sooner looks exactly the same, overlapping one way and following on the other, and
    /// reading its stamps as ends hid a real overlap. So a stamp is read as an end only where the
    /// file's own name vouches for it — a time in the name that is the stamp less the length, and
    /// not the stamp itself — and file by file, since a phone's clips and a recorder's files can
    /// be joined together. Names often carry local time where the stamp is UTC, so the two may
    /// differ by a whole time zone.
    /// </para>
    /// </summary>
    private static HashSet<string> EndStamped(IReadOnlyList<RecordingFacts> timed)
    {
        var ends = new HashSet<string>(StringComparer.Ordinal);

        foreach (var file in timed)
        {
            if (file.RecordedAt is not { } stamp || Length(file) is not { } length || NameTime(file.Name) is not { } named)
            {
                continue;
            }

            var stopped = stamp.UtcDateTime;
            var began = stopped - TimeSpan.FromSeconds(length);

            if (SameMoment(named, began) && !SameMoment(named, stopped))
            {
                ends.Add(file.Path);
            }
        }

        return ends;
    }

    /// <summary>Whether a name's time and a stamp agree, allowing for the name being in any time zone.</summary>
    private static bool SameMoment(DateTime named, DateTime utc)
    {
        var apart = (named - utc).TotalSeconds;
        var zone = Math.Round(apart / 900) * 900;

        return Math.Abs(zone) <= 14 * 3600 && Math.Abs(apart - zone) <= NameAgreesSeconds;
    }

    /// <summary>The date and time a name is stamped with, year first, or null.</summary>
    private static DateTime? NameTime(string name)
    {
        var words = Words(name);
        var markers = words.Where(IsMarker).ToList();

        if (words.Any(IsTwelveHour) || !YearFirst(markers))
        {
            return null;
        }

        var invariant = System.Globalization.CultureInfo.InvariantCulture;
        int Field(int at) => int.Parse(markers[at], invariant);

        try
        {
            if (markers[0].Length == 8)
            {
                // 20261126 then 1806, 180604 or 180604104.
                var date = DateTime.ParseExact(markers[0], "yyyyMMdd", invariant);
                var time = markers[1];
                return date
                    .AddHours(int.Parse(time[..2], invariant))
                    .AddMinutes(int.Parse(time[2..4], invariant))
                    .AddSeconds(time.Length >= 6 ? int.Parse(time[4..6], invariant) : 0)
                    .AddMilliseconds(time.Length == 9 ? int.Parse(time[6..], invariant) : 0);
            }

            // 2026 11 20 09 45 [30]: a date alone vouches for no moment.
            return markers.Count < 5
                ? null
                : new DateTime(Field(0), Field(1), Field(2), Field(3), Field(4), markers.Count > 5 ? Field(5) : 0);
        }
        catch (Exception exception) when (exception is ArgumentOutOfRangeException or FormatException)
        {
            return null;
        }
    }

    private static bool IsTwelveHour(string word) => word is "am" or "pm";

    /// <summary>When a file's recording began, its stamp read as a start or, for these files, as an end.</summary>
    internal static DateTimeOffset? StartOf(RecordingFacts file, IReadOnlySet<string> ends) =>
        file.RecordedAt is not { } stamp ? null
        : !ends.Contains(file.Path) ? stamp
        : Length(file) is { } length ? stamp - TimeSpan.FromSeconds(length)
        : stamp;

    private static bool Overlaps(IReadOnlyList<RecordingFacts> timed, IReadOnlySet<string> ends) =>
        Gaps([.. timed.OrderBy(f => StartOf(f, ends)).ThenBy(f => f.Name, NaturalOrder.Instance)], ends)
            .Any(gap => gap < -OverlapToleranceSeconds);

    /// <summary>A file's length where it is one a recording could have.</summary>
    private static double? Length(RecordingFacts file) =>
        file.DurationSeconds is { } seconds && double.IsFinite(seconds) && seconds is > 0 and <= RecordingFacts.LongestRecordingSeconds
            ? seconds
            : null;

    /// <summary>
    /// A name for the recordings together: what their names share, when that is a name at all,
    /// otherwise the first one's and a count.
    /// </summary>
    public static string CombinedName(IReadOnlyList<string> names)
    {
        ArgumentNullException.ThrowIfNull(names);

        if (names.Count == 0)
        {
            return string.Empty;
        }

        if (names.Count == 1)
        {
            return names[0];
        }

        var shared = names.Aggregate((a, b) => a[..a.Zip(b).TakeWhile(pair => pair.First == pair.Second).Count()]);

        // "Interview part " shares more than "Interview": the number that differed, and whatever
        // led into it, go with it.
        shared = shared.TrimEnd(' ', '-', '_', '.', '(', '#').TrimEnd();
        while (shared.Length > 0 && char.IsAsciiDigit(shared[^1]))
        {
            shared = shared[..^1];
        }

        shared = shared.TrimEnd(' ', '-', '_', '.', '(', '#').TrimEnd();
        if (shared.EndsWith(" part", StringComparison.OrdinalIgnoreCase))
        {
            shared = shared[..^5];
        }

        return shared.Length >= 3 ? shared : $"{names[0]} and {names.Count - 1} more";
    }

    private static List<RecordingFacts>? Distinct<T>(IReadOnlyList<RecordingFacts> files, Func<RecordingFacts, T?> key)
        where T : struct
    {
        if (files.Any(f => key(f) is null) || files.Select(key).Distinct().Count() != files.Count)
        {
            return null;
        }

        return [.. files.OrderBy(key)];
    }

    /// <summary>
    /// Whether the names carry an order: they are the same name apart from a number or a part
    /// letter, and those differ. "Part 1" and "Part 2" do, and "Session A" and "Session B"; so
    /// do "Recording" and "Recording 2", the one without a number being the first. Unrelated
    /// names that happen to hold digits do not: sorting them would be alphabetical order passed
    /// off as a sequence.
    /// </summary>
    /// <para>
    /// And the markers must say one thing. Either one of them differs, and it is short — a part
    /// number, not a phone number or an ID — or they are a date and time written year first,
    /// which sorts correctly digit by digit. A date written day or month first does not:
    /// 28.11.2026 would come after 05.12.2026.
    /// </para>
    private static bool NamesCarryOrder(IReadOnlyList<RecordingFacts> files)
    {
        // A twelve-hour clock does not sort as numbers do, in any form: "1 PM" is after "12 PM".
        if (files.Any(f => Words(f.Name).Any(IsTwelveHour)))
        {
            return false;
        }

        if (files.Select(f => f.Name).Distinct(StringComparer.OrdinalIgnoreCase).Count() != files.Count
            || files.Select(f => string.Join(' ', Words(f.Name).Where(word => !IsMarker(word)))).Distinct(StringComparer.Ordinal).Count() != 1)
        {
            return false;
        }

        var markers = files.Select(f => Words(f.Name).Where(IsMarker).ToList()).ToList();
        var counts = markers.Select(m => m.Count).Distinct().ToList();

        // "Recording" beside "Recording 2": one name without a marker, the rest with one.
        if (counts.Count > 1)
        {
            return counts.All(count => count <= 1) && markers.All(m => m.All(IsPartNumber));
        }

        if (counts[0] == 0)
        {
            return false;
        }

        // A date and time written year first, in either form, sorts as time does: as fields
        // (2026-11-20 9.45) or as phones and cameras write it (PXL_20261126_180604104,
        // 20261126_180604). Compact runs must be the same lengths in every name, or 180604 would
        // be compared with 180604104 as numbers; fields are short enough to compare as numbers.
        // A twelve-hour clock does not sort as time at all: 01.15 PM is after 12.30 PM.
        var compact = markers.All(m => m.Count > 0 && m[0].Length == 8);

        if (markers.All(YearFirst)
            && (!compact || markers.Select(m => string.Join(',', m.Select(marker => marker.Length))).Distinct().Count() == 1))
        {
            return true;
        }

        var differing = Enumerable.Range(0, counts[0])
            .Where(i => markers.Select(m => m[i].TrimStart('0')).Distinct(StringComparer.OrdinalIgnoreCase).Count() > 1)
            .ToList();

        // Three groups of digits or more, not a date: a phone number, an extension, an ID.
        var groups = markers[0].Count(m => m.All(char.IsAsciiDigit));

        return differing.Count == 1 && groups <= 2 && markers.All(m => IsPartNumber(m[differing[0]]));
    }

    /// <summary>
    /// A part number or letter, as opposed to a phone number, a time or a compact date: up to
    /// three digits, or a counter padded with zeros as recorders write them (ZOOM0001). Four
    /// digits without that padding are as likely 1430 or 2811 as part one thousand.
    /// </summary>
    private static bool IsPartNumber(string marker) =>
        !marker.All(char.IsAsciiDigit)
        || marker.Length <= 3
        || (marker.Length <= 6 && marker.StartsWith("00", StringComparison.Ordinal));

    /// <summary>
    /// A date and time written year first: as fields, 2026 11 20 09 45; or compact, as phones
    /// write it, 20261126 then 180604 or 180604104 (hours, minutes, seconds, milliseconds).
    /// </summary>
    private static bool YearFirst(List<string> markers)
    {
        if (markers.Count < 2 || !markers.All(m => m.All(char.IsAsciiDigit)))
        {
            return false;
        }

        if (markers[0].Length == 4 && markers[0] is ['1' or '2', ..])
        {
            return markers.Skip(1).All(m => m.Length <= 2);
        }

        return markers[0].Length == 8
            && DateOnly.TryParseExact(markers[0], "yyyyMMdd", System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.None, out var date)
            && date.Year is >= 1900 and < 2100
            && markers.Skip(1).All(m => m.Length is 4 or 6 or 9 && int.Parse(m[..2], System.Globalization.CultureInfo.InvariantCulture) < 24);
    }

    /// <summary>A name's words in lower case, digits and punctuation splitting them as spaces do.</summary>
    private static string[] Words(string name)
    {
        var spaced = new string([.. name.Select(c => char.IsPunctuation(c) || char.IsSeparator(c) ? ' ' : char.ToLowerInvariant(c))]);

        // Digits are their own words, so "ZOOM0001" is "zoom" and "0001".
        var words = new List<string>();
        foreach (var token in spaced.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            var at = 0;
            while (at < token.Length)
            {
                var digits = char.IsAsciiDigit(token[at]);
                var end = at;
                while (end < token.Length && char.IsAsciiDigit(token[end]) == digits)
                {
                    end++;
                }

                words.Add(token[at..end]);
                at = end;
            }
        }

        return [.. words];
    }

    /// <summary>A number, or a part letter standing alone: what tells one part from the next.</summary>
    private static bool IsMarker(string word) =>
        word.All(char.IsAsciiDigit) || (word.Length == 1 && char.IsAsciiLetter(word[0]));

    /// <summary>By when each was last saved, less its length where known: when it was begun.</summary>
    private static List<RecordingFacts> Saved(IReadOnlyList<RecordingFacts> files) =>
        [.. files
            .OrderBy(Begun)
            .ThenBy(f => f.Name, NaturalOrder.Instance)];

    private static DateTimeOffset Begun(RecordingFacts file)
    {
        if (file.LastSaved is not { } saved)
        {
            return DateTimeOffset.MaxValue;
        }

        // Guarded here too: the record is public, and a length no recording has would throw.
        var length = file.DurationSeconds is { } seconds && double.IsFinite(seconds) && seconds is > 0 and <= RecordingFacts.LongestRecordingSeconds
            ? seconds
            : 0;

        return saved - TimeSpan.FromSeconds(length) is var begun && begun > DateTimeOffset.MinValue ? begun : saved;
    }

    /// <summary>Whether any two files were plainly begun in the other order from the one proposed.</summary>
    private static bool SavedDisagrees(IReadOnlyList<RecordingFacts> ordered)
    {
        for (var i = 0; i < ordered.Count; i++)
        {
            for (var j = i + 1; j < ordered.Count; j++)
            {
                if (ordered[i].LastSaved is not null && ordered[j].LastSaved is not null && Begun(ordered[i]) > Begun(ordered[j]))
                {
                    return true;
                }
            }
        }

        return false;
    }

    private static bool Same(IReadOnlyList<RecordingFacts> a, IReadOnlyList<RecordingFacts> b) =>
        a.Select(f => f.Path).SequenceEqual(b.Select(f => f.Path));

    private static List<double?> Gaps(IReadOnlyList<RecordingFacts> ordered, IReadOnlySet<string> ends)
    {
        var gaps = new List<double?>(ordered.Count);

        for (var i = 0; i < ordered.Count; i++)
        {
            gaps.Add(i > 0
                && StartOf(ordered[i - 1], ends) is { } before
                && Length(ordered[i - 1]) is { } length
                && StartOf(ordered[i], ends) is { } start
                    ? (start - before).TotalSeconds - length
                    : null);
        }

        return gaps;
    }

    /// <summary>Text compared as text, numbers as numbers: "Part 2" before "Part 10".</summary>
    public sealed class NaturalOrder : IComparer<string>
    {
        public static NaturalOrder Instance { get; } = new();

        public int Compare(string? x, string? y)
        {
            if (x is null || y is null)
            {
                return x is null ? (y is null ? 0 : -1) : 1;
            }

            int i = 0, j = 0;

            while (i < x.Length && j < y.Length)
            {
                if (char.IsAsciiDigit(x[i]) && char.IsAsciiDigit(y[j]))
                {
                    var (a, nextI) = Run(x, i);
                    var (b, nextJ) = Run(y, j);
                    var byNumber = a.TrimStart('0').Length.CompareTo(b.TrimStart('0').Length);

                    if (byNumber == 0)
                    {
                        byNumber = string.CompareOrdinal(a.TrimStart('0'), b.TrimStart('0'));
                    }

                    if (byNumber != 0)
                    {
                        return byNumber;
                    }

                    (i, j) = (nextI, nextJ);
                    continue;
                }

                var byText = char.ToLowerInvariant(x[i]).CompareTo(char.ToLowerInvariant(y[j]));
                if (byText != 0)
                {
                    return byText;
                }

                i++;
                j++;
            }

            return (x.Length - i).CompareTo(y.Length - j);
        }

        private static (string Digits, int Next) Run(string text, int at)
        {
            var end = at;
            while (end < text.Length && char.IsAsciiDigit(text[end]))
            {
                end++;
            }

            return (text[at..end], end);
        }
    }
}
