using LocalScribe.Core.Alignment;
using LocalScribe.Core.Archive;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Transcription;
using LocalScribe.Onnx;

namespace LocalScribe.Doctor;

/// <summary>
/// Checks a saved transcript's word times against the recording inside it.
/// <para>
/// "Is the highlight in the right place" has three separate questions hiding in it: the measured
/// times could be wrong, the pairing of words to times could be shifted, or the playback clock
/// could be lying. Reasoning about a screenshot cannot tell those apart, and guessing between
/// them has now cost several rounds. This can tell them apart, because an archive carries the
/// audio and the timed transcript together — so every word can be asked to prove itself by
/// decoding the audio at the moment it claims to have been said.
/// </para>
/// <para>
/// A word whose own seconds decode to its own letters is timed correctly, whatever the app then
/// does with it. A run of words that all read back late is a timing fault. Words that check out
/// here while the app still lights the wrong one puts the fault after this point — in the pairing
/// or in the clock — which is worth knowing before changing either.
/// </para>
/// </summary>
public static class CheckWordsCommand
{
    public static int Run(string archivePath, string modelDirectory, ExecutionPlan plan)
    {
        if (!File.Exists(archivePath))
        {
            Console.Error.WriteLine($"No such file: {archivePath}");
            return 1;
        }

        if (ForcedAligner.Find(modelDirectory) is not { } directory)
        {
            Console.Error.WriteLine($"No alignment model under {modelDirectory}.");
            return 1;
        }

        Heading("Check words");
        Console.WriteLine($"  Archive    {archivePath}");

        TranscriptArchive.Contents contents;

        using (var file = File.OpenRead(archivePath))
        {
            contents = TranscriptArchive.Read(file);
        }

        Console.WriteLine(
            $"  Recording  {contents.Audio.DurationSeconds:F1}s, {contents.Segments.Count} segments");

        using var aligner = ForcedAligner.Load(directory, plan);

        var scores = aligner.Scan(contents.Audio, new Progress<double>(Report));
        Console.WriteLine();

        if (scores is null)
        {
            Console.Error.WriteLine("  The recording could not be scanned.");
            return 1;
        }

        var measured = 0;
        var estimated = 0;
        var checkedWords = 0;
        var unheard = 0;
        var shifts = new List<double>();
        var adrift = new List<(double At, double Shift, string Word, string Heard)>();

        foreach (var segment in contents.Segments)
        {
            var words = aligner.Align(scores, segment, CancellationToken.None);

            if (words is null)
            {
                estimated++;
                continue;
            }

            measured++;

            foreach (var word in words)
            {
                if (word.EndSeconds <= word.StartSeconds)
                {
                    continue;
                }

                if (Letters(word.Text).Length < ShortestFindable)
                {
                    continue;
                }

                checkedWords++;

                if (Locate(aligner, scores, word) is not { } shift)
                {
                    // The greedy read-back misspells words the way any recogniser does, so this
                    // is usually the decode being approximate rather than the word being absent.
                    // Counted rather than listed, because the rate matters and the instances do
                    // not.
                    unheard++;
                    continue;
                }

                shifts.Add(shift);

                if (Math.Abs(shift) >= NoticeableDrift)
                {
                    adrift.Add((
                        word.StartSeconds,
                        shift,
                        word.Text,
                        aligner.Read(scores, word.StartSeconds, word.EndSeconds)));
                }
            }
        }

        Heading("Segments");
        Console.WriteLine($"  Timed      {measured}");
        Console.WriteLine($"  Estimated  {estimated}   (the aligner could not place these)");
        Console.WriteLine($"  Words      {checkedWords} long enough to find, {unheard} not matched in the audio");

        if (shifts.Count > 0)
        {
            shifts.Sort();

            // The middle of the distribution, which is what says whether the whole transcript is
            // sitting late rather than a few words being wrong.
            Console.WriteLine($"  Typical    {shifts[shifts.Count / 2]:+0.00;-0.00}s out");
            Console.WriteLine($"  Worst      {shifts[0]:+0.00;-0.00}s to {shifts[^1]:+0.00;-0.00}s");
        }

        Heading("Words that are not where they say they are");

        if (adrift.Count == 0)
        {
            Console.WriteLine("  None. Every word read back at its own timestamp.");
            Console.WriteLine();
            Console.WriteLine("  The times are right, so a highlight in the wrong place is being");
            Console.WriteLine("  caused after this point: by how words are paired to them, or by the");
            Console.WriteLine("  clock playback is measured against.");
            return 0;
        }

        foreach (var (at, shift, word, heard) in adrift.Take(40))
        {
            Console.WriteLine($"  {at,8:F2}  {shift,7:+0.00;-0.00}s  {word,-18} reads as \"{heard}\"");
        }

        if (adrift.Count > 40)
        {
            Console.WriteLine($"  … and {adrift.Count - 40} more not listed.");
        }

        Console.WriteLine();
        Console.WriteLine($"  {adrift.Count} of {checkedWords} words are adrift.");
        Console.WriteLine("  Positive means the word was said later than its timestamp, so the marker");
        Console.WriteLine("  reaches it early. Negative means it was said earlier, so the marker lags.");

        return 0;
    }

    /// <summary>
    /// How far from its stated time a word has to be before it is worth reporting. Below this it
    /// is the difference between two reasonable opinions about where a word begins.
    /// </summary>
    private const double NoticeableDrift = 0.25;

    /// <summary>
    /// Shorter than this and a word carries too little to recognise. Nothing useful can be said
    /// about where "a" or "of" is.
    /// </summary>
    private const int ShortestFindable = 7;

    /// <summary>How far either side of its stated time a word is looked for.</summary>
    private const double SearchSeconds = 5;

    /// <summary>How finely, in seconds.</summary>
    private const double Step = 0.25;

    /// <summary>
    /// How much of a word's spelling must survive the decode before the audio is agreed to hold
    /// that word.
    /// <para>
    /// Not all of it. The read-back is a greedy pass over the frames and misspells words the way
    /// any recogniser does — "Elijah" comes back as "elija", "artificial" as "artofficial". An
    /// exact match would call those absent, and the first version of this did exactly that: it
    /// searched the whole recording for a perfect spelling, failed to find one at the right
    /// moment, matched some other occurrence seconds away, and reported a word that was correctly
    /// placed as adrift. Every number it produced had to be checked by hand against the decode
    /// beside it, which is not a measurement.
    /// </para>
    /// </summary>
    private const double CloseEnough = 0.6;

    /// <summary>
    /// How far the word's sound actually sits from where it claims to be, or null when nothing
    /// nearby resembles it.
    /// <para>
    /// Positive means the word is really said after its timestamp, so the marker reaches it early.
    /// Measured by reading the audio at a range of offsets and keeping the one that reads most
    /// like the word — a local question, which is the only kind that can be answered without
    /// mistaking one occurrence of a word for another.
    /// </para>
    /// </summary>
    private static double? Locate(ForcedAligner aligner, AlignmentScores scores, WordTimings.Word word)
    {
        var spelled = Letters(word.Text);
        var length = word.EndSeconds - word.StartSeconds;

        var best = double.NaN;
        var closest = CloseEnough;

        for (var shift = -SearchSeconds; shift <= SearchSeconds; shift += Step)
        {
            var from = word.StartSeconds + shift;
            if (from < 0)
            {
                continue;
            }

            var heard = aligner.Read(scores, from, from + length + Step);
            var likeness = Likeness(spelled, heard);

            // Ties go to the smaller shift: a word that reads equally well at its own time and a
            // second away is not evidence of drift.
            if (likeness > closest || (likeness >= closest && !double.IsNaN(best) && Math.Abs(shift) < Math.Abs(best)))
            {
                closest = likeness;
                best = shift;
            }
        }

        return double.IsNaN(best) ? null : best;
    }

    /// <summary>
    /// How much of <paramref name="word"/> appears in <paramref name="heard"/>, in order, as a
    /// share of the word's length.
    /// </summary>
    private static double Likeness(string word, string heard)
    {
        if (word.Length == 0 || heard.Length == 0)
        {
            return 0;
        }

        // Longest common subsequence. Letters in the right order but with the decode's own
        // insertions and omissions between them, which is exactly how a greedy read-back differs
        // from the word it heard.
        var previous = new int[heard.Length + 1];
        var current = new int[heard.Length + 1];

        for (var i = 1; i <= word.Length; i++)
        {
            for (var j = 1; j <= heard.Length; j++)
            {
                current[j] = word[i - 1] == heard[j - 1]
                    ? previous[j - 1] + 1
                    : Math.Max(previous[j], current[j - 1]);
            }

            (previous, current) = (current, previous);
            Array.Clear(current);
        }

        return previous[heard.Length] / (double)word.Length;
    }

    /// <summary>The word as the aligner's romanised alphabet would spell it.</summary>
    private static string Letters(string word) =>
        new([.. word.Where(char.IsLetter).Select(char.ToLowerInvariant)]);

    private static void Report(double fraction) =>
        Console.Write($"\r  Scanning   {(int)(fraction * 100)}%   ");

    private static void Heading(string title)
    {
        Console.WriteLine();
        Console.WriteLine(title);
        Console.WriteLine(new string('-', title.Length));
    }
}
