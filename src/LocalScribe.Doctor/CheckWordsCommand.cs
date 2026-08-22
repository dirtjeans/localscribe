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

        // Read once, whole. Every word is then found by searching this rather than by decoding a
        // window around where it claims to be — which could only ever confirm words that were
        // roughly right already.
        var reading = aligner.ReadAll(scores);

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

                if (Locate(reading, scores, word) is not { } shift)
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
    /// Shorter than this and a word occurs everywhere. Nothing useful can be said about where
    /// "a" or "of" is.
    /// </summary>
    private const int ShortestFindable = 4;

    /// <summary>
    /// How far the word's letters actually sit from where it claims to be, or null when they are
    /// nowhere in the recording.
    /// <para>
    /// Positive means the word is really said after its timestamp, so the marker reaches it
    /// early. The nearest occurrence wins, so a word said several times is credited with the
    /// closest one rather than being reported as wildly adrift.
    /// </para>
    /// </summary>
    private static double? Locate(
        ForcedAligner.AudioReading reading,
        AlignmentScores scores,
        WordTimings.Word word)
    {
        var spelled = Letters(word.Text);
        var best = double.NaN;

        var at = reading.Letters.IndexOf(spelled, StringComparison.Ordinal);

        while (at >= 0)
        {
            var heard = scores.SecondsAt(reading.Frames[at]);
            var shift = heard - word.StartSeconds;

            if (double.IsNaN(best) || Math.Abs(shift) < Math.Abs(best))
            {
                best = shift;
            }

            at = reading.Letters.IndexOf(spelled, at + 1, StringComparison.Ordinal);
        }

        return double.IsNaN(best) ? null : best;
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
