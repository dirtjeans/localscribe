namespace LocalScribe.Core.Diarization;

/// <summary>How to work out who spoke when.</summary>
public enum DiarizationMethod
{
    /// <summary>
    /// Follow each speaker between overlapping windows, then group the tracks.
    /// <para>
    /// Uses the segmentation model's own opinion that two people are talking at a given moment as
    /// a constraint — they cannot be the same person — rather than asking the voice model. On a
    /// phone recording of an argument this was the difference between three speakers and
    /// nineteen, one of which held 85% of the speech.
    /// </para>
    /// </summary>
    Tracking,

    /// <summary>
    /// Cut the recording into stretches of speech and cluster them by voice.
    /// <para>
    /// The older path, and better on some material. On a studio podcast with five voices and long
    /// uninterrupted runs, tracking found three speakers and 22 turns in seven minutes, while
    /// clustering found five and held that answer across four thresholds. Long single-speaker
    /// stretches give tracking few overlaps to constrain anything with, which is exactly what it
    /// depends on.
    /// </para>
    /// </summary>
    Voices,
}

/// <summary>
/// Which method to use, recorded beside the models so it survives a restart.
/// <para>
/// Neither method wins everywhere and nothing here can yet tell which recording is which, so the
/// choice is a setting rather than a decision. Tracking stays the default because the failure it
/// prevents — one voice split into nineteen — is worse than the one it causes.
/// </para>
/// </summary>
public static class DiarizationChoice
{
    /// <summary>What the file is called, in the diarization model directory.</summary>
    public const string FileName = "active-diarizer.txt";

    /// <summary>The method recorded for a model directory, or tracking when none is.</summary>
    public static DiarizationMethod Read(string directory)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);

        var path = Path.Combine(directory, FileName);

        if (!File.Exists(path))
        {
            return DiarizationMethod.Tracking;
        }

        return Parse(File.ReadAllText(path));
    }

    /// <summary>Records a method for a model directory.</summary>
    public static void Write(string directory, DiarizationMethod method)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);

        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, FileName), Name(method));
    }

    /// <summary>The name a method is written under.</summary>
    public static string Name(DiarizationMethod method) =>
        method == DiarizationMethod.Voices ? "voices" : "tracking";

    /// <summary>
    /// The method a name refers to, falling back to tracking.
    /// <para>
    /// A file nobody can parse should not stop a transcription. The default is the safer of the
    /// two, so falling back to it costs accuracy on some recordings and never the whole run.
    /// </para>
    /// </summary>
    public static DiarizationMethod Parse(string? name) =>
        name?.Trim().Equals("voices", StringComparison.OrdinalIgnoreCase) == true
            ? DiarizationMethod.Voices
            : DiarizationMethod.Tracking;
}
