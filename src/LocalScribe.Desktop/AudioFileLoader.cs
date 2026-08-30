using System.Diagnostics;
using LocalScribe.Core.Audio;

namespace LocalScribe.App;

/// <summary>
/// Reads audio files and converts them into the format Whisper needs — the macOS twin of the
/// Media Foundation loader. Plain 16 kHz WAV is read directly; every other audio format goes
/// through <c>afconvert</c>, and video containers have their soundtrack pulled out by
/// <c>avconvert</c> first. Both tools ship with macOS: shelling out to them rather than
/// bundling ffmpeg keeps the dependency count at zero, and the pipeline only ever sees PCM
/// this process read itself.
/// </summary>
public static class AudioFileLoader
{
    private static readonly HashSet<string> VideoExtensions =
        new(StringComparer.OrdinalIgnoreCase) { ".mp4", ".mov", ".m4v" };

    /// <summary>
    /// Every extension the picker should offer. Shorter than the Windows list on purpose:
    /// CoreAudio does not decode wma, ogg, mkv or webm, and offering a file the loader will
    /// refuse is worse than not offering it.
    /// </summary>
    public static IReadOnlyList<string> SupportedExtensions { get; } =
    [
        ".wav", ".mp3", ".m4a", ".aac", ".flac", ".aiff",
        ".mp4", ".mov", ".m4v",
    ];

    /// <summary>True when the file is a video container rather than plain audio.</summary>
    public static bool IsVideo(string path) =>
        VideoExtensions.Contains(Path.GetExtension(path));

    /// <summary>Loads and resamples anything macOS can decode, including a video's soundtrack.</summary>
    public static PcmAudio Load(string path)
    {
        if (IsVideo(path))
        {
            var soundtrack = Path.Combine(
                Path.GetTempPath(), $"localscribe-{Guid.NewGuid():N}.m4a");

            try
            {
                Run("/usr/bin/avconvert",
                    ["--preset", "PresetAppleM4A", "--source", path, "--output", soundtrack],
                    "macOS could not extract the audio track");

                return ConvertAndRead(soundtrack);
            }
            finally
            {
                DeleteQuietly(soundtrack);
            }
        }

        if (Path.GetExtension(path).Equals(".wav", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                var direct = WavReader.Read(path);

                if (direct.SampleRate == PcmAudio.WhisperSampleRate)
                {
                    return direct;
                }
            }
            catch (Exception exception) when (exception is InvalidDataException or NotSupportedException)
            {
                // An exotic WAV variant; afconvert reads more of them than the parser does.
            }
        }

        return ConvertAndRead(path);
    }

    private static PcmAudio ConvertAndRead(string path)
    {
        var converted = Path.Combine(Path.GetTempPath(), $"localscribe-{Guid.NewGuid():N}.wav");

        try
        {
            Run("/usr/bin/afconvert",
                [path, converted, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1"],
                "macOS could not decode this file");

            return WavReader.Read(converted);
        }
        finally
        {
            DeleteQuietly(converted);
        }
    }

    private static void Run(string tool, string[] arguments, string failure)
    {
        var info = new ProcessStartInfo(tool)
        {
            RedirectStandardError = true,
            UseShellExecute = false,
        };

        foreach (var argument in arguments)
        {
            info.ArgumentList.Add(argument);
        }

        using var process = Process.Start(info)
            ?? throw new InvalidOperationException($"{Path.GetFileName(tool)} did not start.");

        var errors = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (process.ExitCode != 0)
        {
            throw new InvalidDataException($"{failure}: {errors.Trim()}");
        }
    }

    private static void DeleteQuietly(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (IOException)
        {
            // A leftover temp file is not worth failing a load over.
        }
    }
}
