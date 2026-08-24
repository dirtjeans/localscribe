using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using LocalScribe.Core.Audio;
using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Archive;

/// <summary>
/// A transcript and the recording it came from, in one file.
/// <para>
/// Saving the text alone loses the half of the work that makes it useful: click a line and hear
/// it, drag the waveform and watch the words follow, correct a speaker by listening again. Those
/// need the audio and the timings together, and until now the audio was only ever in memory — a
/// recording made in the app was gone the moment the app closed, which is a poor way to treat the
/// one artefact the user cannot recreate.
/// </para>
/// <para>
/// The container is a zip with a fixed layout, which is dull on purpose. Anyone can open it
/// without this app, the parts are inspectable, and a future version can add files without
/// breaking an older reader. It carries a plain-text transcript alongside the structured one for
/// exactly that reason: whatever happens to this program, the words remain readable.
/// </para>
/// </summary>
public static class TranscriptArchive
{
    /// <summary>The extension these are saved with.</summary>
    public const string Extension = ".scrb";

    /// <summary>
    /// Extensions earlier versions saved with, still opened.
    /// <para>
    /// A file already on disk does not stop being readable because the name got shorter. Writing
    /// only the current one and reading all of them is the whole of the compatibility story here:
    /// the container has not changed, only what it is called.
    /// </para>
    /// </summary>
    public static IReadOnlyList<string> AlsoOpens { get; } = [".lscribe"];

    /// <summary>Every extension that can be opened, current first.</summary>
    public static IReadOnlyList<string> Extensions { get; } = [Extension, .. AlsoOpens];

    /// <summary>True when this path names an archive, whatever era it was saved in.</summary>
    public static bool IsArchive(string path)
    {
        ArgumentNullException.ThrowIfNull(path);

        var extension = Path.GetExtension(path);

        return Extensions.Any(known => extension.Equals(known, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Bumped when a reader would need to know the difference. Written into every archive so an
    /// older build meeting a newer file can say so rather than misreading it.
    /// </summary>
    public const int CurrentVersion = 1;

    private const string ManifestEntry = "manifest.json";
    private const string AudioEntry = "audio.wav";
    private const string SegmentsEntry = "transcript.json";
    private const string TextEntry = "transcript.txt";

    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    /// <param name="SourceName">What the recording was called when it was made.</param>
    /// <param name="SavedUtc">When the archive was written.</param>
    public sealed record Manifest(
        int Version,
        string SourceName,
        DateTimeOffset SavedUtc,
        double DurationSeconds,
        int SampleRate,
        int SegmentCount,
        int SpeakerCount);

    /// <summary>Everything an archive holds.</summary>
    public sealed record Contents(
        Manifest Manifest,
        PcmAudio Audio,
        IReadOnlyList<TranscriptSegment> Segments);

    /// <summary>Writes the transcript and its audio to <paramref name="path"/>.</summary>
    public static void Save(
        string path,
        PcmAudio audio,
        IReadOnlyList<TranscriptSegment> segments,
        string sourceName,
        DateTimeOffset? savedUtc = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(segments);

        using var file = File.Create(path);
        Write(file, audio, segments, sourceName, savedUtc);
    }

    /// <summary>The same, to a stream, so this can be tested without touching a disk.</summary>
    public static void Write(
        Stream destination,
        PcmAudio audio,
        IReadOnlyList<TranscriptSegment> segments,
        string sourceName,
        DateTimeOffset? savedUtc = null)
    {
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(segments);

        // Left open so a caller that owns the stream still owns it afterwards.
        using var zip = new ZipArchive(destination, ZipArchiveMode.Create, leaveOpen: true);

        var manifest = new Manifest(
            CurrentVersion,
            sourceName ?? string.Empty,
            savedUtc ?? DateTimeOffset.UtcNow,
            audio.DurationSeconds,
            audio.SampleRate,
            segments.Count,
            segments.Select(s => s.Speaker).Where(s => !string.IsNullOrWhiteSpace(s)).Distinct(StringComparer.Ordinal).Count());

        WriteText(zip, ManifestEntry, JsonSerializer.Serialize(manifest, Json));
        WriteText(zip, SegmentsEntry, JsonSerializer.Serialize(segments, Json));

        // A readable copy, so the words survive this program.
        WriteText(zip, TextEntry, PlainText(segments));

        // Deliberately not compressed: PCM does not shrink much and the audio is most of the
        // file, so the time would buy almost nothing.
        var entry = zip.CreateEntry(AudioEntry, CompressionLevel.NoCompression);
        using var stream = entry.Open();
        WriteWave(stream, audio);
    }

    /// <summary>Reads an archive back, or throws if it is not one.</summary>
    public static Contents Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        using var file = File.OpenRead(path);
        return Read(file);
    }

    /// <summary>The same, from a stream.</summary>
    public static Contents Read(Stream source)
    {
        ArgumentNullException.ThrowIfNull(source);

        using var zip = new ZipArchive(source, ZipArchiveMode.Read, leaveOpen: true);

        var manifest = JsonSerializer.Deserialize<Manifest>(ReadText(zip, ManifestEntry), Json)
            ?? throw new InvalidDataException("The archive has no manifest.");

        if (manifest.Version > CurrentVersion)
        {
            throw new InvalidDataException(
                $"This transcript was saved by a newer version of LocalScribe (format {manifest.Version}, "
                + $"this build reads {CurrentVersion}).");
        }

        var segments = JsonSerializer.Deserialize<List<TranscriptSegment>>(ReadText(zip, SegmentsEntry), Json)
            ?? [];

        var audioEntry = zip.GetEntry(AudioEntry)
            ?? throw new InvalidDataException("The archive has no audio.");

        using var audioStream = audioEntry.Open();
        var audio = ReadWave(audioStream);

        return new Contents(manifest, audio, segments);
    }

    /// <summary>True when the file looks like one of ours, without reading all of it.</summary>
    public static bool Looks(string path)
    {
        if (!File.Exists(path))
        {
            return false;
        }

        try
        {
            using var file = File.OpenRead(path);
            using var zip = new ZipArchive(file, ZipArchiveMode.Read);

            return zip.GetEntry(ManifestEntry) is not null && zip.GetEntry(AudioEntry) is not null;
        }
        catch (Exception exception) when (exception is InvalidDataException or IOException)
        {
            return false;
        }
    }

    private static string PlainText(IReadOnlyList<TranscriptSegment> segments)
    {
        var builder = new StringBuilder();
        string? speaker = null;

        foreach (var segment in segments)
        {
            var text = segment.Text.Trim();
            if (text.Length == 0)
            {
                continue;
            }

            if (!string.Equals(segment.Speaker, speaker, StringComparison.Ordinal))
            {
                speaker = segment.Speaker;

                if (builder.Length > 0)
                {
                    builder.AppendLine().AppendLine();
                }

                if (!string.IsNullOrWhiteSpace(speaker))
                {
                    builder.AppendLine($"{speaker}:");
                }
            }
            else if (builder.Length > 0)
            {
                builder.Append(' ');
            }

            builder.Append(text);
        }

        return builder.ToString();
    }

    private static void WriteText(ZipArchive zip, string name, string content)
    {
        using var stream = zip.CreateEntry(name, CompressionLevel.Optimal).Open();
        using var writer = new StreamWriter(stream, new UTF8Encoding(false));
        writer.Write(content);
    }

    private static string ReadText(ZipArchive zip, string name)
    {
        var entry = zip.GetEntry(name) ?? throw new InvalidDataException($"The archive has no {name}.");

        using var stream = entry.Open();
        using var reader = new StreamReader(stream, Encoding.UTF8);

        return reader.ReadToEnd();
    }

    /// <summary>
    /// Sixteen-bit PCM, which every audio program on earth opens. The samples are floats in
    /// memory, but they came from a sixteen-bit source and quantising them back costs nothing
    /// audible while halving the file.
    /// </summary>
    private static void WriteWave(Stream destination, PcmAudio audio)
    {
        var samples = audio.Samples;
        var dataBytes = samples.Length * sizeof(short);

        using var writer = new BinaryWriter(destination, Encoding.ASCII, leaveOpen: true);

        writer.Write("RIFF"u8);
        writer.Write(36 + dataBytes);
        writer.Write("WAVE"u8);

        writer.Write("fmt "u8);
        writer.Write(16);                                   // chunk size
        writer.Write((short)1);                             // PCM
        writer.Write((short)1);                             // mono
        writer.Write(audio.SampleRate);
        writer.Write(audio.SampleRate * sizeof(short));     // bytes per second
        writer.Write((short)sizeof(short));                 // block align
        writer.Write((short)16);                            // bits per sample

        writer.Write("data"u8);
        writer.Write(dataBytes);

        foreach (var sample in samples)
        {
            writer.Write((short)Math.Clamp(sample * short.MaxValue, short.MinValue, short.MaxValue));
        }
    }

    private static PcmAudio ReadWave(Stream source)
    {
        // Copied out first: zip entry streams cannot seek, and the header needs reading before
        // the data length is known.
        using var buffer = new MemoryStream();
        source.CopyTo(buffer);

        var bytes = buffer.ToArray();
        if (bytes.Length < 44 || Encoding.ASCII.GetString(bytes, 0, 4) != "RIFF")
        {
            throw new InvalidDataException("The archive's audio is not a WAV file.");
        }

        var sampleRate = BitConverter.ToInt32(bytes, 24);
        var at = 12;

        while (at + 8 <= bytes.Length)
        {
            var name = Encoding.ASCII.GetString(bytes, at, 4);
            var size = BitConverter.ToInt32(bytes, at + 4);

            if (name == "data")
            {
                var count = Math.Min(size, bytes.Length - at - 8) / sizeof(short);
                var samples = new float[count];

                for (var i = 0; i < count; i++)
                {
                    samples[i] = BitConverter.ToInt16(bytes, at + 8 + (i * sizeof(short))) / (float)short.MaxValue;
                }

                return new PcmAudio(samples, sampleRate);
            }

            at += 8 + size + (size % 2);
        }

        throw new InvalidDataException("The archive's audio has no samples.");
    }
}
