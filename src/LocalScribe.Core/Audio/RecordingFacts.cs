using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace LocalScribe.Core.Audio;

/// <summary>
/// What a recording file says about itself: when it was recorded, its track number, how long it
/// is. Read from the container's own fields rather than decoded, so a handful of files can be put
/// in order before any of them is converted.
/// <para>
/// Only the fields recorders and phones actually write: an MP4-family file's movie header (its
/// creation time, which Voice Memos and most phones set to when recording began) and track
/// number; a WAV's broadcast extension (<c>bext</c>, which field recorders stamp with the date
/// and time recording began) and its INFO list; an MP3's ID3 recording date and track; a FLAC's
/// stream info and comments. Anything missing is null, and a file that cannot be read at all
/// still has the time it was last saved, which the file system always knows.
/// </para>
/// </summary>
/// <param name="Path">The file.</param>
/// <param name="RecordedAt">
/// When recording began, as the file itself records it — or, for some writers, when it stopped:
/// a Pixel stamps its videos as it closes them. <see cref="RecordingOrder"/> tells which.
/// </param>
/// <param name="Track">A track or part number written into the file.</param>
/// <param name="DurationSeconds">How long it is, where the header says without decoding.</param>
/// <param name="LastSaved">When the file system says it was last written.</param>
public sealed partial record RecordingFacts(
    string Path,
    DateTimeOffset? RecordedAt,
    int? Track,
    double? DurationSeconds,
    DateTimeOffset? LastSaved)
{
    /// <summary>The file's name without its folder or extension.</summary>
    public string Name => System.IO.Path.GetFileNameWithoutExtension(Path);

    /// <summary>Movie headers count seconds from the start of 1904, in UTC.</summary>
    private static readonly DateTimeOffset Mp4Epoch = new(1904, 1, 1, 0, 0, 0, TimeSpan.Zero);

    /// <summary>A header larger than this is not one worth reading to put files in order.</summary>
    private const long LargestHeaderBytes = 64L * 1024 * 1024;

    /// <summary>Reads what the file says about itself. Never throws for a file that exists.</summary>
    public static RecordingFacts Read(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        DateTimeOffset? saved = File.Exists(path) ? new DateTimeOffset(File.GetLastWriteTimeUtc(path), TimeSpan.Zero) : null;
        var facts = new RecordingFacts(path, null, null, null, saved);

        try
        {
            using var stream = File.OpenRead(path);
            return Read(stream, facts);
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            // A header that cannot be read leaves the file system's time, which is still an order.
            // Every failure, not a list of expected ones: these headers come from any program
            // that ever wrote a file, and a file this cannot read must never stop the others
            // being ordered — the first review found three exception types the list missed.
            return facts;
        }
    }

    /// <summary>The same, from a stream, so this can be tested without files.</summary>
    public static RecordingFacts Read(Stream stream, RecordingFacts known)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(known);

        RecordingFacts facts;

        try
        {
            facts = Parse(stream, known);
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            // The same promise as the path overload: a header this cannot read leaves what was known.
            return known;
        }

        // A length no recording has is a field meaning "unknown". The MP4 sentinels are caught
        // where they are read; this is the backstop for any other writer's.
        return facts.DurationSeconds is { } seconds && !(double.IsFinite(seconds) && seconds > 0 && seconds <= LongestRecordingSeconds)
            ? facts with { DurationSeconds = null }
            : facts;
    }

    private static RecordingFacts Parse(Stream stream, RecordingFacts known)
    {
        var head = new byte[12];
        var got = stream.Read(head, 0, head.Length);
        stream.Position = 0;

        return
            got >= 12 && head.AsSpan(0, 4).SequenceEqual("RIFF"u8) && head.AsSpan(8, 4).SequenceEqual("WAVE"u8) ? Wave(stream, known)
            : got >= 3 && head.AsSpan(0, 3).SequenceEqual("ID3"u8) ? Id3(stream, known)
            : got >= 4 && head.AsSpan(0, 4).SequenceEqual("fLaC"u8) ? Flac(stream, known)
            : got >= 8 && head.AsSpan(4, 4).SequenceEqual("ftyp"u8) ? Mp4(stream, known)
            : known;
    }

    /// <summary>A week: longer than anything anyone records.</summary>
    public const double LongestRecordingSeconds = 7 * 86_400;

    /* ---- MP4, M4A, MOV ---------------------------------------------------------------- */

    private static RecordingFacts Mp4(Stream stream, RecordingFacts known)
    {
        foreach (var (type, start, length) in Boxes(stream, 0, stream.Length))
        {
            if (type != "moov")
            {
                continue;
            }

            var moov = ReadExactly(stream, start, length);
            var facts = known;

            foreach (var (inner, at, size) in Boxes(moov, 0, moov.Length))
            {
                if (inner == "mvhd")
                {
                    facts = MovieHeader(moov.AsSpan(at, size), facts);
                }
                else if (inner == "udta" && TrackNumber(moov.AsSpan(at, size)) is { } track)
                {
                    facts = facts with { Track = track };
                }
            }

            return facts;
        }

        return known;
    }

    private static RecordingFacts MovieHeader(ReadOnlySpan<byte> box, RecordingFacts facts)
    {
        if (box.Length < 20 || (box[0] == 1 && box.Length < 32))
        {
            return facts;
        }

        var version = box[0];
        ulong created;
        uint timescale;
        ulong duration;

        if (version == 1)
        {
            created = BinaryPrimitives.ReadUInt64BigEndian(box[4..]);
            timescale = BinaryPrimitives.ReadUInt32BigEndian(box[20..]);
            duration = BinaryPrimitives.ReadUInt64BigEndian(box[24..]);
        }
        else
        {
            created = BinaryPrimitives.ReadUInt32BigEndian(box[4..]);
            timescale = BinaryPrimitives.ReadUInt32BigEndian(box[12..]);
            duration = BinaryPrimitives.ReadUInt32BigEndian(box[16..]);
        }

        // Zero means nobody set it, which a writer that stamps nothing leaves.
        DateTimeOffset? at = created > 0 && created < 200UL * 365 * 86_400 ? Mp4Epoch.AddSeconds(created) : null;
        // All ones is the standard's "unknown", which a writer that does not know the length is
        // told to write. Divided by an audio timescale it passes for a day-long recording.
        var unknown = duration == (version == 1 ? ulong.MaxValue : uint.MaxValue);
        double? seconds = timescale > 0 && duration > 0 && !unknown ? duration / (double)timescale : null;

        return facts with { RecordedAt = at ?? facts.RecordedAt, DurationSeconds = seconds ?? facts.DurationSeconds };
    }

    /// <summary>udta / meta / ilst / trkn / data: two reserved bytes, then the track number.</summary>
    private static int? TrackNumber(ReadOnlySpan<byte> udta)
    {
        var bytes = udta.ToArray();

        foreach (var (type, at, size) in Boxes(bytes, 0, bytes.Length))
        {
            // meta is a full box: four bytes of version and flags before its children.
            if (type == "meta" && size > 4)
            {
                foreach (var (list, listAt, listSize) in Boxes(bytes, at + 4, size - 4))
                {
                    if (list != "ilst")
                    {
                        continue;
                    }

                    foreach (var (item, itemAt, itemSize) in Boxes(bytes, listAt, listSize))
                    {
                        if (item != "trkn")
                        {
                            continue;
                        }

                        foreach (var (data, dataAt, dataSize) in Boxes(bytes, itemAt, itemSize))
                        {
                            // data: type (4), locale (4), then reserved (2) and the number (2).
                            if (data == "data" && dataSize >= 12)
                            {
                                var number = BinaryPrimitives.ReadUInt16BigEndian(bytes.AsSpan(dataAt + 10));
                                return number > 0 ? number : null;
                            }
                        }
                    }
                }
            }
        }

        return null;
    }

    private static IEnumerable<(string Type, long Start, long Length)> Boxes(Stream stream, long from, long end)
    {
        var header = new byte[16];
        var at = from;

        while (at + 8 <= end)
        {
            stream.Position = at;
            stream.ReadExactly(header, 0, 8);

            long size = BinaryPrimitives.ReadUInt32BigEndian(header);
            var type = Encoding.ASCII.GetString(header, 4, 4);
            var headerLength = 8L;

            if (size == 1)
            {
                stream.ReadExactly(header, 8, 8);
                size = (long)BinaryPrimitives.ReadUInt64BigEndian(header.AsSpan(8));
                headerLength = 16;
            }
            else if (size == 0)
            {
                size = end - at;
            }

            // Subtracted rather than added: a 64-bit size near the maximum wraps the sum negative
            // and passes, then walks the cursor backwards forever.
            if (size < headerLength || size > end - at)
            {
                yield break;
            }

            yield return (type, at + headerLength, size - headerLength);
            at += size;
        }
    }

    private static IEnumerable<(string Type, int Start, int Length)> Boxes(byte[] buffer, int from, int length)
    {
        var at = from;
        var end = from + length;

        while (at + 8 <= end)
        {
            var size = (long)BinaryPrimitives.ReadUInt32BigEndian(buffer.AsSpan(at));
            var type = Encoding.ASCII.GetString(buffer, at + 4, 4);
            var headerLength = 8;

            if (size == 1 && at + 16 <= end)
            {
                size = (long)BinaryPrimitives.ReadUInt64BigEndian(buffer.AsSpan(at + 8));
                headerLength = 16;
            }
            else if (size == 0)
            {
                size = end - at;
            }

            if (size < headerLength || size > end - at)
            {
                yield break;
            }

            yield return (type, at + headerLength, (int)size - headerLength);
            at += (int)size;
        }
    }

    /* ---- WAV ---------------------------------------------------------------------------- */

    private static RecordingFacts Wave(Stream stream, RecordingFacts known)
    {
        var facts = known;
        var header = new byte[8];
        long at = 12;
        uint byteRate = 0;
        long? dataBytes = null;

        while (at + 8 <= stream.Length)
        {
            stream.Position = at;
            stream.ReadExactly(header, 0, 8);
            var id = Encoding.ASCII.GetString(header, 0, 4);
            long size = BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(4));
            var body = at + 8;

            // A chunk that runs past the end is a file cut short, or a recorder still writing:
            // what was read before it stands. Only the audio may run on, and is cut to the file.
            if (id != "data" && size > stream.Length - body)
            {
                break;
            }

            switch (id)
            {
                case "fmt " when size >= 16:
                    byteRate = BinaryPrimitives.ReadUInt32LittleEndian(ReadExactly(stream, body, 16).AsSpan(8));
                    break;

                case "data" when size is 0 or uint.MaxValue:
                    // A recorder still writing, or one that crashed, leaves the size unset. The
                    // rest of the file is audio, and walking it as chunks would crawl through
                    // silence eight bytes at a time.
                    dataBytes = stream.Length - body;
                    at = stream.Length;
                    continue;

                case "data":
                    dataBytes = Math.Min(size, stream.Length - body);
                    break;

                case "bext" when size >= 346:
                    var bext = ReadExactly(stream, body, 346);
                    facts = facts with { RecordedAt = BextTime(bext) ?? facts.RecordedAt };
                    break;

                case "LIST" when size >= 4 && size <= LargestHeaderBytes:
                    facts = InfoList(ReadExactly(stream, body, size), facts);
                    break;
            }

            // Chunks are padded to an even length.
            at = body + size + (size % 2);
        }

        return byteRate > 0 && dataBytes is { } bytes
            ? facts with { DurationSeconds = bytes / (double)byteRate }
            : facts;
    }

    /// <summary>
    /// OriginationDate (10 characters, at 320) and OriginationTime (8, at 330). The standard says
    /// yyyy-mm-dd and hh:mm:ss but allows any separator, and recorders use several. Local time:
    /// the field has no zone, and it is the recorder's clock.
    /// </summary>
    private static DateTimeOffset? BextTime(byte[] bext)
    {
        var date = Encoding.ASCII.GetString(bext, 320, 10);
        var time = Encoding.ASCII.GetString(bext, 330, 8);

        if (!Digits(date, 0, 4, out var year) || !Digits(date, 5, 2, out var month) || !Digits(date, 8, 2, out var day))
        {
            return null;
        }

        Digits(time, 0, 2, out var hour);
        Digits(time, 3, 2, out var minute);
        Digits(time, 6, 2, out var second);

        try
        {
            var local = new DateTime(year, month, day, hour, minute, second, DateTimeKind.Local);
            return new DateTimeOffset(local);
        }
        catch (ArgumentOutOfRangeException)
        {
            return null;
        }
    }

    private static bool Digits(string text, int at, int count, out int value)
    {
        value = 0;
        return at + count <= text.Length
            && int.TryParse(text.AsSpan(at, count), NumberStyles.None, CultureInfo.InvariantCulture, out value);
    }

    /// <summary>LIST/INFO: ICRD is the creation date, ITRK or IPRT a track or part number.</summary>
    private static RecordingFacts InfoList(byte[] list, RecordingFacts facts)
    {
        if (Encoding.ASCII.GetString(list, 0, 4) != "INFO")
        {
            return facts;
        }

        var at = 4;

        while (at + 8 <= list.Length)
        {
            var id = Encoding.ASCII.GetString(list, at, 4);
            long declared = BinaryPrimitives.ReadUInt32LittleEndian(list.AsSpan(at + 4));

            if (declared > list.Length - at - 8)
            {
                break;
            }

            var size = (int)declared;

            var value = Encoding.UTF8.GetString(list, at + 8, size).TrimEnd('\0', ' ');

            if (id == "ICRD" && facts.RecordedAt is null && When(value) is { } created)
            {
                facts = facts with { RecordedAt = created };
            }
            else if (id is "ITRK" or "IPRT" && Number(value) is { } track)
            {
                facts = facts with { Track = track };
            }

            at += 8 + size + (size % 2);
        }

        return facts;
    }

    /* ---- MP3 (ID3v2) -------------------------------------------------------------------- */

    private static RecordingFacts Id3(Stream stream, RecordingFacts known)
    {
        var header = ReadExactly(stream, 0, 10);
        var version = header[3];
        var size = Synchsafe(header.AsSpan(6));

        if (size > LargestHeaderBytes)
        {
            return known;
        }

        var flags = header[5];
        var tag = ReadExactly(stream, 10, Math.Min(size, stream.Length - 10));

        // Before 2.4, unsynchronisation applies to the whole tag: every 0xFF 0x00 was written for
        // a 0xFF, so the frames' sizes count the bytes after it is undone.
        if (version < 4 && (flags & 0x80) != 0)
        {
            tag = Resynchronise(tag);
        }

        var frames = new Dictionary<string, string>(StringComparer.Ordinal);
        var at = 0;
        var idLength = version == 2 ? 3 : 4;
        var headerLength = version == 2 ? 6 : 10;

        // An extended header sits before the first frame, saying its own size: in 2.3 the bytes
        // after the size field, in 2.4 a synchsafe count that includes it.
        if (version >= 3 && (flags & 0x40) != 0 && tag.Length >= 4)
        {
            at = version == 4
                ? (int)Math.Min(Synchsafe(tag), tag.Length)
                : (int)Math.Min(BinaryPrimitives.ReadUInt32BigEndian(tag) + 4L, tag.Length);
        }

        while (at >= 0 && at <= tag.Length - headerLength && tag[at] != 0)
        {
            var id = Encoding.ASCII.GetString(tag, at, idLength);
            long frameSize = version switch
            {
                2 => (tag[at + 3] << 16) | (tag[at + 4] << 8) | tag[at + 5],
                4 => Synchsafe(tag.AsSpan(at + 4)),
                _ => BinaryPrimitives.ReadUInt32BigEndian(tag.AsSpan(at + 4)),
            };

            // Subtracted rather than added, so a size near the limit cannot wrap past the check.
            if (frameSize <= 0 || frameSize > tag.Length - at - headerLength)
            {
                break;
            }

            if (id[0] == 'T')
            {
                var body = tag.AsSpan(at + headerLength, (int)frameSize);
                var readable = true;

                if (version == 3)
                {
                    // Compressed or encrypted text cannot be read here; a group byte is skipped.
                    var frameFlags = tag[at + 9];
                    readable = (frameFlags & 0xC0) == 0;

                    if ((frameFlags & 0x20) != 0)
                    {
                        body = body.Length >= 1 ? body[1..] : [];
                    }
                }
                else if (version == 4)
                {
                    // 2.4 marks it frame by frame: a group byte, then a four-byte length, before
                    // the data. Unsynchronisation is flagged on the frame, or on the whole tag,
                    // which the standard says applies to every frame.
                    var frameFlags = tag[at + 9];
                    readable = (frameFlags & 0x0C) == 0;

                    if ((frameFlags & 0x40) != 0)
                    {
                        body = body.Length >= 1 ? body[1..] : [];
                    }

                    if ((frameFlags & 0x01) != 0)
                    {
                        body = body.Length >= 4 ? body[4..] : [];
                    }

                    if ((frameFlags & 0x02) != 0 || (flags & 0x80) != 0)
                    {
                        body = Resynchronise(body.ToArray());
                    }
                }

                if (readable)
                {
                    frames[id] = Id3Text(body);
                }
            }

            at += headerLength + (int)frameSize;
        }

        var facts = known;

        // 2.4 writes the whole moment in TDRC; 2.3 splits it into year, DDMM and HHMM.
        if (frames.TryGetValue("TDRC", out var recorded) && When(recorded) is { } at4)
        {
            facts = facts with { RecordedAt = at4 };
        }
        else if ((frames.GetValueOrDefault("TYER") ?? frames.GetValueOrDefault("TYE")) is { Length: 4 } year
            && (frames.GetValueOrDefault("TDAT") ?? frames.GetValueOrDefault("TDA")) is { Length: 4 } dayMonth
            && (frames.GetValueOrDefault("TIME") ?? frames.GetValueOrDefault("TIM")) is { Length: 4 } hhmm)
        {
            // Only with the time: a date alone ties every part recorded that day.
            facts = facts with { RecordedAt = When($"{year}-{dayMonth[2..]}-{dayMonth[..2]} {hhmm[..2]}:{hhmm[2..]}") ?? facts.RecordedAt };
        }

        if ((frames.GetValueOrDefault("TRCK") ?? frames.GetValueOrDefault("TRK")) is { } track && Number(track) is { } number)
        {
            facts = facts with { Track = number };
        }

        return facts;
    }

    /// <summary>Undoes unsynchronisation: every 0xFF 0x00 was written for a single 0xFF.</summary>
    private static byte[] Resynchronise(byte[] bytes)
    {
        var restored = new List<byte>(bytes.Length);

        for (var i = 0; i < bytes.Length; i++)
        {
            restored.Add(bytes[i]);

            if (bytes[i] == 0xFF && i + 1 < bytes.Length && bytes[i + 1] == 0x00)
            {
                i++;
            }
        }

        return [.. restored];
    }

    private static long Synchsafe(ReadOnlySpan<byte> bytes) =>
        ((long)(bytes[0] & 0x7F) << 21) | ((long)(bytes[1] & 0x7F) << 14) | ((long)(bytes[2] & 0x7F) << 7) | (long)(bytes[3] & 0x7F);

    private static string Id3Text(ReadOnlySpan<byte> frame)
    {
        if (frame.Length < 2)
        {
            return string.Empty;
        }

        var text = frame[0] switch
        {
            1 => Encoding.Unicode.GetString(frame[1..]).TrimStart('﻿', '￾'),
            2 => Encoding.BigEndianUnicode.GetString(frame[1..]),
            3 => Encoding.UTF8.GetString(frame[1..]),
            _ => Encoding.Latin1.GetString(frame[1..]),
        };

        return text.Trim('\0', ' ');
    }

    /* ---- FLAC --------------------------------------------------------------------------- */

    private static RecordingFacts Flac(Stream stream, RecordingFacts known)
    {
        var facts = known;
        long at = 4;
        var header = new byte[4];

        while (at + 4 <= stream.Length)
        {
            stream.Position = at;
            stream.ReadExactly(header, 0, 4);
            var last = (header[0] & 0x80) != 0;
            var type = header[0] & 0x7F;
            var length = (header[1] << 16) | (header[2] << 8) | header[3];

            if (length > stream.Length - at - 4)
            {
                break;
            }

            if (type == 0 && length >= 18)
            {
                // STREAMINFO: 20 bits of sample rate at byte 10, 36 bits of total samples at 13.
                var info = ReadExactly(stream, at + 4, 18);
                var rate = (info[10] << 12) | (info[11] << 4) | (info[12] >> 4);
                var samples = ((long)(info[13] & 0x0F) << 32) | BinaryPrimitives.ReadUInt32BigEndian(info.AsSpan(14));

                if (rate > 0 && samples > 0)
                {
                    facts = facts with { DurationSeconds = samples / (double)rate };
                }
            }
            else if (type == 4 && length <= LargestHeaderBytes)
            {
                facts = VorbisComments(ReadExactly(stream, at + 4, length), facts);
            }

            if (last)
            {
                break;
            }

            at += 4 + length;
        }

        return facts;
    }

    private static RecordingFacts VorbisComments(byte[] block, RecordingFacts facts)
    {
        // Little-endian, unlike the rest of FLAC: vendor string, then a count of KEY=value lines.
        // Every length is checked against what is left, by subtraction, so one that is wrong
        // ends the list with what was read so far.
        if (block.Length < 4)
        {
            return facts;
        }

        long vendor = BinaryPrimitives.ReadUInt32LittleEndian(block);
        if (vendor > block.Length - 8)
        {
            return facts;
        }

        var at = 4 + (int)vendor;
        var count = BinaryPrimitives.ReadUInt32LittleEndian(block.AsSpan(at));
        at += 4;

        for (var i = 0; i < count && block.Length - at >= 4; i++)
        {
            long length = BinaryPrimitives.ReadUInt32LittleEndian(block.AsSpan(at));
            if (length > block.Length - at - 4)
            {
                break;
            }

            var line = Encoding.UTF8.GetString(block, at + 4, (int)length);
            at += 4 + (int)length;

            var equals = line.IndexOf('=');
            if (equals <= 0)
            {
                continue;
            }

            var key = line[..equals].ToUpperInvariant();
            var value = line[(equals + 1)..];

            if (key == "DATE" && When(value) is { } date)
            {
                facts = facts with { RecordedAt = date };
            }
            else if (key == "TRACKNUMBER" && Number(value) is { } track)
            {
                facts = facts with { Track = track };
            }
        }

        return facts;
    }

    /* ---- shared ------------------------------------------------------------------------- */

    /// <summary>
    /// A written date and time, or null.
    /// <para>
    /// A time of day is required: a year, or a date alone, is not when recording began, and
    /// two parts of one recording would tie on it, which reads as two recordings of the same
    /// moment. And the date must be written year first, as the tag standards say: a parser
    /// guessing at 05/12/2026 reads May where whoever wrote it may have meant December, and a
    /// recording time outranks every other clue.
    /// </para>
    /// </summary>
    private static DateTimeOffset? When(string text)
    {
        text = text.Trim();

        if (!YearFirstWithTime().IsMatch(text))
        {
            return null;
        }

        return DateTimeOffset.TryParse(text, CultureInfo.InvariantCulture, DateTimeStyles.AssumeLocal, out var value)
            ? value
            : null;
    }

    [System.Text.RegularExpressions.GeneratedRegex(@"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")]
    private static partial System.Text.RegularExpressions.Regex YearFirstWithTime();

    /// <summary>"3", "3/10", "03": the part number before any total.</summary>
    private static int? Number(string text)
    {
        var digits = new string([.. text.Trim().TakeWhile(char.IsAsciiDigit)]);
        return int.TryParse(digits, NumberStyles.None, CultureInfo.InvariantCulture, out var value) && value > 0 ? value : null;
    }

    private static byte[] ReadExactly(Stream stream, long at, long count)
    {
        if (count < 0 || count > LargestHeaderBytes || at + count > stream.Length)
        {
            throw new InvalidDataException("A header runs past the end of the file.");
        }

        var buffer = new byte[count];
        stream.Position = at;
        stream.ReadExactly(buffer);
        return buffer;
    }
}
