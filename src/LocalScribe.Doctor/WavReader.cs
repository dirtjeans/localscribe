using System.Buffers.Binary;
using LocalScribe.Core.Audio;

namespace LocalScribe.Doctor;

/// <summary>
/// Reads uncompressed PCM WAV files.
/// <para>
/// Deliberately minimal and dependency-free. The app uses NAudio for this, but the doctor
/// targets plain net8.0 so it still builds and runs on the CI machines that have no Windows
/// audio stack, and pulling NAudio in here would end that.
/// </para>
/// <para>
/// It handles what a transcription test actually encounters — 16- and 32-bit PCM, and 32-bit
/// float, mono or multi-channel — and refuses everything else rather than guessing. A
/// mis-parsed header produces noise, and noise transcribes to confident nonsense.
/// </para>
/// </summary>
internal static class WavReader
{
    private const ushort FormatPcm = 1;
    private const ushort FormatFloat = 3;
    private const ushort FormatExtensible = 0xFFFE;

    public static PcmAudio Read(string path)
    {
        var bytes = File.ReadAllBytes(path);

        if (bytes.Length < 44
            || !Matches(bytes, 0, "RIFF")
            || !Matches(bytes, 8, "WAVE"))
        {
            throw new InvalidDataException($"{path} is not a RIFF/WAVE file.");
        }

        ushort channels = 0;
        ushort bitsPerSample = 0;
        ushort format = 0;
        var sampleRate = 0;

        var offset = 12;
        while (offset + 8 <= bytes.Length)
        {
            var chunkId = System.Text.Encoding.ASCII.GetString(bytes, offset, 4);
            var chunkSize = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(offset + 4, 4));
            var body = offset + 8;

            if (chunkId == "fmt ")
            {
                format = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(body, 2));
                channels = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(body + 2, 2));
                sampleRate = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(body + 4, 4));
                bitsPerSample = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(body + 14, 2));

                // WAVE_FORMAT_EXTENSIBLE hides the real format in a sub-GUID whose first two
                // bytes carry the original tag.
                if (format == FormatExtensible && chunkSize >= 40)
                {
                    format = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(body + 24, 2));
                }
            }
            else if (chunkId == "data")
            {
                var length = Math.Min(chunkSize, bytes.Length - body);
                var samples = Decode(bytes.AsSpan(body, length), format, bitsPerSample, channels);

                return new PcmAudio(samples, sampleRate);
            }

            // Chunks are word-aligned, so an odd size carries a pad byte.
            offset = body + chunkSize + (chunkSize % 2);
        }

        throw new InvalidDataException($"{path} has no data chunk.");
    }

    /// <summary>
    /// Converts to mono float in [-1, 1]. Multi-channel audio is averaged rather than having one
    /// channel picked, since a speaker on one side of a stereo recording would otherwise vanish.
    /// </summary>
    private static float[] Decode(ReadOnlySpan<byte> data, ushort format, ushort bits, ushort channels)
    {
        if (channels == 0)
        {
            throw new InvalidDataException("The fmt chunk declares no channels.");
        }

        var bytesPerSample = bits / 8;
        if (bytesPerSample == 0)
        {
            throw new InvalidDataException("The fmt chunk declares no sample width.");
        }

        var frames = data.Length / (bytesPerSample * channels);
        var output = new float[frames];

        for (var frame = 0; frame < frames; frame++)
        {
            double total = 0;

            for (var channel = 0; channel < channels; channel++)
            {
                var at = ((frame * channels) + channel) * bytesPerSample;
                total += (format, bits) switch
                {
                    (FormatPcm, 16) => BinaryPrimitives.ReadInt16LittleEndian(data.Slice(at, 2)) / 32768.0,
                    (FormatPcm, 32) => BinaryPrimitives.ReadInt32LittleEndian(data.Slice(at, 4)) / 2147483648.0,
                    (FormatFloat, 32) => BinaryPrimitives.ReadSingleLittleEndian(data.Slice(at, 4)),
                    _ => throw new NotSupportedException(
                        $"Unsupported WAV format {format} at {bits} bits. Convert to 16-bit PCM."),
                };
            }

            output[frame] = (float)(total / channels);
        }

        return output;
    }

    private static bool Matches(byte[] bytes, int offset, string tag) =>
        System.Text.Encoding.ASCII.GetString(bytes, offset, 4) == tag;
}
