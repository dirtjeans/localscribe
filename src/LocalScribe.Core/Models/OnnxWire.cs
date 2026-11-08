using System.Text;

namespace LocalScribe.Core.Models;

/// <summary>
/// ONNX's protobuf wire format, as far as the model rewrites here need it: read a message's
/// fields with where each sits, copy the ones left alone byte for byte, and write fresh ones.
/// Written directly because Core takes no dependencies, and a protobuf library would be one.
/// </summary>
internal static class OnnxWire
{
    /// <summary>A slice of the source, or fresh bytes in its place.</summary>
    internal readonly record struct Piece(int Start, int Length, byte[]? Extra)
    {
        public static Piece Of(int start, int end) => new(start, end - start, null);

        public static Piece New(byte[] bytes) => new(0, 0, bytes);
    }

    internal static void Write(Stream stream, byte[] source, List<Piece> pieces)
    {
        foreach (var piece in pieces)
        {
            if (piece.Extra is { } fresh)
            {
                stream.Write(fresh);
            }
            else
            {
                stream.Write(source, piece.Start, piece.Length);
            }
        }
    }

    internal readonly record struct Field(int Number, int WireType, int Start, int End, int ValueStart, int ValueEnd, ulong Varint)
    {
        public int Length => ValueEnd - ValueStart;
    }

    /// <summary>The fields of one message, with where each sits in the buffer.</summary>
    internal static List<Field> Fields(byte[] buffer, int start, int end)
    {
        var fields = new List<Field>();
        var at = start;

        while (at < end)
        {
            var fieldStart = at;
            var tag = ReadVarint(buffer, ref at);
            var number = (int)(tag >> 3);
            var wire = (int)(tag & 7);

            switch (wire)
            {
                case 0:
                    var valueStart0 = at;
                    var value = ReadVarint(buffer, ref at);
                    fields.Add(new Field(number, wire, fieldStart, at, valueStart0, at, value));
                    break;
                case 1:
                    fields.Add(new Field(number, wire, fieldStart, at + 8, at, at + 8, 0));
                    at += 8;
                    break;
                case 2:
                    var length = (long)ReadVarint(buffer, ref at);
                    if (length < 0 || at + length > end)
                    {
                        throw new InvalidDataException("Not an ONNX model: a field runs past the end of its message.");
                    }

                    fields.Add(new Field(number, wire, fieldStart, at + (int)length, at, at + (int)length, 0));
                    at += (int)length;
                    break;
                case 5:
                    fields.Add(new Field(number, wire, fieldStart, at + 4, at, at + 4, 0));
                    at += 4;
                    break;
                default:
                    throw new InvalidDataException($"Not an ONNX model: unknown wire type {wire}.");
            }
        }

        return fields;
    }

    internal static ulong ReadVarint(byte[] buffer, ref int at)
    {
        ulong value = 0;
        var shift = 0;

        while (true)
        {
            if (at >= buffer.Length || shift > 63)
            {
                throw new InvalidDataException("Not an ONNX model: a number runs past the end.");
            }

            var b = buffer[at++];
            value |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0)
            {
                return value;
            }

            shift += 7;
        }
    }

    internal static IEnumerable<ulong> PackedVarints(byte[] buffer, Field field)
    {
        var at = field.ValueStart;
        var values = new List<ulong>();
        while (at < field.ValueEnd)
        {
            values.Add(ReadVarint(buffer, ref at));
        }

        return values;
    }

    internal static ulong FirstVarint(byte[] buffer, Field field)
    {
        if (field.WireType == 0)
        {
            return field.Varint;
        }

        var at = field.ValueStart;
        return ReadVarint(buffer, ref at);
    }

    internal static string Text(byte[] buffer, Field field) =>
        Encoding.UTF8.GetString(buffer, field.ValueStart, field.Length);

    internal static byte[] LengthDelimited(int number, byte[] value) =>
        [.. Tag(number, 2), .. Varint((ulong)value.Length), .. value];

    internal static byte[] Tag(int number, int wire) => Varint(((ulong)number << 3) | (uint)wire);

    internal static byte[] Varint(ulong value)
    {
        var bytes = new List<byte>();
        do
        {
            var b = (byte)(value & 0x7F);
            value >>= 7;
            bytes.Add(value != 0 ? (byte)(b | 0x80) : b);
        }
        while (value != 0);

        return [.. bytes];
    }
}
