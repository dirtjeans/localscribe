using System.Runtime.InteropServices;
using System.Text;
using static LocalScribe.Core.Models.OnnxWire;

namespace LocalScribe.Core.Models;

/// <summary>
/// Rewrites every <c>MatMul</c> whose weight is stored in the model as <c>Reshape</c>, <c>Gemm</c>
/// with <c>transB=1</c> on the weight stored transposed, <c>Reshape</c> back.
/// <para>
/// The same arithmetic, in a form Core ML can keep compact. Core ML's <c>linear</c> wants its weight
/// as [out, in] and ONNX's MatMul stores it as [in, out], so ONNX Runtime's Core ML provider
/// transposes each one, and the transposed copy is written into the compiled program as text:
/// for the MMS aligner, 600 MB of half-precision weights became a 3.7 GB program that took two to
/// three minutes to read every time the model was opened, cache or not. A Gemm with
/// <c>transB=1</c> takes the weight exactly as stored, so it stays binary: opening took 6 s the
/// first time and under a second after, measured by the doctor's <c>--aligner-coreml</c>.
/// </para>
/// <para>
/// Gemm multiplies matrices, so the activations are flattened to two dimensions around it and
/// given back as [1, frames, out]. That assumes a batch of one and three-dimensional activations,
/// which is what the aligner runs with; it is not a general-purpose rewrite. A weight that
/// something other than these MatMuls also reads is left as it is.
/// </para>
/// </summary>
public static class GemmRewrite
{
    // GraphProto, NodeProto, TensorProto and AttributeProto field numbers.
    private const int ModelGraph = 7;
    private const int GraphNode = 1, GraphInitializer = 5;
    private const int NodeInput = 1, NodeOutput = 2, NodeName = 3, NodeOpType = 4, NodeAttribute = 5, NodeDomain = 7;
    private const int TensorDims = 1, TensorDataType = 2, TensorName = 8, TensorRawData = 9, TensorDataLocation = 14;
    private const int AttributeName = 1, AttributeInt = 3, AttributeType = 20;

    private const int Float = 1, Int64 = 7, Float16 = 10;
    private const int AttributeTypeInt = 2;

    /// <summary>Rewrites a model file, returning how many MatMuls were replaced.</summary>
    /// <remarks>Nothing is written when none matched, so the caller can tell.</remarks>
    public static int RewriteFile(string source, string destination)
    {
        var bytes = File.ReadAllBytes(source);
        var (rewritten, pieces) = RewritePieces(bytes);

        if (rewritten > 0)
        {
            var temporary = destination + ".part";
            using (var stream = File.Create(temporary))
            {
                Write(stream, bytes, pieces);
            }

            File.Move(temporary, destination, overwrite: true);
        }

        return rewritten;
    }

    /// <summary>Rewrites a model held in memory.</summary>
    public static (int Rewritten, byte[] Model) Rewrite(byte[] model)
    {
        ArgumentNullException.ThrowIfNull(model);

        var (rewritten, pieces) = RewritePieces(model);
        if (rewritten == 0)
        {
            return (0, model);
        }

        using var stream = new MemoryStream();
        Write(stream, model, pieces);
        return (rewritten, stream.ToArray());
    }

    /// <summary>One initialiser as the rewrite sees it: enough to transpose it.</summary>
    internal sealed record Weight(string Name, long[] Dims, int Type, int RawStart, int RawLength);

    /// <summary>The model's initialisers, for checking a rewrite.</summary>
    internal static IReadOnlyList<Weight> Weights(byte[] model)
    {
        var graph = Fields(model, 0, model.Length).First(f => f.Number == ModelGraph);
        return [.. Fields(model, graph.ValueStart, graph.ValueEnd)
            .Where(f => f.Number == GraphInitializer)
            .Select(f => ReadWeight(model, f))
            .OfType<Weight>()];
    }

    private sealed record ParsedNode(string OpType, string Domain, string Name, List<string> Inputs, List<string> Outputs);

    private static (int Rewritten, List<Piece> Pieces) RewritePieces(byte[] model)
    {
        var top = Fields(model, 0, model.Length);
        var graph = top.FirstOrDefault(f => f.Number == ModelGraph);
        if (graph.Number != ModelGraph)
        {
            return (0, []);
        }

        var graphFields = Fields(model, graph.ValueStart, graph.ValueEnd);
        var nodes = new Dictionary<int, ParsedNode>();
        var weights = new Dictionary<string, (int Field, Weight Weight)>(StringComparer.Ordinal);
        var readers = new Dictionary<string, List<ParsedNode>>(StringComparer.Ordinal);

        for (var i = 0; i < graphFields.Count; i++)
        {
            var field = graphFields[i];

            if (field.Number == GraphNode)
            {
                var node = ReadNode(model, field);
                nodes[i] = node;

                foreach (var input in node.Inputs)
                {
                    if (!readers.TryGetValue(input, out var list))
                    {
                        readers[input] = list = [];
                    }

                    list.Add(node);
                }
            }
            else if (field.Number == GraphInitializer && ReadWeight(model, field) is { } weight)
            {
                weights[weight.Name] = (i, weight);
            }
        }

        bool Rewritable(ParsedNode node) =>
            node.OpType == "MatMul"
            && node.Domain.Length == 0
            && node.Inputs.Count == 2
            && node.Outputs.Count == 1
            && weights.TryGetValue(node.Inputs[1], out var found)
            && found.Weight is { Dims.Length: 2, Type: Float or Float16 } w
            && w.RawLength == w.Dims[0] * w.Dims[1] * (w.Type == Float16 ? 2 : 4)
            && readers[node.Inputs[1]].All(reader => reader.OpType == "MatMul" && reader.Inputs[1] == node.Inputs[1] && reader.Inputs[0] != node.Inputs[1]);

        var chosen = nodes.Where(pair => Rewritable(pair.Value)).ToDictionary(pair => pair.Key, pair => pair.Value);
        if (chosen.Count == 0)
        {
            return (0, []);
        }

        var transposed = chosen.Values.Select(node => node.Inputs[1]).ToHashSet(StringComparer.Ordinal);
        var transposedFields = transposed.ToDictionary(name => weights[name].Field, name => weights[name].Weight);
        var shapes = new SortedDictionary<string, long[]>(StringComparer.Ordinal);
        var graphPieces = new List<Piece>();

        for (var i = 0; i < graphFields.Count; i++)
        {
            var field = graphFields[i];

            if (chosen.TryGetValue(i, out var node))
            {
                var weight = weights[node.Inputs[1]].Weight;
                var (inner, outer) = (weight.Dims[0], weight.Dims[1]);
                var flat = $"localscribe_gemm_flat_{inner}";
                var back = $"localscribe_gemm_back_{outer}";
                shapes[flat] = [-1, inner];
                shapes[back] = [1, -1, outer];

                var name = node.Name.Length > 0 ? node.Name : node.Outputs[0];
                var a = name + "/flat";
                var product = name + "/gemm";

                graphPieces.Add(Piece.New(LengthDelimited(GraphNode, EncodeNode("Reshape", name + "/Reshape_in", [node.Inputs[0], flat], [a]))));
                graphPieces.Add(Piece.New(LengthDelimited(GraphNode, EncodeNode("Gemm", name + "/Gemm", [a, TransposedName(weight.Name)], [product], transB: true))));
                graphPieces.Add(Piece.New(LengthDelimited(GraphNode, EncodeNode("Reshape", name + "/Reshape_out", [product, back], [node.Outputs[0]]))));
                continue;
            }

            if (transposedFields.TryGetValue(i, out var stored))
            {
                graphPieces.Add(Piece.New(LengthDelimited(GraphInitializer, EncodeTensor(
                    TransposedName(stored.Name),
                    stored.Type,
                    [stored.Dims[1], stored.Dims[0]],
                    Transpose(model, stored)))));
                continue;
            }

            graphPieces.Add(Piece.Of(field.Start, field.End));
        }

        foreach (var (name, values) in shapes)
        {
            var raw = new byte[values.Length * sizeof(long)];
            MemoryMarshal.Cast<long, byte>(values).CopyTo(raw);
            graphPieces.Add(Piece.New(LengthDelimited(GraphInitializer, EncodeTensor(name, Int64, [values.Length], raw))));
        }

        long graphLength = graphPieces.Sum(p => p.Extra?.LongLength ?? p.Length);

        var pieces = new List<Piece>();
        foreach (var field in top)
        {
            if (field.Number == ModelGraph)
            {
                pieces.Add(Piece.New([.. Tag(ModelGraph, 2), .. Varint((ulong)graphLength)]));
                pieces.AddRange(graphPieces);
                continue;
            }

            pieces.Add(Piece.Of(field.Start, field.End));
        }

        return (chosen.Count, pieces);
    }

    // A new name rather than the old one reused: anything that described the weight's shape by
    // name, as value_info may, would otherwise contradict it.
    private static string TransposedName(string name) => name + "/transposed";

    private static byte[] Transpose(byte[] model, Weight weight)
    {
        var (rows, columns) = ((int)weight.Dims[0], (int)weight.Dims[1]);
        var raw = new byte[weight.RawLength];

        if (weight.Type == Float16)
        {
            Transpose(MemoryMarshal.Cast<byte, ushort>(model.AsSpan(weight.RawStart, weight.RawLength)), MemoryMarshal.Cast<byte, ushort>(raw.AsSpan()), rows, columns);
        }
        else
        {
            Transpose(MemoryMarshal.Cast<byte, uint>(model.AsSpan(weight.RawStart, weight.RawLength)), MemoryMarshal.Cast<byte, uint>(raw.AsSpan()), rows, columns);
        }

        return raw;
    }

    private static void Transpose<T>(ReadOnlySpan<T> source, Span<T> destination, int rows, int columns)
    {
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < columns; c++)
            {
                destination[(c * rows) + r] = source[(r * columns) + c];
            }
        }
    }

    private static ParsedNode ReadNode(byte[] model, Field node)
    {
        var inputs = new List<string>();
        var outputs = new List<string>();
        string op = string.Empty, domain = string.Empty, name = string.Empty;

        foreach (var field in Fields(model, node.ValueStart, node.ValueEnd))
        {
            switch (field.Number)
            {
                case NodeInput: inputs.Add(Text(model, field)); break;
                case NodeOutput: outputs.Add(Text(model, field)); break;
                case NodeName: name = Text(model, field); break;
                case NodeOpType: op = Text(model, field); break;
                case NodeDomain: domain = Text(model, field); break;
            }
        }

        return new ParsedNode(op, domain, name, inputs, outputs);
    }

    /// <summary>An initialiser with its bytes inside the model, or null for one stored elsewhere.</summary>
    private static Weight? ReadWeight(byte[] model, Field tensor)
    {
        var dims = new List<long>();
        var name = string.Empty;
        var type = 0;
        Field? raw = null;

        foreach (var field in Fields(model, tensor.ValueStart, tensor.ValueEnd))
        {
            switch (field.Number)
            {
                case TensorDims:
                    if (field.WireType == 0)
                    {
                        dims.Add((long)field.Varint);
                    }
                    else
                    {
                        dims.AddRange(PackedVarints(model, field).Select(d => (long)d));
                    }

                    break;
                case TensorDataType: type = (int)field.Varint; break;
                case TensorName: name = Text(model, field); break;
                case TensorRawData: raw = field; break;
                case TensorDataLocation when field.Varint != 0: return null;
            }
        }

        return raw is { } data ? new Weight(name, [.. dims], type, data.ValueStart, data.Length) : null;
    }

    private static byte[] EncodeTensor(string name, int type, long[] dims, byte[] raw)
    {
        var bytes = new List<byte>(raw.Length + 64);

        foreach (var dim in dims)
        {
            bytes.AddRange(Tag(TensorDims, 0));
            bytes.AddRange(Varint(unchecked((ulong)dim)));
        }

        bytes.AddRange(Tag(TensorDataType, 0));
        bytes.AddRange(Varint((ulong)type));
        bytes.AddRange(LengthDelimited(TensorName, Encoding.UTF8.GetBytes(name)));
        bytes.AddRange(LengthDelimited(TensorRawData, raw));
        return [.. bytes];
    }

    private static byte[] EncodeNode(string op, string name, IReadOnlyList<string> inputs, IReadOnlyList<string> outputs, bool transB = false)
    {
        var bytes = new List<byte>();

        foreach (var input in inputs)
        {
            bytes.AddRange(LengthDelimited(NodeInput, Encoding.UTF8.GetBytes(input)));
        }

        foreach (var output in outputs)
        {
            bytes.AddRange(LengthDelimited(NodeOutput, Encoding.UTF8.GetBytes(output)));
        }

        bytes.AddRange(LengthDelimited(NodeName, Encoding.UTF8.GetBytes(name)));
        bytes.AddRange(LengthDelimited(NodeOpType, Encoding.UTF8.GetBytes(op)));

        if (transB)
        {
            bytes.AddRange(LengthDelimited(NodeAttribute,
            [
                .. LengthDelimited(AttributeName, "transB"u8.ToArray()),
                .. Tag(AttributeInt, 0), .. Varint(1),
                .. Tag(AttributeType, 0), .. Varint(AttributeTypeInt),
            ]));
        }

        return [.. bytes];
    }
}
