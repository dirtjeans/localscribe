using System.Text;

namespace LocalScribe.Core.Models;

/// <summary>
/// Rewrites an ONNX model so every GELU written as <c>x · 0.5 · (1 + erf(x / √2))</c> becomes one
/// <c>com.microsoft</c> <c>Gelu</c> node.
/// <para>
/// The same function, in a form the NPU can run. The Qualcomm provider in the ONNX Runtime this
/// app ships cannot run <c>Erf</c>, and the MMS aligner's export uses it in every one of its 32
/// GELUs, which cut the graph into pieces the NPU would not finalize. ONNX Runtime's own
/// optimiser fuses GELU, but not on a half-precision model, so the rewrite is done here, on the
/// file's bytes: only the node list changes, and everything else — half a gigabyte of weights —
/// is copied through untouched.
/// </para>
/// <para>
/// Written against ONNX's protobuf wire format directly, because Core takes no dependencies and a
/// protobuf library would be one. The format is simple where it is used here: a model holds a
/// graph, a graph holds nodes and initialisers, and every field says how long it is.
/// </para>
/// </summary>
public static class GeluRewrite
{
    /// <summary>One operator as the rewrite sees it.</summary>
    public sealed record Node(string OpType, string Domain, string Name, IReadOnlyList<string> Inputs, IReadOnlyList<string> Outputs);

    private const string MicrosoftDomain = "com.microsoft";

    // ModelProto, GraphProto, NodeProto, TensorProto and AttributeProto field numbers.
    private const int ModelGraph = 7, ModelOpsetImport = 8;
    private const int GraphNode = 1, GraphInitializer = 5;
    private const int NodeInput = 1, NodeOutput = 2, NodeName = 3, NodeOpType = 4, NodeAttribute = 5, NodeDomain = 7;
    private const int TensorDims = 1, TensorDataType = 2, TensorFloatData = 4, TensorInt32Data = 5, TensorName = 8, TensorRawData = 9, TensorDoubleData = 10;
    private const int AttributeName = 1, AttributeTensor = 5;
    private const int OpsetDomain = 1, OpsetVersion = 2;

    private const int Float = 1, Float16 = 10, Double = 11;

    /// <summary>Rewrites a model file, returning how many GELUs were replaced.</summary>
    /// <remarks>Nothing is written when none matched, so the caller can tell.</remarks>
    public static int RewriteFile(string source, string destination)
    {
        var bytes = File.ReadAllBytes(source);
        var (fused, pieces) = RewritePieces(bytes);

        if (fused > 0)
        {
            var temporary = destination + ".part";
            using (var stream = File.Create(temporary))
            {
                Write(stream, bytes, pieces);
            }

            File.Move(temporary, destination, overwrite: true);
        }

        return fused;
    }

    /// <summary>Rewrites a model held in memory.</summary>
    public static (int Fused, byte[] Model) Rewrite(byte[] model)
    {
        var (fused, pieces) = RewritePieces(model);
        if (fused == 0)
        {
            return (0, model);
        }

        using var stream = new MemoryStream();
        Write(stream, model, pieces);
        return (fused, stream.ToArray());
    }

    /// <summary>The graph's nodes, in order, for checking a rewrite.</summary>
    public static IReadOnlyList<Node> Nodes(byte[] model)
    {
        var graph = Fields(model, 0, model.Length).First(f => f.Number == ModelGraph);
        return [.. Fields(model, graph.ValueStart, graph.ValueEnd)
            .Where(f => f.Number == GraphNode)
            .Select(f => ReadNode(model, f))
            .Select(n => new Node(n.OpType, n.Domain, n.Name, n.Inputs, n.Outputs))];
    }

    private static void Write(Stream stream, byte[] source, List<Piece> pieces)
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

    /// <summary>The model's operator-set imports, as (domain, version).</summary>
    public static IReadOnlyList<(string Domain, long Version)> OpsetImports(byte[] model) =>
        [.. Fields(model, 0, model.Length)
            .Where(f => f.Number == ModelOpsetImport)
            .Select(f =>
            {
                var domain = string.Empty;
                long version = 0;
                foreach (var field in Fields(model, f.ValueStart, f.ValueEnd))
                {
                    if (field.Number == OpsetDomain) domain = Text(model, field);
                    if (field.Number == OpsetVersion) version = (long)field.Varint;
                }

                return (domain, version);
            })];

    /// <summary>A slice of the source, or fresh bytes in its place.</summary>
    private readonly record struct Piece(int Start, int Length, byte[]? Extra)
    {
        public static Piece Of(int start, int end) => new(start, end - start, null);

        public static Piece New(byte[] bytes) => new(0, 0, bytes);
    }

    private static (int Fused, List<Piece> Pieces) RewritePieces(byte[] model)
    {
        var top = Fields(model, 0, model.Length);
        var graph = top.FirstOrDefault(f => f.Number == ModelGraph);
        if (graph.Number != ModelGraph)
        {
            return (0, []);
        }

        var graphFields = Fields(model, graph.ValueStart, graph.ValueEnd);
        var nodes = graphFields
            .Select((field, index) => (field, index))
            .Where(pair => pair.field.Number == GraphNode)
            .Select(pair => (pair.index, node: ReadNode(model, pair.field)))
            .ToList();

        var constants = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var field in graphFields.Where(f => f.Number == GraphInitializer))
        {
            if (Scalar(model, field.ValueStart, field.ValueEnd) is { } scalar)
            {
                constants[scalar.Name] = scalar.Value;
            }
        }

        foreach (var (_, node) in nodes.Where(n => n.node.OpType == "Constant" && n.node.Outputs.Count == 1))
        {
            if (node.ConstantValue is { } value)
            {
                constants[node.Outputs[0]] = value;
            }
        }

        var producer = new Dictionary<string, int>(StringComparer.Ordinal);
        var consumers = new Dictionary<string, List<int>>(StringComparer.Ordinal);

        for (var i = 0; i < nodes.Count; i++)
        {
            foreach (var output in nodes[i].node.Outputs)
            {
                producer[output] = i;
            }

            foreach (var input in nodes[i].node.Inputs)
            {
                if (!consumers.TryGetValue(input, out var list))
                {
                    consumers[input] = list = [];
                }

                list.Add(i);
            }
        }

        double? Const(string name) => constants.TryGetValue(name, out var value) ? value : null;
        bool Near(double? value, double target) => value is { } v && Math.Abs(v - target) < 1e-3;
        string Other(ParsedNode node, string name) => node.Inputs[0] == name ? node.Inputs[1] : node.Inputs[0];

        int? Only(string name, string op)
        {
            if (consumers.TryGetValue(name, out var list) && list.Count == 1 && nodes[list[0]].node.OpType == op)
            {
                return list[0];
            }

            return null;
        }

        var removed = new HashSet<int>();
        var replacement = new Dictionary<int, byte[]>();

        for (var e = 0; e < nodes.Count; e++)
        {
            var erf = nodes[e].node;
            if (erf.OpType != "Erf" || erf.Inputs.Count != 1 || !producer.TryGetValue(erf.Inputs[0], out var d))
            {
                continue;
            }

            var div = nodes[d].node;
            if (div.Inputs.Count != 2)
            {
                continue;
            }

            var x = div.Inputs[0];
            var scaled = div.OpType switch
            {
                "Div" => Near(Const(div.Inputs[1]), Math.Sqrt(2)),
                "Mul" => Near(Const(div.Inputs[1]), 1 / Math.Sqrt(2)),
                _ => false,
            };

            if (!scaled || Only(erf.Outputs[0], "Add") is not { } a)
            {
                continue;
            }

            var add = nodes[a].node;
            if (!Near(Const(Other(add, erf.Outputs[0])), 1) || Only(add.Outputs[0], "Mul") is not { } f)
            {
                continue;
            }

            var first = nodes[f].node;
            var partner = Other(first, add.Outputs[0]);
            var second = Only(first.Outputs[0], "Mul");
            var chain = new List<int> { d, e, a, f };
            string? output = null;

            if (partner == x && second is { } s1 && Near(Const(Other(nodes[s1].node, first.Outputs[0])), 0.5))
            {
                chain.Add(s1);
                output = nodes[s1].node.Outputs[0];
            }
            else if (Near(Const(partner), 0.5) && second is { } s2 && Other(nodes[s2].node, first.Outputs[0]) == x)
            {
                chain.Add(s2);
                output = nodes[s2].node.Outputs[0];
            }
            else if (producer.TryGetValue(partner, out var h)
                && nodes[h].node is { OpType: "Mul" } half
                && half.Inputs.Contains(x)
                && Near(Const(Other(half, x)), 0.5)
                && consumers[half.Outputs[0]].Count == 1)
            {
                chain.Add(h);
                output = first.Outputs[0];
            }

            if (output is null || chain.Any(removed.Contains))
            {
                continue;
            }

            foreach (var index in chain)
            {
                removed.Add(index);
            }

            // In the chain's place: where its first node stood, so the order stays topological.
            var name = erf.Name.EndsWith("/Erf", StringComparison.Ordinal) ? erf.Name[..^4] + "/Gelu" : erf.Name + "_gelu";
            replacement[chain.Min()] = EncodeNode("Gelu", MicrosoftDomain, name, [x], [output]);
        }

        if (replacement.Count == 0)
        {
            return (0, []);
        }

        // The graph, rebuilt: nodes in their order with the chains replaced, everything else copied.
        var graphPieces = new List<Piece>();
        var nodeIndexOfField = nodes.Select((n, i) => (n.index, i)).ToDictionary(p => p.index, p => p.i);

        for (var i = 0; i < graphFields.Count; i++)
        {
            var field = graphFields[i];

            if (field.Number == GraphNode && nodeIndexOfField.TryGetValue(i, out var nodeIndex))
            {
                if (replacement.TryGetValue(nodeIndex, out var fresh))
                {
                    graphPieces.Add(Piece.New(LengthDelimited(GraphNode, fresh)));
                    continue;
                }

                if (removed.Contains(nodeIndex))
                {
                    continue;
                }
            }

            graphPieces.Add(Piece.Of(field.Start, field.End));
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

        if (!OpsetImports(model).Any(o => o.Domain == MicrosoftDomain))
        {
            pieces.Add(Piece.New(LengthDelimited(
                ModelOpsetImport,
                [.. LengthDelimited(OpsetDomain, Encoding.UTF8.GetBytes(MicrosoftDomain)), .. Tag(OpsetVersion, 0), .. Varint(1)])));
        }

        return (replacement.Count, pieces);
    }

    private sealed record ParsedNode(
        string OpType,
        string Domain,
        string Name,
        List<string> Inputs,
        List<string> Outputs,
        double? ConstantValue);

    private static ParsedNode ReadNode(byte[] model, Field node)
    {
        var inputs = new List<string>();
        var outputs = new List<string>();
        string op = string.Empty, domain = string.Empty, name = string.Empty;
        double? constant = null;

        foreach (var field in Fields(model, node.ValueStart, node.ValueEnd))
        {
            switch (field.Number)
            {
                case NodeInput: inputs.Add(Text(model, field)); break;
                case NodeOutput: outputs.Add(Text(model, field)); break;
                case NodeName: name = Text(model, field); break;
                case NodeOpType: op = Text(model, field); break;
                case NodeDomain: domain = Text(model, field); break;
                case NodeAttribute:
                    foreach (var attribute in Fields(model, field.ValueStart, field.ValueEnd))
                    {
                        if (attribute.Number == AttributeTensor && Scalar(model, attribute.ValueStart, attribute.ValueEnd) is { } scalar)
                        {
                            constant = scalar.Value;
                        }
                    }

                    break;
            }
        }

        return new ParsedNode(op, domain, name, inputs, outputs, constant);
    }

    /// <summary>A tensor's name and value, when it holds exactly one number of a float type.</summary>
    private static (string Name, double Value)? Scalar(byte[] model, int start, int end)
    {
        string name = string.Empty;
        var type = 0;
        long elements = 1;
        Field? raw = null, floats = null, int32s = null, doubles = null;

        foreach (var field in Fields(model, start, end))
        {
            switch (field.Number)
            {
                case TensorName: name = Text(model, field); break;
                case TensorDataType: type = (int)field.Varint; break;
                case TensorDims:
                    if (field.WireType == 0)
                    {
                        elements *= (long)field.Varint;
                    }
                    else
                    {
                        foreach (var dim in PackedVarints(model, field))
                        {
                            elements *= (long)dim;
                        }
                    }

                    break;
                case TensorRawData: raw = field; break;
                case TensorFloatData: floats = field; break;
                case TensorInt32Data: int32s = field; break;
                case TensorDoubleData: doubles = field; break;
            }
        }

        if (elements != 1)
        {
            return null;
        }

        double? value = type switch
        {
            Float when raw is { Length: 4 } r => BitConverter.ToSingle(model, r.ValueStart),
            Float when floats is { } f && f.WireType == 2 && f.Length == 4 => BitConverter.ToSingle(model, f.ValueStart),
            Float when floats is { } f && f.WireType == 5 => BitConverter.ToSingle(model, f.ValueStart),
            Float16 when raw is { Length: 2 } r => (double)BitConverter.ToHalf(model, r.ValueStart),
            Float16 when int32s is { } i => (double)BitConverter.UInt16BitsToHalf((ushort)FirstVarint(model, i)),
            Double when raw is { Length: 8 } r => BitConverter.ToDouble(model, r.ValueStart),
            Double when doubles is { } dd && dd.WireType == 2 && dd.Length == 8 => BitConverter.ToDouble(model, dd.ValueStart),
            _ => null,
        };

        return value is { } v ? (name, v) : null;
    }

    private readonly record struct Field(int Number, int WireType, int Start, int End, int ValueStart, int ValueEnd, ulong Varint)
    {
        public int Length => ValueEnd - ValueStart;
    }

    /// <summary>The fields of one message, with where each sits in the buffer.</summary>
    private static List<Field> Fields(byte[] buffer, int start, int end)
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

    private static ulong ReadVarint(byte[] buffer, ref int at)
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

    private static IEnumerable<ulong> PackedVarints(byte[] buffer, Field field)
    {
        var at = field.ValueStart;
        var values = new List<ulong>();
        while (at < field.ValueEnd)
        {
            values.Add(ReadVarint(buffer, ref at));
        }

        return values;
    }

    private static ulong FirstVarint(byte[] buffer, Field field)
    {
        if (field.WireType == 0)
        {
            return field.Varint;
        }

        var at = field.ValueStart;
        return ReadVarint(buffer, ref at);
    }

    private static string Text(byte[] buffer, Field field) =>
        Encoding.UTF8.GetString(buffer, field.ValueStart, field.Length);

    private static byte[] EncodeNode(string op, string domain, string name, IReadOnlyList<string> inputs, IReadOnlyList<string> outputs)
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
        bytes.AddRange(LengthDelimited(NodeDomain, Encoding.UTF8.GetBytes(domain)));
        return [.. bytes];
    }

    private static byte[] LengthDelimited(int number, byte[] value) =>
        [.. Tag(number, 2), .. Varint((ulong)value.Length), .. value];

    private static byte[] Tag(int number, int wire) => Varint(((ulong)number << 3) | (uint)wire);

    private static byte[] Varint(ulong value)
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
