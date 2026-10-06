using System.Text;
using LocalScribe.Core.Models;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// The NPU cannot run <c>Erf</c>, so the aligner's GELUs are rewritten as one fused operator.
/// The rewrite must replace exactly the function GELU and nothing that only resembles it, and
/// must leave every other byte of the model — its weights above all — as it found them.
/// </summary>
public class GeluRewriteTests
{
    private static readonly double Root2 = Math.Sqrt(2);

    [Fact]
    public void TheExportedShapeBecomesOneGelu()
    {
        // As the MMS export writes it: x · (1 + erf(x / √2)), then halved.
        var model = Model(
            [
                Node("Div", ["x", "root2"], ["d"]),
                Node("Erf", ["d"], ["e"], name: "/layer/Erf"),
                Node("Add", ["e", "one"], ["a"]),
                Node("Mul", ["x", "a"], ["m"]),
                Node("Mul", ["m", "half"], ["y"]),
            ],
            [Scalar("root2", Root2), Scalar("one", 1), Scalar("half", 0.5)]);

        var (fused, rewritten) = GeluRewrite.Rewrite(model);

        Assert.Equal(1, fused);
        var node = Assert.Single(GeluRewrite.Nodes(rewritten));
        Assert.Equal(("Gelu", "com.microsoft"), (node.OpType, node.Domain));
        Assert.Equal(["x"], node.Inputs);
        Assert.Equal(["y"], node.Outputs);
        Assert.Equal("/layer/Gelu", node.Name);
    }

    [Fact]
    public void HalvingFirstIsTheSameFunction()
    {
        // 0.5 · (1 + erf) first, then times x: the other order exporters use.
        var model = Model(
            [
                Node("Mul", ["x", "inverse"], ["d"]),
                Node("Erf", ["d"], ["e"]),
                Node("Add", ["one", "e"], ["a"]),
                Node("Mul", ["a", "half"], ["h"]),
                Node("Mul", ["h", "x"], ["y"]),
            ],
            [Scalar("inverse", 1 / Root2), Scalar("one", 1), Scalar("half", 0.5)]);

        var (fused, rewritten) = GeluRewrite.Rewrite(model);

        Assert.Equal(1, fused);
        Assert.Equal("Gelu", Assert.Single(GeluRewrite.Nodes(rewritten)).OpType);
    }

    [Fact]
    public void HalfPrecisionConstantsAreRead()
    {
        // The aligner's fp16 build stores its constants as two-byte halves.
        var model = Model(
            [
                Node("Div", ["x", "root2"], ["d"]),
                Node("Erf", ["d"], ["e"]),
                Node("Add", ["e", "one"], ["a"]),
                Node("Mul", ["x", "a"], ["m"]),
                Node("Mul", ["m", "half"], ["y"]),
            ],
            [Half("root2", Root2), Half("one", 1), Half("half", 0.5)]);

        Assert.Equal(1, GeluRewrite.Rewrite(model).Fused);
    }

    [Fact]
    public void ConstantNodesCountAsConstants()
    {
        var model = Model(
            [
                ConstantNode("root2", Root2),
                ConstantNode("one", 1),
                ConstantNode("half", 0.5),
                Node("Div", ["x", "root2"], ["d"]),
                Node("Erf", ["d"], ["e"]),
                Node("Add", ["e", "one"], ["a"]),
                Node("Mul", ["x", "a"], ["m"]),
                Node("Mul", ["m", "half"], ["y"]),
            ],
            []);

        var (fused, rewritten) = GeluRewrite.Rewrite(model);

        Assert.Equal(1, fused);
        Assert.Equal(["Constant", "Constant", "Constant", "Gelu"], GeluRewrite.Nodes(rewritten).Select(n => n.OpType));
    }

    [Fact]
    public void ADifferentScaleIsADifferentFunction()
    {
        // x / 2 rather than x / √2: shaped like GELU, and not it.
        var model = Model(
            [
                Node("Div", ["x", "two"], ["d"]),
                Node("Erf", ["d"], ["e"]),
                Node("Add", ["e", "one"], ["a"]),
                Node("Mul", ["x", "a"], ["m"]),
                Node("Mul", ["m", "half"], ["y"]),
            ],
            [Scalar("two", 2), Scalar("one", 1), Scalar("half", 0.5)]);

        var (fused, rewritten) = GeluRewrite.Rewrite(model);

        Assert.Equal(0, fused);
        Assert.Same(model, rewritten);
    }

    [Fact]
    public void AnIntermediateUsedElsewhereIsLeftAlone()
    {
        // Removing the chain would take "a" away from the node that also reads it.
        var model = Model(
            [
                Node("Div", ["x", "root2"], ["d"]),
                Node("Erf", ["d"], ["e"]),
                Node("Add", ["e", "one"], ["a"]),
                Node("Mul", ["x", "a"], ["m"]),
                Node("Mul", ["m", "half"], ["y"]),
                Node("Relu", ["a"], ["z"]),
            ],
            [Scalar("root2", Root2), Scalar("one", 1), Scalar("half", 0.5)]);

        Assert.Equal(0, GeluRewrite.Rewrite(model).Fused);
    }

    [Fact]
    public void EverythingElseIsKeptInPlace()
    {
        var weights = Enumerable.Range(0, 4096).Select(i => (byte)(i * 7)).ToArray();
        var model = Model(
            [
                Node("MatMul", ["input", "w"], ["x"]),
                Node("Div", ["x", "root2"], ["d"]),
                Node("Erf", ["d"], ["e"]),
                Node("Add", ["e", "one"], ["a"]),
                Node("Mul", ["x", "a"], ["m"]),
                Node("Mul", ["m", "half"], ["g"]),
                Node("Add", ["g", "x"], ["output"]),
            ],
            [Scalar("root2", Root2), Scalar("one", 1), Scalar("half", 0.5), Tensor("w", 10, [64, 32], weights)]);

        var (_, rewritten) = GeluRewrite.Rewrite(model);

        // The surrounding nodes keep their order, so the graph stays topologically sorted.
        Assert.Equal(["MatMul", "Gelu", "Add"], GeluRewrite.Nodes(rewritten).Select(n => n.OpType));

        // The weights are carried over byte for byte.
        Assert.True(Contains(rewritten, weights));

        // And the fused operator's domain is declared, once, beside the original.
        Assert.Equal([(string.Empty, 17L), ("com.microsoft", 1L)], GeluRewrite.OpsetImports(rewritten));
    }

    [Fact]
    public void EveryGeluIsReplaced()
    {
        var nodes = new List<byte[]>();
        for (var layer = 0; layer < 3; layer++)
        {
            var x = layer == 0 ? "x" : $"y{layer - 1}";
            nodes.AddRange(
            [
                Node("Div", [x, "root2"], [$"d{layer}"]),
                Node("Erf", [$"d{layer}"], [$"e{layer}"]),
                Node("Add", [$"e{layer}", "one"], [$"a{layer}"]),
                Node("Mul", [x, $"a{layer}"], [$"m{layer}"]),
                Node("Mul", [$"m{layer}", "half"], [$"y{layer}"]),
            ]);
        }

        var model = Model(nodes, [Scalar("root2", Root2), Scalar("one", 1), Scalar("half", 0.5)]);
        var (fused, rewritten) = GeluRewrite.Rewrite(model);

        Assert.Equal(3, fused);
        Assert.Equal(
            [("x", "y0"), ("y0", "y1"), ("y1", "y2")],
            GeluRewrite.Nodes(rewritten).Select(n => (n.Inputs[0], n.Outputs[0])));
    }

    // A minimal ONNX writer: just the fields the rewrite reads.

    private static byte[] Model(IEnumerable<byte[]> nodes, IEnumerable<byte[]> initializers)
    {
        var graph = new List<byte>();
        foreach (var node in nodes)
        {
            graph.AddRange(Field(1, node));
        }

        graph.AddRange(Field(2, Encoding.UTF8.GetBytes("test")));

        foreach (var initializer in initializers)
        {
            graph.AddRange(Field(5, initializer));
        }

        return
        [
            .. Varint((1 << 3) | 0), .. Varint(8),
            .. Field(7, [.. graph]),
            .. Field(8, [.. Field(1, []), .. Varint((2 << 3) | 0), .. Varint(17)]),
        ];
    }

    private static byte[] Node(string op, string[] inputs, string[] outputs, string name = "")
    {
        var bytes = new List<byte>();
        foreach (var input in inputs) bytes.AddRange(Text(1, input));
        foreach (var output in outputs) bytes.AddRange(Text(2, output));
        if (name.Length > 0) bytes.AddRange(Text(3, name));
        bytes.AddRange(Text(4, op));
        return [.. bytes];
    }

    private static byte[] ConstantNode(string output, double value)
    {
        // The attribute: name "value", tensor t (field 5), type TENSOR (20).
        byte[] attribute = [.. Text(1, "value"), .. Field(5, Scalar(string.Empty, value)), .. Varint((20 << 3) | 0), .. Varint(4)];
        return [.. Text(2, output), .. Text(4, "Constant"), .. Field(5, attribute)];
    }

    private static byte[] Scalar(string name, double value) =>
        Tensor(name, 1, [], BitConverter.GetBytes((float)value));

    private static byte[] Half(string name, double value) =>
        Tensor(name, 10, [], BitConverter.GetBytes((Half)value));

    private static byte[] Tensor(string name, int type, int[] dims, byte[] raw)
    {
        var bytes = new List<byte>();
        foreach (var dim in dims)
        {
            bytes.AddRange(Varint((1 << 3) | 0));
            bytes.AddRange(Varint((ulong)dim));
        }

        bytes.AddRange(Varint((2 << 3) | 0));
        bytes.AddRange(Varint((ulong)type));
        if (name.Length > 0) bytes.AddRange(Text(8, name));
        bytes.AddRange(Field(9, raw));
        return [.. bytes];
    }

    private static byte[] Text(int number, string value) => Field(number, Encoding.UTF8.GetBytes(value));

    private static byte[] Field(int number, byte[] value) =>
        [.. Varint(((ulong)number << 3) | 2), .. Varint((ulong)value.Length), .. value];

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

    private static bool Contains(byte[] haystack, byte[] needle) =>
        haystack.AsSpan().IndexOf(needle) >= 0;
}
