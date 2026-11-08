using System.Runtime.InteropServices;
using System.Text;
using LocalScribe.Core.Models;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// Core ML stores a MatMul's weight as text, but a Gemm's as binary, so the aligner's MatMuls are
/// rewritten as Gemms on the weight transposed. The arithmetic must be exactly the same, which
/// here means the weight is transposed element for element and the graph still computes x·W,
/// and nothing a MatMul does not own may change.
/// </summary>
public class GemmRewriteTests
{
    [Fact]
    public void AMatMulOnAStoredWeightBecomesAGemmBetweenTwoReshapes()
    {
        var model = Model([Node("MatMul", ["x", "w"], ["y"], name: "/proj/MatMul")], [Weight("w", 3, 2)]);

        var (rewritten, result) = GemmRewrite.Rewrite(model);

        Assert.Equal(1, rewritten);
        var nodes = GeluRewrite.Nodes(result);
        Assert.Equal(["Reshape", "Gemm", "Reshape"], nodes.Select(n => n.OpType));
        Assert.Equal("x", nodes[0].Inputs[0]);
        Assert.Equal(nodes[0].Outputs[0], nodes[1].Inputs[0]);
        Assert.Equal(nodes[1].Outputs[0], nodes[2].Inputs[0]);
        Assert.Equal(["y"], nodes[2].Outputs);
    }

    [Fact]
    public void TheGemmReadsTheWeightTransposedAndSaysSo()
    {
        var model = Model([Node("MatMul", ["x", "w"], ["y"])], [Weight("w", 3, 2)]);

        var (_, result) = GemmRewrite.Rewrite(model);
        var gemm = GeluRewrite.Nodes(result)[1];

        Assert.Contains("w/transposed", gemm.Inputs);

        // transB = 1, as the attribute is written: without it the Gemm would compute x·Wᵀ.
        byte[] transB = [.. Encoding.UTF8.GetBytes("transB"), 0x18, 0x01];
        Assert.True(result.AsSpan().IndexOf(transB) >= 0);
    }

    [Fact]
    public void EveryElementLandsWhereTheTransposeSays()
    {
        // 3 × 2, values 0..5 in row-major order: [[0, 1], [2, 3], [4, 5]].
        var model = Model([Node("MatMul", ["x", "w"], ["y"])], [Weight("w", 3, 2)]);

        var (_, result) = GemmRewrite.Rewrite(model);
        var transposed = GemmRewrite.Weights(result).Single(w => w.Name == "w/transposed");
        var values = MemoryMarshal.Cast<byte, Half>(result.AsSpan(transposed.RawStart, transposed.RawLength)).ToArray();

        Assert.Equal([2L, 3L], transposed.Dims);
        Assert.Equal([0, 2, 4, 1, 3, 5], values.Select(v => (int)v));
    }

    [Fact]
    public void SinglePrecisionWeightsTransposeToo()
    {
        var model = Model([Node("MatMul", ["x", "w"], ["y"])], [Weight("w", 2, 3, single: true)]);

        var (_, result) = GemmRewrite.Rewrite(model);
        var transposed = GemmRewrite.Weights(result).Single(w => w.Name == "w/transposed");
        var values = MemoryMarshal.Cast<byte, float>(result.AsSpan(transposed.RawStart, transposed.RawLength)).ToArray();

        Assert.Equal([0f, 3f, 1f, 4f, 2f, 5f], values);
    }

    [Fact]
    public void TheShapesFlattenToTheWeightsInnerSizeAndRestoreItsOuter()
    {
        var model = Model([Node("MatMul", ["x", "w"], ["y"])], [Weight("w", 3, 2)]);

        var (_, result) = GemmRewrite.Rewrite(model);
        var nodes = GeluRewrite.Nodes(result);

        Assert.Equal([-1L, 3L], Longs(result, nodes[0].Inputs[1]));
        Assert.Equal([1L, -1L, 2L], Longs(result, nodes[2].Inputs[1]));
    }

    [Fact]
    public void TheOriginalWeightIsGoneAndNoMatMulIsLeftToRewrite()
    {
        var model = Model([Node("MatMul", ["x", "w"], ["y"])], [Weight("w", 3, 2)]);

        var (_, result) = GemmRewrite.Rewrite(model);

        Assert.DoesNotContain(GemmRewrite.Weights(result), w => w.Name == "w");
        Assert.Equal(0, GemmRewrite.Rewrite(result).Rewritten);
    }

    [Fact]
    public void AWeightSomethingElseReadsIsLeftAlone()
    {
        var model = Model(
            [Node("MatMul", ["x", "w"], ["y"]), Node("Add", ["w", "w"], ["z"])],
            [Weight("w", 3, 2)]);

        Assert.Equal(0, GemmRewrite.Rewrite(model).Rewritten);
    }

    [Fact]
    public void AMatMulOfTwoActivationsIsLeftAlone()
    {
        var model = Model([Node("MatMul", ["q", "k"], ["scores"])], []);

        Assert.Equal(0, GemmRewrite.Rewrite(model).Rewritten);
    }

    [Fact]
    public void AWeightThatIsNotAMatrixIsLeftAlone()
    {
        var model = Model([Node("MatMul", ["x", "w"], ["y"])], [Tensor("w", 10, [2, 2, 2], new byte[16])]);

        Assert.Equal(0, GemmRewrite.Rewrite(model).Rewritten);
    }

    [Fact]
    public void TwoMatMulsSharingAWeightShareOneTransposedCopy()
    {
        var model = Model(
            [Node("MatMul", ["a", "w"], ["y1"]), Node("MatMul", ["b", "w"], ["y2"])],
            [Weight("w", 3, 2)]);

        var (rewritten, result) = GemmRewrite.Rewrite(model);

        Assert.Equal(2, rewritten);
        Assert.Single(GemmRewrite.Weights(result), w => w.Name == "w/transposed");
    }

    [Fact]
    public void EverythingElseIsCopiedThrough()
    {
        var model = Model(
            [Node("Relu", ["in"], ["x"]), Node("MatMul", ["x", "w"], ["y"]), Node("Softmax", ["y"], ["out"])],
            [Weight("w", 3, 2), Tensor("bias", 1, [2], BitConverter.GetBytes(7f).Concat(BitConverter.GetBytes(9f)).ToArray())]);

        var (_, result) = GemmRewrite.Rewrite(model);
        var nodes = GeluRewrite.Nodes(result);

        Assert.Equal(["Relu", "Reshape", "Gemm", "Reshape", "Softmax"], nodes.Select(n => n.OpType));
        Assert.Contains(GemmRewrite.Weights(result), w => w.Name == "bias");
        Assert.Equal(GeluRewrite.OpsetImports(model), GeluRewrite.OpsetImports(result));
    }

    private static long[] Longs(byte[] model, string name)
    {
        var weight = GemmRewrite.Weights(model).Single(w => w.Name == name);
        return MemoryMarshal.Cast<byte, long>(model.AsSpan(weight.RawStart, weight.RawLength)).ToArray();
    }

    // A minimal ONNX writer: just the fields the rewrite reads.

    private static byte[] Weight(string name, int rows, int columns, bool single = false)
    {
        var values = Enumerable.Range(0, rows * columns);
        var raw = single
            ? values.SelectMany(v => BitConverter.GetBytes((float)v)).ToArray()
            : values.SelectMany(v => BitConverter.GetBytes((Half)v)).ToArray();

        return Tensor(name, single ? 1 : 10, [rows, columns], raw);
    }

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
            .. Field(8, [.. Field(1, []), .. Varint((2 << 3) | 0), .. Varint(14)]),
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
}
