using LocalScribe.Core.Hardware;
using LocalScribe.Core.Models;
using Xunit;

namespace LocalScribe.Core.Tests;

public class ModelLayoutTests
{
    [Theory]
    [InlineData(SocFamily.SnapdragonXElite, "snapdragon-x-elite")]
    [InlineData(SocFamily.SnapdragonXPlus, "snapdragon-x-plus")]
    [InlineData(SocFamily.SnapdragonX2, "snapdragon-x2")]
    public void ChipsetFolderNamesTheChip(SocFamily family, string expected) =>
        Assert.Equal(expected, ModelLayout.ChipsetFolder(family));

    [Theory]
    [InlineData(SocFamily.NonQualcomm)]
    [InlineData(SocFamily.Unknown)]
    public void MachinesWithoutAnNpuFallThroughToPortable(SocFamily family) =>
        Assert.Equal(ModelLayout.PortableFolder, ModelLayout.ChipsetFolder(family));

    [Fact]
    public void NpuWorkOpensTheChipsetBinaries()
    {
        var path = ModelLayout.Resolve("root", SocFamily.SnapdragonXElite, ComputeDevice.Npu, "small.en");

        Assert.Equal(Path.Combine("root", "snapdragon-x-elite", "small.en"), path);
    }

    /// <summary>
    /// The case that motivates keying on the device rather than the chip. A Snapdragon whose
    /// encoder landed on the CPU has no chipset binaries to open, and looking for them there
    /// would fail on the machine the app was written for.
    /// </summary>
    [Theory]
    [InlineData(ComputeDevice.Cpu)]
    [InlineData(ComputeDevice.Gpu)]
    public void AnythingButTheNpuOpensThePortableExport(ComputeDevice device)
    {
        var path = ModelLayout.Resolve("root", SocFamily.SnapdragonXElite, device, "base.en");

        Assert.Equal(Path.Combine("root", ModelLayout.PortableFolder, "base.en"), path);
    }

    [Fact]
    public void PortableAndChipsetFoldersNeverCollideOnAQualcommPart()
    {
        // If these were ever the same directory, fetched portable weights would satisfy the
        // NPU asset check and the planner would send the encoder somewhere it cannot run.
        Assert.NotEqual(
            ModelLayout.PortableFolder,
            ModelLayout.ChipsetFolder(SocFamily.SnapdragonXElite));
    }

    private static string TempRoot()
    {
        var root = Path.Combine(Path.GetTempPath(), "localscribe-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        return root;
    }

    private static void PlacePortable(string directory)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "encoder.onnx"), string.Empty);
        File.WriteAllText(Path.Combine(directory, "decoder.onnx"), string.Empty);
    }

    private static void PlaceAiHub(string directory)
    {
        foreach (var half in new[] { "encoder", "decoder" })
        {
            Directory.CreateDirectory(Path.Combine(directory, half));
            File.WriteAllText(Path.Combine(directory, half, "model.onnx"), string.Empty);
            File.WriteAllText(Path.Combine(directory, half, "model.bin"), string.Empty);
        }
    }

    [Fact]
    public void BothOnDiskLayoutsCount()
    {
        var root = TempRoot();
        PlacePortable(Path.Combine(root, "flat"));
        PlaceAiHub(Path.Combine(root, "nested"));

        Assert.True(ModelLayout.HasModel(Path.Combine(root, "flat")));
        Assert.True(ModelLayout.HasModel(Path.Combine(root, "nested")));
        Assert.False(ModelLayout.HasModel(Path.Combine(root, "absent")));
    }

    [Fact]
    public void HalfAnExportDoesNotCountAsOne()
    {
        var root = TempRoot();
        var directory = Path.Combine(root, "partial");
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "encoder.onnx"), string.Empty);

        Assert.False(ModelLayout.HasModel(directory));
    }

    [Fact]
    public void ThePlannersPreferredSizeWinsWhenItIsInstalled()
    {
        var root = TempRoot();
        PlacePortable(Path.Combine(root, ModelLayout.PortableFolder, "small.en"));
        PlacePortable(Path.Combine(root, ModelLayout.PortableFolder, "tiny.en"));

        var located = ModelLayout.Locate(root, SocFamily.Unknown, ComputeDevice.Cpu, "small.en");

        Assert.Equal(Path.Combine(root, ModelLayout.PortableFolder, "small.en"), located);
    }

    /// <summary>
    /// The case that sent a working NPU model to the CPU. The plan asks for medium.en; what is
    /// installed is a published build under its own name. Anything usable beats nothing.
    /// </summary>
    [Fact]
    public void AnInstalledModelUnderADifferentNameIsStillFound()
    {
        var root = TempRoot();
        PlaceAiHub(Path.Combine(root, "snapdragon-x-elite", "large-v3-turbo"));

        var located = ModelLayout.Locate(
            root, SocFamily.SnapdragonXElite, ComputeDevice.Npu, "medium.en");

        Assert.Equal(Path.Combine(root, "snapdragon-x-elite", "large-v3-turbo"), located);
    }

    [Fact]
    public void NothingInstalledLocatesNothing()
    {
        var root = TempRoot();
        Directory.CreateDirectory(Path.Combine(root, ModelLayout.PortableFolder));

        Assert.Null(ModelLayout.Locate(root, SocFamily.Unknown, ComputeDevice.Cpu, "small.en"));
    }

    /// <summary>
    /// Portable weights must never satisfy the NPU check. If they did, the planner would send
    /// the encoder to the Hexagon, where an ordinary ONNX export cannot run.
    /// </summary>
    [Fact]
    public void PortableWeightsDoNotCountAsChipsetWeights()
    {
        var root = TempRoot();
        PlacePortable(Path.Combine(root, ModelLayout.PortableFolder, "small.en"));

        Assert.False(ModelLayout.HasChipsetModels(root, SocFamily.SnapdragonXElite));
    }

    [Fact]
    public void ChipsetWeightsAreFoundBeneathASizeFolder()
    {
        var root = TempRoot();
        PlaceAiHub(Path.Combine(root, "snapdragon-x-elite", "large-v3-turbo"));

        Assert.True(ModelLayout.HasChipsetModels(root, SocFamily.SnapdragonXElite));

        // A different chip must not be told it has weights built for this one.
        Assert.False(ModelLayout.HasChipsetModels(root, SocFamily.SnapdragonX2));
    }

    [Fact]
    public void AMachineWithNoNpuNeverClaimsChipsetWeights()
    {
        var root = TempRoot();
        PlacePortable(Path.Combine(root, ModelLayout.PortableFolder, "small.en"));

        Assert.False(ModelLayout.HasChipsetModels(root, SocFamily.NonQualcomm));
        Assert.False(ModelLayout.HasChipsetModels(root, SocFamily.Unknown));
    }
}
