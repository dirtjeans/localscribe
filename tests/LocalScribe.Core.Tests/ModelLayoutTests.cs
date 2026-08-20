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
}
