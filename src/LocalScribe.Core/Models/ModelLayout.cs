using LocalScribe.Core.Hardware;

namespace LocalScribe.Core.Models;

/// <summary>
/// Decides which directory a set of Whisper weights belongs in, and which set to open.
/// <para>
/// There are two kinds of weights in this app and they are not interchangeable. Precompiled
/// QNN context binaries are built for one chipset and run only on the Hexagon NPU. Portable
/// ONNX exports run anywhere the CPU or DirectML can reach, and cannot run on the NPU at all.
/// </para>
/// <para>
/// Keeping them in separate directories is what stops the two being confused, and that matters
/// more than it sounds. <see cref="DeviceCapabilities.NpuUsable"/> is satisfied partly by the
/// presence of chipset weights, so a portable export dropped into the chipset folder would
/// convince the planner to send the encoder to the NPU, where it would either fail to load or
/// quietly relocate to the CPU. Silent fallback is the failure this project exists to avoid,
/// so the layout is designed to make it unrepresentable rather than merely unlikely.
/// </para>
/// </summary>
public static class ModelLayout
{
    /// <summary>
    /// Directory holding portable ONNX exports: the ones the CPU and DirectML paths use.
    /// Named for the device rather than the chip because these are not chipset-specific.
    /// </summary>
    public const string PortableFolder = "cpu";

    /// <summary>
    /// Directory name for a chipset's precompiled QNN binaries. Machines with no Qualcomm
    /// NPU have no chipset weights to speak of and fall through to the portable folder.
    /// </summary>
    public static string ChipsetFolder(SocFamily family) => family switch
    {
        SocFamily.SnapdragonXElite => "snapdragon-x-elite",
        SocFamily.SnapdragonXPlus => "snapdragon-x-plus",
        SocFamily.SnapdragonX2 => "snapdragon-x2",
        _ => PortableFolder,
    };

    /// <summary>
    /// The directory holding the weights a given plan will actually open.
    /// <para>
    /// Keyed on where the encoder was placed rather than on the chip, because a Snapdragon
    /// running on the CPU needs portable weights exactly as much as an Intel laptop does.
    /// </para>
    /// </summary>
    public static string Resolve(string root, SocFamily family, ComputeDevice encoderDevice, string modelSize)
    {
        ArgumentException.ThrowIfNullOrEmpty(root);
        ArgumentException.ThrowIfNullOrEmpty(modelSize);

        var folder = encoderDevice == ComputeDevice.Npu ? ChipsetFolder(family) : PortableFolder;
        return Path.Combine(root, folder, modelSize);
    }
}
