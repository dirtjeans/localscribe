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
/// <para>
/// Every question about where files sit is answered here and nowhere else. The probe, the
/// loader and the app all used to know this independently, and every time one learned something
/// the others did not, a correctly installed model reported as missing.
/// </para>
/// </summary>
public static class ModelLayout
{
    /// <summary>
    /// Directory holding portable ONNX exports: the ones the CPU and DirectML paths use.
    /// Named for the device rather than the chip because these are not chipset-specific.
    /// </summary>
    public const string PortableFolder = "cpu";

    /// <summary>The two halves of a Whisper export, by the names used on disk.</summary>
    private static readonly string[] Halves = ["encoder", "decoder"];

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
    /// The directory a given plan's weights would ideally live in.
    /// <para>
    /// Keyed on where the encoder was placed rather than on the chip, because a Snapdragon
    /// running on the CPU needs portable weights exactly as much as an Intel laptop does.
    /// </para>
    /// </summary>
    public static string Resolve(string root, SocFamily family, ComputeDevice encoderDevice, string modelSize)
    {
        ArgumentException.ThrowIfNullOrEmpty(root);
        ArgumentException.ThrowIfNullOrEmpty(modelSize);

        return Path.Combine(root, FolderFor(family, encoderDevice), modelSize);
    }

    /// <summary>Which of the two folders a placement calls for.</summary>
    public static string FolderFor(SocFamily family, ComputeDevice encoderDevice) =>
        encoderDevice == ComputeDevice.Npu ? ChipsetFolder(family) : PortableFolder;

    /// <summary>
    /// Finds weights that are actually installed, preferring the size the plan asked for.
    /// <para>
    /// The plan names a size from Whisper's own family — <c>small.en</c>, <c>medium.en</c> — but
    /// what is on disk is whatever the user managed to obtain, and for the NPU that is usually a
    /// published build under its own name such as <c>large-v3-turbo</c>. Insisting on an exact
    /// match makes a perfectly good model invisible and sends the work to the CPU, which is the
    /// outcome this whole project is trying to avoid. So the preferred name wins when it is
    /// there, and anything usable wins over nothing.
    /// </para>
    /// </summary>
    /// <returns>The directory to load from, or null when the folder holds no usable set.</returns>
    public static string? Locate(
        string root,
        SocFamily family,
        ComputeDevice encoderDevice,
        string preferredSize)
    {
        ArgumentException.ThrowIfNullOrEmpty(root);
        ArgumentException.ThrowIfNullOrEmpty(preferredSize);

        var folder = Path.Combine(root, FolderFor(family, encoderDevice));

        if (!Directory.Exists(folder))
        {
            return null;
        }

        var preferred = Path.Combine(folder, preferredSize);
        if (HasModel(preferred))
        {
            return preferred;
        }

        // Some layouts put the files straight in the folder with no size beneath it.
        if (HasModel(folder))
        {
            return folder;
        }

        // Ordered so the answer does not depend on how the filesystem feels today.
        return Directory.EnumerateDirectories(folder)
            .OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault(HasModel);
    }

    /// <summary>True when a directory holds both halves of an export, in either layout.</summary>
    public static bool HasModel(string directory) =>
        Directory.Exists(directory) && Halves.All(half => GraphPath(directory, half) is not null);

    /// <summary>
    /// The file to open for one half, or null when it is not there.
    /// <para>
    /// Two layouts are accepted. <c>encoder.onnx</c> beside <c>decoder.onnx</c> is what portable
    /// exports look like. AI Hub ships <c>encoder/model.onnx</c> beside a <c>model.bin</c>
    /// holding the actual context binary; those two must stay together and keep their names,
    /// because the wrapper references the binary by relative path.
    /// </para>
    /// </summary>
    public static string? GraphPath(string directory, string half)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        ArgumentException.ThrowIfNullOrEmpty(half);

        var flat = Path.Combine(directory, $"{half}.onnx");
        if (File.Exists(flat))
        {
            return flat;
        }

        var nested = Path.Combine(directory, half, "model.onnx");
        return File.Exists(nested) ? nested : null;
    }

    /// <summary>
    /// True when the chipset folder holds precompiled weights for this family. Used by the probe
    /// to decide whether the NPU is a real option.
    /// </summary>
    public static bool HasChipsetModels(string? root, SocFamily family)
    {
        // A machine with no NPU of its own has no chipset folder, and must not be told it has
        // NPU weights just because portable ones were fetched into the shared one.
        if (root is null || !Directory.Exists(root) || ChipsetFolder(family) == PortableFolder)
        {
            return false;
        }

        var folder = Path.Combine(root, ChipsetFolder(family));

        if (!Directory.Exists(folder))
        {
            return false;
        }

        return HasModel(folder) || Directory.EnumerateDirectories(folder).Any(HasModel);
    }
}
