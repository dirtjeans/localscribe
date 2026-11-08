namespace LocalScribe.Core.Models;

/// <summary>
/// Where the Nemotron 3 Diarization model comes from.
/// <para>
/// NVIDIA publishes PyTorch weights only. This is a community ONNX export of them, split the way
/// streaming needs — an embedding graph and a transformer step — with the mel filterbank and the
/// learned silence frame alongside. Before it was adopted, the C# driver that runs it was checked
/// against the exporter's own reference implementation on two recordings, 23 minutes in all, and
/// every one of the 137,837 frames came back bit-identical; that reference was in turn checked
/// against NVIDIA's transformers code by the exporter.
/// </para>
/// <para>
/// Because the source is not the model's author, everything is pinned: the repository commit in
/// the address, and the SHA-256 of every file. An upload changed after that check fails the
/// download rather than quietly becoming the model.
/// </para>
/// <para>
/// The int8 transformer on the Snapdragon: twice as fast at a quarter of the size, and its
/// thresholded speaker decisions differ from float32's on a few seconds of one minor speaker in
/// a twenty-minute recording. On the Mac the same int8 file invents speakers — three on the
/// two-person debate, seven on Karl where float32 and the Snapdragon find six — so the Mac runs
/// float32, at two to three times the time and about 300 MB more memory. See docs/model-trials.md.
/// </para>
/// </summary>
public static class SortformerModelSource
{
    private const string Repository =
        "https://huggingface.co/NealCaren/Nemotron-3-Diarization-ONNX/resolve/46642c09b3bd014c5ceb19d20524fa2f1dc27dee/";

    /// <summary>Where the model lives under the model root.</summary>
    public const string DirectoryName = "diarization-nemotron";

    /// <summary>Roughly what the download comes to.</summary>
    public const long ApproximateBytes = 105_564_094;

    /// <summary>Roughly what the float32 download comes to.</summary>
    public const long FullPrecisionApproximateBytes = 397_926_679;

    /// <summary>The float32 transformer's file, under its published name.</summary>
    public const string FullPrecisionFileName = "step.onnx";

    private const string Int8FileName = "step_int8.onnx";

    /// <summary>The int8 transformer and the files around it, as the driver names them.</summary>
    public static IReadOnlyList<ModelDownload> Files { get; } =
    [
        Pinned("embed.onnx", "2e8796e42286abe2e2d7d44a97fab7e5a7d06b2d4cbee82a3a0f730cfff84a33"),
        Pinned(Int8FileName, "1f18a5aed4909677d3618c9e6ed6e3003720b6d6e766f0f8b337aa4fcb0a9d2a"),
        Pinned("mel_filters.bin", "bce5ec5f194a5913f6508cee5a85512e7bad2352db8fc28f5c6ff75af8b09137"),
        Pinned("silence_embeds.bin", "d4417b3c0eabdf7c47032fac2b5b5a7ee83d819a6ddda8fd8eaf74e2b5cc4ac7"),

        // Read by nothing. The OpenMDW licence travels with the weights it covers.
        Pinned("LICENSE", "2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df"),
    ];

    /// <summary>The same set with the float32 transformer in place of the int8 one.</summary>
    public static IReadOnlyList<ModelDownload> FullPrecisionFiles { get; } =
    [
        .. Files.Where(file => file.FileName != Int8FileName),
        Pinned(FullPrecisionFileName, "cd7fa5b062e0cbcd8c43e27627eb7fb305e5bce3ec4faa89d8ff8d8879bae82f"),
    ];

    /// <summary>Whether this machine runs the float32 transformer: the Mac, as measured.</summary>
    public static bool PreferFullPrecision => OperatingSystem.IsMacOS();

    public static IReadOnlyList<ModelDownload> FilesForThisMachine =>
        PreferFullPrecision ? FullPrecisionFiles : Files;

    public static long ApproximateBytesForThisMachine =>
        PreferFullPrecision ? FullPrecisionApproximateBytes : ApproximateBytes;

    /// <summary>True when everything this machine should have is on disk, so nothing is left to fetch.</summary>
    public static bool IsInstalled(string directory) =>
        FilesForThisMachine
            .Where(file => file.FileName != "LICENSE")
            .All(file => File.Exists(Path.Combine(directory, file.FileName)));

    /// <summary>
    /// Whether speakers can be found at all with what is on disk.
    /// <para>
    /// Wider than <see cref="IsInstalled"/> for one case: a Mac that has only the int8 file,
    /// because it installed before the Mac moved to float32 and the larger download has not
    /// succeeded. Speakers with a known flaw beat no speakers; each launch tries the download again.
    /// </para>
    /// </summary>
    public static bool CanRun(string directory) =>
        Files
            .Where(file => file.FileName is not ("LICENSE" or Int8FileName))
            .All(file => File.Exists(Path.Combine(directory, file.FileName)))
        && (RunsFullPrecision(directory) || File.Exists(Path.Combine(directory, Int8FileName)));

    /// <summary>Which transformer to load from this directory: float32 where preferred and present.</summary>
    public static bool RunsFullPrecision(string directory) =>
        PreferFullPrecision && File.Exists(Path.Combine(directory, FullPrecisionFileName));

    private static ModelDownload Pinned(string name, string sha256) =>
        new(new Uri(Repository + name), name, Sha256: sha256);
}
