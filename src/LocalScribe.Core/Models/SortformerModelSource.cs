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
/// The int8 transformer, not the float32 one. On the Snapdragon it is twice as fast at a quarter
/// of the size, and its thresholded speaker decisions differ from float32's on a few seconds of
/// one minor speaker in a twenty-minute recording.
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

    /// <summary>The files the driver reads, as the driver names them.</summary>
    public static IReadOnlyList<ModelDownload> Files { get; } =
    [
        Pinned("embed.onnx", "2e8796e42286abe2e2d7d44a97fab7e5a7d06b2d4cbee82a3a0f730cfff84a33"),
        Pinned("step_int8.onnx", "1f18a5aed4909677d3618c9e6ed6e3003720b6d6e766f0f8b337aa4fcb0a9d2a"),
        Pinned("mel_filters.bin", "bce5ec5f194a5913f6508cee5a85512e7bad2352db8fc28f5c6ff75af8b09137"),
        Pinned("silence_embeds.bin", "d4417b3c0eabdf7c47032fac2b5b5a7ee83d819a6ddda8fd8eaf74e2b5cc4ac7"),

        // Read by nothing. The OpenMDW licence travels with the weights it covers.
        Pinned("LICENSE", "2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df"),
    ];

    /// <summary>True when every file the driver reads is on disk.</summary>
    public static bool IsInstalled(string directory) =>
        Files
            .Where(file => file.FileName != "LICENSE")
            .All(file => File.Exists(Path.Combine(directory, file.FileName)));

    private static ModelDownload Pinned(string name, string sha256) =>
        new(new Uri(Repository + name), name, Sha256: sha256);
}
