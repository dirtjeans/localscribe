namespace LocalScribe.Core.Models;

/// <summary>
/// Where the forced-alignment model comes from.
/// <para>
/// A pure lookup with no I/O, like <see cref="WhisperModelSource"/>, so the URL and filename
/// decisions can be checked without touching the network.
/// </para>
/// <para>
/// The source is the <c>onnx-community</c> conversion of Meta's MMS-300m forced aligner. It is
/// ungated and carries the same vocabulary and feature settings the aligner expects.
/// </para>
/// <para>
/// Half precision rather than one of the quantised builds, which are a quarter of the size and
/// would be the obvious choice. They use <c>ConvInteger</c>, and ONNX Runtime has no ARM64
/// implementation of it — on this machine they fail at load rather than run slowly, which is
/// the sort of saving that costs the whole feature.
/// </para>
/// </summary>
public static class AlignmentModelSource
{
    private const string Repository =
        "https://huggingface.co/onnx-community/mms-300m-1130-forced-aligner-ONNX/resolve/main/";

    /// <summary>Where the aligner lives under the model root.</summary>
    public const string DirectoryName = "alignment";

    /// <summary>
    /// Roughly what the download comes to. Used to warn before spending it, so the figure only
    /// has to be close.
    /// </summary>
    public const long ApproximateBytes = 631_591_191;

    /// <summary>The files the aligner needs on disk.</summary>
    public static IReadOnlyList<ModelDownload> Files { get; } =
    [
        new ModelDownload(new Uri(Repository + "onnx/model_fp16.onnx"), "model_fp16.onnx"),
        new ModelDownload(new Uri(Repository + "vocab.json"), "vocab.json"),

        // Read by nothing at runtime — the aligner has the sample rate and normalisation built
        // in. Fetched so that what is on disk says what it was built from.
        new ModelDownload(new Uri(Repository + "preprocessor_config.json"), "preprocessor_config.json", Optional: true),
    ];

    /// <summary>The 4-bit build's weights file, under its published name.</summary>
    public const string QuantisedModelFileName = "model_q4.onnx";

    /// <summary>Roughly what the 4-bit download comes to.</summary>
    public const long QuantisedApproximateBytes = 241_000_000;

    /// <summary>
    /// The same aligner with its MatMuls quantised to 4 bits (MatMulNBits, which has ARM64
    /// kernels — unlike the int8 builds' ConvInteger, which is why those fail at load).
    /// <para>
    /// Graded by the doctor's --aligner-trial against the fp16 build on the two reference
    /// recordings: 99% and 92% of words within 0.1 s of it, every word within 0.5 s, no drift
    /// in any fifth — from a download two-fifths the size, holding half the memory while it
    /// scans. Faster too: 19–30% alone, and about a third less wall time across the whole
    /// pipeline at the Balanced pace on the M2.
    /// </para>
    /// </summary>
    public static IReadOnlyList<ModelDownload> QuantisedFiles { get; } =
    [
        new ModelDownload(new Uri(Repository + "onnx/" + QuantisedModelFileName), QuantisedModelFileName),
        new ModelDownload(new Uri(Repository + "vocab.json"), "vocab.json"),
        new ModelDownload(new Uri(Repository + "preprocessor_config.json"), "preprocessor_config.json", Optional: true),
    ];

    /// <summary>
    /// Whether this machine runs the 4-bit build. macOS only, because that is where the trial
    /// was run; the Snapdragon keeps fp16 until the same trial is run there.
    /// </summary>
    public static bool PreferQuantised => OperatingSystem.IsMacOS();

    public static IReadOnlyList<ModelDownload> FilesForThisMachine =>
        PreferQuantised ? QuantisedFiles : Files;

    public static string ModelFileNameForThisMachine =>
        PreferQuantised ? QuantisedModelFileName : "model_fp16.onnx";

    public static long ApproximateBytesForThisMachine =>
        PreferQuantised ? QuantisedApproximateBytes : ApproximateBytes;
}
