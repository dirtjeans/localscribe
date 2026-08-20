namespace LocalScribe.Core.Models;

/// <summary>One file to fetch, and the name it takes on disk.</summary>
/// <param name="Source">Where to get it.</param>
/// <param name="FileName">What the app expects it to be called locally.</param>
/// <param name="Optional">
/// When true, a download failure is reported and tolerated rather than fatal.
/// </param>
public sealed record ModelDownload(Uri Source, string FileName, bool Optional = false);

/// <summary>
/// Maps a Whisper size onto the files that make up a portable export.
/// <para>
/// This is deliberately a pure lookup with no I/O, so the URL and filename decisions are
/// testable without touching the network.
/// </para>
/// <para>
/// The source is the <c>Xenova</c> Whisper conversions on Hugging Face. They are ungated,
/// carry every size the planner can ask for, and use one consistent layout. Qualcomm's own
/// Hugging Face repositories are not an option despite what older setup notes say: they were
/// deprecated and now hold nothing but a pointer to AI Hub.
/// </para>
/// <para>
/// These are ordinary ONNX exports, not precompiled QNN context binaries, so they serve the
/// CPU and DirectML paths only. Nothing here can light up the NPU; that needs a chipset
/// specific build from AI Hub, which requires an account and a compile job.
/// </para>
/// </summary>
public static class WhisperModelSource
{
    private const string RepositoryRoot = "https://huggingface.co/Xenova/whisper-";

    /// <summary>The sizes <c>AcceleratorPlanner</c> can choose, and therefore the ones we fetch.</summary>
    public static IReadOnlyList<string> SupportedSizes { get; } =
        new[] { "tiny.en", "base.en", "small.en", "medium.en" };

    /// <summary>True when <paramref name="modelSize"/> is one we know how to fetch.</summary>
    public static bool IsSupported(string modelSize) => SupportedSizes.Contains(modelSize);

    /// <summary>
    /// The files making up one portable Whisper export.
    /// </summary>
    /// <param name="modelSize">A size from <see cref="SupportedSizes"/>, such as <c>base.en</c>.</param>
    /// <exception cref="ArgumentOutOfRangeException">The size is not one we publish a mapping for.</exception>
    public static IReadOnlyList<ModelDownload> For(string modelSize)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelSize);

        if (!IsSupported(modelSize))
        {
            throw new ArgumentOutOfRangeException(
                nameof(modelSize),
                modelSize,
                $"Unknown Whisper size. Known sizes: {string.Join(", ", SupportedSizes)}.");
        }

        var repository = $"{RepositoryRoot}{modelSize}/resolve/main/";

        return new[]
        {
            // The plain encoder and decoder, not the _merged or _with_past variants. The decode
            // loop here re-runs the whole prefix each step rather than carrying a key/value
            // cache, so it wants the export that takes no past state.
            new ModelDownload(new Uri(repository + "onnx/encoder_model.onnx"), "encoder.onnx"),
            new ModelDownload(new Uri(repository + "onnx/decoder_model.onnx"), "decoder.onnx"),

            new ModelDownload(new Uri(repository + "vocab.json"), "vocab.json"),

            // Exports disagree about whether the special tokens live in vocab.json or beside
            // it, and WhisperTokenizer reads this only when it is there.
            new ModelDownload(new Uri(repository + "added_tokens.json"), "added_tokens.json", Optional: true),
        };
    }
}
