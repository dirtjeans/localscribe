using System.IO.Compression;

namespace LocalScribe.Core.Models;

/// <summary>
/// Where whisper.cpp models come from.
/// <para>
/// A pure lookup with no I/O, like <see cref="WhisperModelSource"/>, so the URL and filename
/// decisions can be checked without touching the network.
/// </para>
/// <para>
/// The source is ggerganov's own whisper.cpp repository, which publishes every size as a
/// single GGML file plus, for the Apple Neural Engine path, a zipped Core ML encoder bundle.
/// Both keep their published names on disk — whisper.cpp derives the encoder bundle's path
/// from the GGML file's name, so renaming either breaks the pairing silently, which is the
/// never-rename invariant wearing its Core ML coat.
/// </para>
/// </summary>
public static class WhisperCppModelSource
{
    private const string Repository = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";

    /// <summary>Where whisper.cpp models live under the model root.</summary>
    public const string DirectoryName = "whisper-cpp";

    /// <summary>
    /// The size the macOS port runs. Large, because the encoder — the expensive half — leaves
    /// the CPU for the Neural Engine, which is the same reasoning that sends the NPU path to a
    /// larger model than the CPU path would tolerate.
    /// </summary>
    public const string DefaultSize = "large-v3-turbo";

    /// <summary>
    /// Roughly what the default download comes to, GGML file and Core ML encoder together.
    /// Used to warn before spending it, so the figure only has to be close.
    /// </summary>
    public const long ApproximateBytes = 2_300_000_000;

    /// <summary>The GGML model file's published name for a size.</summary>
    public static string GgmlFileName(string size) => $"ggml-{size}.bin";

    /// <summary>The zipped Core ML encoder's published name for a size.</summary>
    public static string CoreMlZipName(string size) => $"ggml-{size}-encoder.mlmodelc.zip";

    /// <summary>The files whisper.cpp needs on disk for a size.</summary>
    /// <param name="size">A whisper.cpp size such as <c>large-v3-turbo</c> or <c>base.en</c>.</param>
    /// <param name="coreMl">
    /// Whether to also fetch the Core ML encoder. Only Apple silicon can use it, and it is
    /// optional even there: without it whisper.cpp still runs, with the encoder on CPU/Metal
    /// instead of the Neural Engine.
    /// </param>
    public static IReadOnlyList<ModelDownload> Files(string size, bool coreMl)
    {
        ArgumentException.ThrowIfNullOrEmpty(size);

        var files = new List<ModelDownload>
        {
            new(new Uri(Repository + GgmlFileName(size)), GgmlFileName(size)),
        };

        if (coreMl)
        {
            // Optional at the fetch layer too: a size the repository has no Core ML build for
            // should degrade to the Metal encoder, not fail the whole download.
            files.Add(new ModelDownload(new Uri(Repository + CoreMlZipName(size)), CoreMlZipName(size), Optional: true));
        }

        return files;
    }

    /// <summary>
    /// Unpacks a downloaded Core ML encoder zip beside its GGML file and removes the zip.
    /// <para>
    /// whisper.cpp wants the <c>.mlmodelc</c> bundle as a directory, not an archive. Extraction
    /// is separated from download so a failure here leaves the fetched zip in place for a
    /// retry rather than costing the six hundred megabytes again.
    /// </para>
    /// </summary>
    /// <returns>True when a bundle is in place, whether this call unpacked it or found it.</returns>
    public static bool UnpackCoreMlEncoder(string directory, string size)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);

        var bundle = Path.Combine(
            directory,
            Path.GetFileNameWithoutExtension(CoreMlZipName(size)));

        if (Directory.Exists(bundle))
        {
            return true;
        }

        var zip = Path.Combine(directory, CoreMlZipName(size));

        if (!File.Exists(zip))
        {
            return false;
        }

        ZipFile.ExtractToDirectory(zip, directory, overwriteFiles: true);

        if (!Directory.Exists(bundle))
        {
            return false;
        }

        File.Delete(zip);
        return true;
    }
}
