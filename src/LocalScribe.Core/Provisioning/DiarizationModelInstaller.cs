namespace LocalScribe.Core.Provisioning;

/// <summary>
/// Unpacks a downloaded archive.
/// <para>
/// Implemented outside this assembly on purpose. <c>LocalScribe.Core</c> takes no external
/// dependencies so its logic stays testable anywhere, and .NET has no built-in bzip2 decoder,
/// so the concrete extractor lives beside the code that already carries native dependencies.
/// </para>
/// </summary>
public interface IArchiveExtractor
{
    /// <summary>True when this extractor recognises the file's format.</summary>
    bool CanExtract(string archivePath);

    /// <summary>
    /// Unpacks an archive into a directory, flattening any single wrapping folder so callers get
    /// the files rather than a folder containing them.
    /// </summary>
    Task ExtractAsync(string archivePath, string targetDirectory, CancellationToken cancellationToken = default);
}

/// <summary>
/// Fetches the two models diarization needs: pyannote segmentation, and a speaker embedding
/// extractor. Both come from sherpa-onnx release assets, already converted to ONNX.
/// </summary>
public sealed class DiarizationModelInstaller
{
    private const string Owner = "k2-fsa";
    private const string Repository = "sherpa-onnx";

    /// <summary>File name the segmentation model is stored under once unpacked.</summary>
    public const string SegmentationFileName = "segmentation.onnx";

    /// <summary>File name the embedding extractor is stored under.</summary>
    public const string EmbeddingFileName = "embedding.onnx";

    /// <summary>
    /// Release tags to search for the segmentation model.
    /// </summary>
    public static IReadOnlyList<string> SegmentationTags { get; } = ["speaker-segmentation-models"];

    /// <summary>
    /// Release tags to search for the embedding model. The misspelled tag is the real one in the
    /// upstream repository; the corrected spelling is tried too in case it is ever fixed.
    /// </summary>
    public static IReadOnlyList<string> EmbeddingTags { get; } =
        ["speaker-recongition-models", "speaker-recognition-models"];

    /// <summary>Segmentation model preference. pyannote 3.0 is the one this pipeline is tuned for.</summary>
    public static IReadOnlyList<string> SegmentationPreference { get; } =
        ["pyannote-segmentation-3-0", "pyannote-segmentation"];

    /// <summary>
    /// Embedding model preference. WeSpeaker only, and that is a hard requirement rather than a
    /// ranking.
    /// <para>
    /// These extractors do not share an input contract. WeSpeaker's take precomputed filterbank
    /// features on an input named <c>feats</c>, which is what this app computes and feeds. NeMo's
    /// TitaNet takes raw audio on an input named <c>audio_signal</c> and does its own feature
    /// extraction. Handing one the other's input does not degrade the answer, it fails.
    /// </para>
    /// <para>
    /// The list led with <c>nemo_en_titanet_small</c>, which is right for a runtime that supports
    /// several architectures and wrong here, where the feature pipeline — Povey window, HTK mel,
    /// cepstral mean normalisation — is WeSpeaker's specifically. Fetching by preference alone
    /// would have installed a model this app cannot use, over one it already had.
    /// </para>
    /// <para>
    /// English-trained first among those: several published extractors are trained on Mandarin
    /// speakers and separate English voices noticeably less well.
    /// </para>
    /// </summary>
    /// <para>
    /// ResNet34-LM specifically, because that is the model the separation threshold was measured
    /// against. WeSpeaker publishes CAM++ and several deeper ResNets under the same input
    /// contract, so they would run — and would quietly place voices at different distances,
    /// against a threshold calibrated for none of them. The plain ResNet34 is the only fallback:
    /// same architecture, same scale, and the closest thing to the model actually tuned for.
    /// </para>
    public static IReadOnlyList<string> EmbeddingPreference { get; } =
        ["wespeaker_en_voxceleb_resnet34_LM", "wespeaker_en_voxceleb_resnet34"];

    private readonly GitHubReleaseCatalog _catalog;
    private readonly IFileDownloader _downloader;
    private readonly IArchiveExtractor? _extractor;

    public DiarizationModelInstaller(
        GitHubReleaseCatalog? catalog = null,
        IFileDownloader? downloader = null,
        IArchiveExtractor? extractor = null)
    {
        _catalog = catalog ?? new GitHubReleaseCatalog();
        _downloader = downloader ?? new HttpFileDownloader();
        _extractor = extractor;
    }

    /// <summary>True when both models are already present.</summary>
    public static bool IsInstalled(string directory) =>
        File.Exists(Path.Combine(directory, SegmentationFileName))
        && File.Exists(Path.Combine(directory, EmbeddingFileName));

    /// <summary>Downloads whatever is missing.</summary>
    public async Task<InstallResult> EnsureInstalledAsync(
        string directory,
        IProgress<InstallProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (IsInstalled(directory))
        {
            return new InstallResult("diarization-models", true, "Diarization models are already installed.");
        }

        Directory.CreateDirectory(directory);

        try
        {
            if (!File.Exists(Path.Combine(directory, SegmentationFileName)))
            {
                await InstallSegmentationAsync(directory, progress, cancellationToken).ConfigureAwait(false);
            }

            if (!File.Exists(Path.Combine(directory, EmbeddingFileName)))
            {
                await InstallEmbeddingAsync(directory, progress, cancellationToken).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception exception)
        {
            return new InstallResult("diarization-models", false, exception.Message);
        }

        return IsInstalled(directory)
            ? new InstallResult("diarization-models", true, "Diarization models installed.")
            : new InstallResult(
                "diarization-models",
                false,
                "Downloads finished but the expected files are missing. The release layout has "
                + "probably changed; see docs/diarization.md.");
    }

    private async Task InstallSegmentationAsync(
        string directory,
        IProgress<InstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        var asset = await FindAssetAsync(SegmentationTags, SegmentationPreference, null, cancellationToken)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException(
                "No pyannote segmentation asset found in the sherpa-onnx releases.");

        progress?.Report(new InstallProgress("diarization-models", $"Downloading {asset.Name}…"));

        // Loose .onnx assets are used as they are; archives need unpacking, and that needs an
        // extractor the caller supplies.
        if (asset.Name.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            await _downloader.DownloadAsync(
                new RemoteFile(asset.DownloadUrl, SegmentationFileName, asset.SizeBytes),
                directory,
                progress,
                cancellationToken).ConfigureAwait(false);
            return;
        }

        if (_extractor is null || !_extractor.CanExtract(asset.Name))
        {
            throw new InvalidOperationException(
                $"{asset.Name} is an archive and no extractor for it was supplied. Reference "
                + "the doctor, which provides one, or unpack it by hand.");
        }

        var staging = Path.Combine(directory, ".staging");
        Directory.CreateDirectory(staging);

        try
        {
            await _downloader.DownloadAsync(
                new RemoteFile(asset.DownloadUrl, asset.Name, asset.SizeBytes),
                staging,
                progress,
                cancellationToken).ConfigureAwait(false);

            progress?.Report(new InstallProgress("diarization-models", $"Unpacking {asset.Name}…"));

            await _extractor
                .ExtractAsync(Path.Combine(staging, asset.Name), staging, cancellationToken)
                .ConfigureAwait(false);

            var model = Directory
                .EnumerateFiles(staging, "*.onnx", SearchOption.AllDirectories)
                .OrderBy(f => Path.GetFileName(f).Length)
                .FirstOrDefault()
                ?? throw new InvalidOperationException($"{asset.Name} contained no .onnx file.");

            File.Move(model, Path.Combine(directory, SegmentationFileName), overwrite: true);
        }
        finally
        {
            TryDeleteDirectory(staging);
        }
    }

    private async Task InstallEmbeddingAsync(
        string directory,
        IProgress<InstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        var asset = await FindAssetAsync(EmbeddingTags, EmbeddingPreference, ".onnx", cancellationToken)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException(
                "No speaker embedding asset matched. The published names have probably changed; "
                + "see docs/diarization.md for how to point at one by hand.");

        progress?.Report(new InstallProgress("diarization-models", $"Downloading {asset.Name}…"));

        await _downloader.DownloadAsync(
            new RemoteFile(asset.DownloadUrl, EmbeddingFileName, asset.SizeBytes),
            directory,
            progress,
            cancellationToken).ConfigureAwait(false);
    }

    private async Task<ReleaseAsset?> FindAssetAsync(
        IReadOnlyList<string> tags,
        IReadOnlyList<string> preference,
        string? requiredExtension,
        CancellationToken cancellationToken)
    {
        foreach (var tag in tags)
        {
            var assets = await _catalog
                .ListAssetsAsync(Owner, Repository, tag, cancellationToken)
                .ConfigureAwait(false);

            var picked = GitHubReleaseCatalog.PickByPreference(assets, preference, requiredExtension);
            if (picked is not null)
            {
                return picked;
            }
        }

        return null;
    }

    private static void TryDeleteDirectory(string path)
    {
        try
        {
            if (Directory.Exists(path))
            {
                Directory.Delete(path, recursive: true);
            }
        }
        catch (IOException)
        {
            // Leftover staging is untidy, not harmful.
        }
    }
}
