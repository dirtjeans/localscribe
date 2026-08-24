namespace LocalScribe.Core.Provisioning;

/// <summary>
/// Downloads Whisper assets into the layout the app expects.
/// <para>
/// Assets are chipset-specific, so this resolves a build matching the machine it is running on
/// rather than downloading a generic one. A binary compiled for the wrong Snapdragon does not
/// warn — it fails to load later, well away from anything that would explain why.
/// </para>
/// </summary>
public sealed class WhisperModelInstaller
{
    private readonly HuggingFaceCatalog _catalog;
    private readonly IFileDownloader _downloader;

    public WhisperModelInstaller(HuggingFaceCatalog? catalog = null, IFileDownloader? downloader = null)
    {
        _catalog = catalog ?? new HuggingFaceCatalog();
        _downloader = downloader ?? new HttpFileDownloader();
    }

    /// <summary>
    /// Ensures a usable model exists in <paramref name="modelDirectory"/>, downloading one if not.
    /// </summary>
    /// <param name="modelDirectory">Directory for this specific chipset and model size.</param>
    /// <param name="whisperModel">Model size, e.g. <c>base.en</c>.</param>
    /// <param name="chipsetSlug">Chipset folder name used to match a matching build.</param>
    public async Task<InstallResult> EnsureInstalledAsync(
        string modelDirectory,
        string whisperModel,
        string chipsetSlug,
        IProgress<InstallProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (ModelManifest.Discover(modelDirectory) is not null)
        {
            return new InstallResult("whisper-model", true, $"{whisperModel} is already installed.");
        }

        var repositories = HuggingFaceCatalog.RepositoriesFor(whisperModel);
        var attempts = new List<string>();

        foreach (var repository in repositories)
        {
            cancellationToken.ThrowIfCancellationRequested();

            progress?.Report(new InstallProgress("whisper-model", $"Looking in {repository}…"));

            var files = await _catalog.ListFilesAsync(repository, cancellationToken).ConfigureAwait(false);
            if (files.Count == 0)
            {
                attempts.Add($"{repository}: not found or empty");
                continue;
            }

            var selected = HuggingFaceCatalog.SelectAssets(files, chipsetSlug);
            var layout = ModelManifest.Infer(selected, repository);

            if (layout is null)
            {
                // The repository exists but does not hold a complete encoder/decoder/vocab set
                // for this chipset. Move on rather than downloading something unusable.
                attempts.Add($"{repository}: no complete asset set for {chipsetSlug}");
                continue;
            }

            await DownloadAllAsync(repository, selected, modelDirectory, progress, cancellationToken)
                .ConfigureAwait(false);

            layout.Save(modelDirectory);

            return new InstallResult(
                "whisper-model",
                true,
                $"Installed {whisperModel} from {repository} ({selected.Count} files).");
        }

        return new InstallResult(
            "whisper-model",
            false,
            $"Could not find a {whisperModel} build for {chipsetSlug}. Tried: {string.Join("; ", attempts)}. "
            + "See docs/setup-snapdragon.md to place files by hand.");
    }

    private async Task DownloadAllAsync(
        string repository,
        IReadOnlyList<string> paths,
        string modelDirectory,
        IProgress<InstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(modelDirectory);

        for (var i = 0; i < paths.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var path = paths[i];
            var fileName = Path.GetFileName(path);

            progress?.Report(new InstallProgress(
                "whisper-model",
                $"Downloading {fileName} ({i + 1} of {paths.Count})…",
                i / (double)paths.Count));

            await _downloader.DownloadAsync(
                new RemoteFile(HuggingFaceCatalog.DownloadUrl(repository, path), fileName),
                modelDirectory,
                progress,
                cancellationToken).ConfigureAwait(false);
        }
    }
}
