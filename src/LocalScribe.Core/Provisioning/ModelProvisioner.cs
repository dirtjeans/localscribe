using LocalScribe.Core.Models;

namespace LocalScribe.Core.Provisioning;

/// <summary>
/// Makes a bare machine ready: whatever models are missing are downloaded, whatever is present
/// is left alone.
/// <para>
/// This exists so that a first launch needs no manual step at all — the app provisions its own
/// weights, narrating cost and progress as it goes. It downloads only model files, into the
/// app's own model directory; installing anything beyond that (Foundry Local, drivers) stays
/// behind an explicit user action, which is the line the app has always held.
/// </para>
/// <para>
/// Every stage is idempotent and every failure is tolerated in the direction the pipeline
/// already degrades: no transcription model is fatal, a missing aligner means loudness
/// estimates, missing speaker models mean no labels. The fetcher writes through .part files,
/// so a failed download retries from nothing rather than trusting a half-written graph.
/// </para>
/// </summary>
public sealed class ModelProvisioner(
    ModelFetcher? fetcher = null,
    IArchiveExtractor? diarizationExtractor = null)
{
    private readonly ModelFetcher _fetcher = fetcher ?? new ModelFetcher();

    /// <summary>What a machine is missing, so the caller can say so before spending bandwidth.</summary>
    public static bool NeedsAnything(string modelRoot, bool whisperCpp) =>
        (whisperCpp && !HasWhisperCpp(modelRoot))
        || !HasAligner(modelRoot)
        || !DiarizationModelInstaller.IsInstalled(Path.Combine(modelRoot, "diarization"));

    /// <summary>
    /// Ensures everything transcription wants is on disk, reporting each stage.
    /// </summary>
    /// <param name="modelRoot">The model directory the app reads from.</param>
    /// <param name="whisperCpp">Whether this platform transcribes with whisper.cpp.</param>
    /// <param name="coreMl">Whether to fetch the Core ML encoder alongside it.</param>
    /// <returns>True when a transcription model is present by the end.</returns>
    public async Task<bool> EnsureAsync(
        string modelRoot,
        bool whisperCpp,
        bool coreMl,
        IProgress<InstallProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var canTranscribe = !whisperCpp || HasWhisperCpp(modelRoot);

        if (whisperCpp && !HasWhisperCpp(modelRoot))
        {
            canTranscribe = await FetchWhisperCppAsync(modelRoot, coreMl, progress, cancellationToken)
                .ConfigureAwait(false);
        }

        if (!HasAligner(modelRoot))
        {
            await FetchAlignerAsync(modelRoot, progress, cancellationToken).ConfigureAwait(false);
        }

        await FetchDiarizationAsync(modelRoot, progress, cancellationToken).ConfigureAwait(false);

        return canTranscribe;
    }

    private static bool HasWhisperCpp(string modelRoot)
    {
        var directory = Path.Combine(modelRoot, WhisperCppModelSource.DirectoryName);

        return Directory.Exists(directory)
            && Directory.EnumerateFiles(directory, "ggml-*.bin").Any();
    }

    private static bool HasAligner(string modelRoot) =>
        File.Exists(Path.Combine(modelRoot, AlignmentModelSource.DirectoryName, "model_fp16.onnx"));

    private async Task<bool> FetchWhisperCppAsync(
        string modelRoot,
        bool coreMl,
        IProgress<InstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        var directory = Path.Combine(modelRoot, WhisperCppModelSource.DirectoryName);
        var size = WhisperCppModelSource.DefaultSize;

        try
        {
            await _fetcher.FetchAsync(
                    directory,
                    WhisperCppModelSource.Files(size, coreMl),
                    Staged(progress, $"the {size} speech model", WhisperCppModelSource.ApproximateBytes),
                    force: false,
                    cancellationToken)
                .ConfigureAwait(false);

            WhisperCppModelSource.UnpackCoreMlEncoder(directory, size);
            return true;
        }
        catch (Exception exception) when (exception is HttpRequestException or IOException or InvalidDataException)
        {
            progress?.Report(new InstallProgress(
                "models",
                $"The speech model could not be downloaded: {exception.Message} "
                + "Nothing was left half-written; it will be tried again next launch."));
            return HasWhisperCpp(modelRoot);
        }
    }

    private async Task FetchAlignerAsync(
        string modelRoot,
        IProgress<InstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        try
        {
            await _fetcher.FetchAsync(
                    Path.Combine(modelRoot, AlignmentModelSource.DirectoryName),
                    AlignmentModelSource.Files,
                    Staged(progress, "the word aligner", AlignmentModelSource.ApproximateBytes),
                    force: false,
                    cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception exception) when (exception is HttpRequestException or IOException)
        {
            // Optional: without it, word times are estimated from loudness — a loss of
            // precision, not of the transcript.
            progress?.Report(new InstallProgress(
                "models", $"The word aligner could not be downloaded: {exception.Message}"));
        }
    }

    private async Task FetchDiarizationAsync(
        string modelRoot,
        IProgress<InstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        var directory = Path.Combine(modelRoot, "diarization");

        if (DiarizationModelInstaller.IsInstalled(directory))
        {
            return;
        }

        try
        {
            var installer = new DiarizationModelInstaller(extractor: diarizationExtractor);
            var result = await installer.EnsureInstalledAsync(directory, progress, cancellationToken)
                .ConfigureAwait(false);

            if (!result.Succeeded)
            {
                progress?.Report(new InstallProgress("models", result.Message));
            }
        }
        catch (Exception exception) when (exception is HttpRequestException or IOException)
        {
            // Optional: the transcript still says what was said, just not who said it.
            progress?.Report(new InstallProgress(
                "models", $"The speaker models could not be downloaded: {exception.Message}"));
        }
    }

    /// <summary>
    /// Maps the fetcher's per-file byte counts onto one narrated stage, so the user sees
    /// "downloading the speech model… 42%" rather than raw file names and byte totals.
    /// </summary>
    private static IProgress<FetchProgress> Staged(
        IProgress<InstallProgress>? progress,
        string what,
        long approximateBytes)
    {
        long finished = 0;
        var lastPercent = -1;

        return new Progress<FetchProgress>(update =>
        {
            if (update.Done)
            {
                finished += update.BytesRead;
                return;
            }

            var fraction = Math.Clamp(
                (finished + update.BytesRead) / (double)approximateBytes, 0, 1);
            var percent = (int)(fraction * 100);

            if (percent == lastPercent)
            {
                return;
            }

            lastPercent = percent;
            progress?.Report(new InstallProgress(
                "models", $"Downloading {what}… {percent}%", fraction));
        });
    }
}
