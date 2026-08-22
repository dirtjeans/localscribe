namespace LocalScribe.Core.Models;

/// <summary>What happened to one file.</summary>
public enum FetchOutcome
{
    /// <summary>Downloaded in full.</summary>
    Downloaded,

    /// <summary>Already present, so left alone.</summary>
    AlreadyPresent,

    /// <summary>Optional and unavailable. The export simply does not carry this file.</summary>
    SkippedOptional,
}

/// <summary>The result of fetching one file.</summary>
public sealed record FetchResult(string FileName, FetchOutcome Outcome, long Bytes);

/// <summary>
/// Downloads a set of model files into a directory.
/// <para>
/// Every write goes to a <c>.part</c> file that is moved into place only once the body is
/// complete. A half-written encoder.onnx is worse than no encoder at all: the probe would see
/// the filename, report the assets present, and the failure would surface later as an opaque
/// ONNX Runtime error rather than as a failed download.
/// </para>
/// </summary>
public sealed class ModelFetcher(HttpClient? httpClient = null)
{
    private readonly HttpClient _httpClient = httpClient ?? new HttpClient
    {
        // Model files run to hundreds of megabytes, and the default 100 seconds is not enough
        // on a slow connection.
        Timeout = TimeSpan.FromMinutes(30),
    };

    /// <summary>
    /// Fetches every file for <paramref name="modelSize"/> into <paramref name="targetDirectory"/>.
    /// </summary>
    /// <param name="targetDirectory">Created if it does not exist.</param>
    /// <param name="modelSize">A size from <see cref="WhisperModelSource.SupportedSizes"/>.</param>
    /// <param name="progress">Called as each file starts, progresses, and finishes.</param>
    /// <param name="force">Re-download files that are already present.</param>
    public Task<IReadOnlyList<FetchResult>> FetchAsync(
        string targetDirectory,
        string modelSize,
        IProgress<FetchProgress>? progress = null,
        bool force = false,
        CancellationToken cancellationToken = default) =>
        FetchAsync(targetDirectory, WhisperModelSource.For(modelSize), progress, force, cancellationToken);

    /// <summary>Fetches an arbitrary set of files into <paramref name="targetDirectory"/>.</summary>
    /// <param name="targetDirectory">Created if it does not exist.</param>
    /// <param name="downloads">What to fetch, and what to call each file locally.</param>
    /// <param name="progress">Called as each file starts, progresses, and finishes.</param>
    /// <param name="force">Re-download files that are already present.</param>
    public async Task<IReadOnlyList<FetchResult>> FetchAsync(
        string targetDirectory,
        IReadOnlyList<ModelDownload> downloads,
        IProgress<FetchProgress>? progress = null,
        bool force = false,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(targetDirectory);
        ArgumentNullException.ThrowIfNull(downloads);

        Directory.CreateDirectory(targetDirectory);

        var results = new List<FetchResult>();

        foreach (var download in downloads)
        {
            var destination = Path.Combine(targetDirectory, download.FileName);

            if (!force && File.Exists(destination) && new FileInfo(destination).Length > 0)
            {
                results.Add(new FetchResult(download.FileName, FetchOutcome.AlreadyPresent, new FileInfo(destination).Length));
                progress?.Report(new FetchProgress(download.FileName, 0, 0, Done: true));
                continue;
            }

            try
            {
                var bytes = await DownloadAsync(download, destination, progress, cancellationToken)
                    .ConfigureAwait(false);
                results.Add(new FetchResult(download.FileName, FetchOutcome.Downloaded, bytes));
            }
            catch (HttpRequestException) when (download.Optional)
            {
                results.Add(new FetchResult(download.FileName, FetchOutcome.SkippedOptional, 0));
                progress?.Report(new FetchProgress(download.FileName, 0, 0, Done: true));
            }
        }

        return results;
    }

    private async Task<long> DownloadAsync(
        ModelDownload download,
        string destination,
        IProgress<FetchProgress>? progress,
        CancellationToken cancellationToken)
    {
        using var response = await _httpClient
            .GetAsync(download.Source, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);

        response.EnsureSuccessStatusCode();

        var total = response.Content.Headers.ContentLength ?? 0;
        var partial = destination + ".part";
        long written = 0;

        await using (var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
        await using (var target = File.Create(partial))
        {
            var buffer = new byte[81920];
            int read;

            while ((read = await source.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
            {
                await target.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                written += read;
                progress?.Report(new FetchProgress(download.FileName, written, total, Done: false));
            }
        }

        File.Move(partial, destination, overwrite: true);
        progress?.Report(new FetchProgress(download.FileName, written, total, Done: true));

        return written;
    }
}

/// <summary>Progress for one file.</summary>
/// <param name="FileName">The file being fetched.</param>
/// <param name="BytesRead">Bytes written so far.</param>
/// <param name="TotalBytes">Total expected, or zero when the server did not say.</param>
/// <param name="Done">True on the final report for this file.</param>
public sealed record FetchProgress(string FileName, long BytesRead, long TotalBytes, bool Done);
