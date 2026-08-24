using System.Security.Cryptography;

namespace LocalScribe.Core.Provisioning;

/// <summary>A file available for download.</summary>
/// <param name="Url">Where to fetch it.</param>
/// <param name="FileName">What to call it locally.</param>
/// <param name="SizeBytes">Expected size, when the catalogue reported one.</param>
/// <param name="Sha256">Expected hash in lowercase hex, when the catalogue reported one.</param>
public sealed record RemoteFile(string Url, string FileName, long? SizeBytes = null, string? Sha256 = null);

/// <summary>
/// Fetches files. Abstracted so the installer's decisions can be tested without a network.
/// </summary>
public interface IFileDownloader
{
    /// <summary>
    /// Downloads one file into a directory, replacing any existing copy.
    /// </summary>
    Task DownloadAsync(
        RemoteFile file,
        string targetDirectory,
        IProgress<InstallProgress>? progress = null,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Downloads over HTTPS, writing to a temporary file and moving it into place only once the
/// whole transfer has succeeded and any published hash matches.
/// <para>
/// The atomic move is the important part. A model download is hundreds of megabytes over a
/// laptop's Wi-Fi, and a partial file left at the final path is worse than no file: the app
/// finds it, decides the model is present, and then fails deep inside ONNX Runtime with an
/// error that says nothing about the real cause.
/// </para>
/// </summary>
public sealed class HttpFileDownloader : IFileDownloader, IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttpClient;

    public HttpFileDownloader(HttpClient? httpClient = null)
    {
        _ownsHttpClient = httpClient is null;
        _http = httpClient ?? new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
    }

    public async Task DownloadAsync(
        RemoteFile file,
        string targetDirectory,
        IProgress<InstallProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(file);

        Directory.CreateDirectory(targetDirectory);

        var finalPath = Path.Combine(targetDirectory, file.FileName);
        var temporaryPath = finalPath + ".partial";

        try
        {
            using var response = await _http
                .GetAsync(file.Url, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
                .ConfigureAwait(false);

            response.EnsureSuccessStatusCode();

            var total = response.Content.Headers.ContentLength ?? file.SizeBytes;

            await using (var source = await response.Content
                .ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
            await using (var destination = File.Create(temporaryPath))
            {
                await CopyWithProgressAsync(
                    source, destination, total, file.FileName, progress, cancellationToken)
                    .ConfigureAwait(false);
            }

            VerifyHash(temporaryPath, file, progress);

            File.Move(temporaryPath, finalPath, overwrite: true);
        }
        catch
        {
            // Never leave a partial file where the app would mistake it for a working model.
            TryDelete(temporaryPath);
            throw;
        }
    }

    private static async Task CopyWithProgressAsync(
        Stream source,
        Stream destination,
        long? total,
        string fileName,
        IProgress<InstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        var buffer = new byte[81920];
        long copied = 0;
        var lastReportedPercent = -1;
        int read;

        while ((read = await source.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
        {
            await destination.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
            copied += read;

            if (progress is null || total is not > 0)
            {
                continue;
            }

            // Reporting every buffer would flood the console and the UI thread alike.
            var percent = (int)(copied * 100 / total.Value);
            if (percent == lastReportedPercent)
            {
                continue;
            }

            lastReportedPercent = percent;
            progress.Report(new InstallProgress(
                "download",
                $"{fileName} — {copied / (1024 * 1024)} of {total.Value / (1024 * 1024)} MB",
                percent / 100.0));
        }
    }

    /// <summary>
    /// Checks the published hash when there is one, and says so plainly when there is not.
    /// Silence about an unverified download would be the wrong kind of quiet.
    /// </summary>
    private static void VerifyHash(string path, RemoteFile file, IProgress<InstallProgress>? progress)
    {
        if (file.Sha256 is not { Length: > 0 } expected)
        {
            progress?.Report(new InstallProgress(
                "download",
                $"{file.FileName} downloaded. The catalogue published no checksum, so its contents "
                + "could not be verified."));
            return;
        }

        using var stream = File.OpenRead(path);
        var actual = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();

        if (!actual.Equals(expected.ToLowerInvariant(), StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"{file.FileName} failed its checksum. Expected {expected}, got {actual}. "
                + "The download was corrupted or the file has changed upstream.");
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (IOException)
        {
            // Cleanup is best effort; the caller is already handling a failure.
        }
    }

    public void Dispose()
    {
        if (_ownsHttpClient)
        {
            _http.Dispose();
        }
    }
}
