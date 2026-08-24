using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace LocalScribe.Core.Provisioning;

/// <summary>One downloadable file attached to a GitHub release.</summary>
/// <param name="Name">File name.</param>
/// <param name="DownloadUrl">Direct download URL.</param>
/// <param name="SizeBytes">Size as reported by the API.</param>
public sealed record ReleaseAsset(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("browser_download_url")] string DownloadUrl,
    [property: JsonPropertyName("size")] long SizeBytes);

/// <summary>
/// Reads the asset list from a GitHub release.
/// <para>
/// sherpa-onnx distributes its models as release assets, and the exact file names change as
/// models are added and revised. Asking the release what it holds beats hard-coding a name that
/// was correct when this was written.
/// </para>
/// </summary>
public sealed class GitHubReleaseCatalog : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttpClient;

    public GitHubReleaseCatalog(HttpClient? httpClient = null)
    {
        _ownsHttpClient = httpClient is null;
        _http = httpClient ?? new HttpClient { Timeout = TimeSpan.FromMinutes(2) };

        // The API rejects requests with no user agent.
        if (!_http.DefaultRequestHeaders.UserAgent.Any())
        {
            _http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("LocalScribe", "1.0"));
        }
    }

    /// <summary>
    /// Lists a release's assets by tag. Returns empty when the tag does not exist, which is the
    /// normal outcome when walking a list of candidate tags.
    /// </summary>
    public async Task<IReadOnlyList<ReleaseAsset>> ListAssetsAsync(
        string owner,
        string repository,
        string tag,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var release = await _http
                .GetFromJsonAsync<ReleaseInfo>(
                    $"https://api.github.com/repos/{owner}/{repository}/releases/tags/{tag}",
                    cancellationToken)
                .ConfigureAwait(false);

            return release?.Assets ?? [];
        }
        catch (Exception exception) when (exception is HttpRequestException or TaskCanceledException)
        {
            return [];
        }
    }

    /// <summary>
    /// Picks the first asset matching a preference-ordered list of name fragments.
    /// <para>
    /// Matching on fragments rather than whole names survives the version suffixes and date
    /// stamps that these files carry, while still keeping the choice explicit and ordered.
    /// </para>
    /// </summary>
    public static ReleaseAsset? PickByPreference(
        IEnumerable<ReleaseAsset> assets,
        IReadOnlyList<string> preferredFragments,
        string? requiredExtension = null)
    {
        ArgumentNullException.ThrowIfNull(assets);
        ArgumentNullException.ThrowIfNull(preferredFragments);

        var candidates = assets
            .Where(a => requiredExtension is null
                || a.Name.EndsWith(requiredExtension, StringComparison.OrdinalIgnoreCase))
            .ToList();

        foreach (var fragment in preferredFragments)
        {
            var match = candidates
                .Where(a => a.Name.Contains(fragment, StringComparison.OrdinalIgnoreCase))
                // Shortest name first: among several variants of the same model, the plain one
                // has the fewest qualifiers appended.
                .OrderBy(a => a.Name.Length)
                .ThenBy(a => a.Name, StringComparer.OrdinalIgnoreCase)
                .FirstOrDefault();

            if (match is not null)
            {
                return match;
            }
        }

        return null;
    }

    public void Dispose()
    {
        if (_ownsHttpClient)
        {
            _http.Dispose();
        }
    }

    private sealed record ReleaseInfo(
        [property: JsonPropertyName("assets")] IReadOnlyList<ReleaseAsset>? Assets);
}
