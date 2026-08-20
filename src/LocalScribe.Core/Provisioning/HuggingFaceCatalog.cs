using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace LocalScribe.Core.Provisioning;

/// <summary>How a repository lookup turned out.</summary>
public enum RepositoryLookup
{
    /// <summary>Hugging Face answered and described the repository.</summary>
    Found,

    /// <summary>Hugging Face answered, and no such repository is published.</summary>
    NotFound,

    /// <summary>
    /// Hugging Face could not be reached: no network, a proxy in the way, or a firewall.
    /// Distinct from <see cref="NotFound"/> because the remedies share nothing, and conflating
    /// them sends people to rewrite a repository list when their connection is the problem.
    /// </summary>
    Unreachable,
}

/// <summary>What a repository holds, or why we could not find out.</summary>
/// <param name="Outcome">Whether the lookup succeeded, and if not, why.</param>
/// <param name="Files">Every file path in the repository. Empty unless <paramref name="Outcome"/> is Found.</param>
public sealed record RepositoryContents(RepositoryLookup Outcome, IReadOnlyList<string> Files)
{
    public static RepositoryContents NotFound { get; } = new(RepositoryLookup.NotFound, []);

    public static RepositoryContents Unreachable { get; } = new(RepositoryLookup.Unreachable, []);
}

/// <summary>
/// Finds downloadable Whisper assets on Hugging Face.
/// <para>
/// The file list is queried at runtime rather than hard-coded. Publishers reorganise their
/// repositories, and a baked-in path that has since moved fails as a 404 during setup — the
/// least helpful moment. Asking the repository what it contains costs one small request and
/// survives renames.
/// </para>
/// </summary>
public sealed class HuggingFaceCatalog
{
    private const string ApiRoot = "https://huggingface.co/api/models/";
    private const string DownloadRoot = "https://huggingface.co/";

    private readonly HttpClient _http;
    private readonly bool _ownsHttpClient;

    public HuggingFaceCatalog(HttpClient? httpClient = null)
    {
        _ownsHttpClient = httpClient is null;
        _http = httpClient ?? new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
    }

    /// <summary>
    /// Repositories to search, in preference order, for a given Whisper size.
    /// <para>
    /// Qualcomm publishes precompiled QNN builds for Snapdragon; the openai repositories hold
    /// the reference weights and, importantly, the <c>vocab.json</c> that the QNN exports do not
    /// always carry.
    /// </para>
    /// </summary>
    public static IReadOnlyList<string> RepositoriesFor(string whisperModel) => whisperModel switch
    {
        "tiny.en" => ["qualcomm/Whisper-Tiny-En", "qualcomm/Whisper-Tiny", "openai/whisper-tiny.en"],
        "base.en" => ["qualcomm/Whisper-Base-En", "qualcomm/Whisper-Base", "openai/whisper-base.en"],
        "small.en" => ["qualcomm/Whisper-Small-En", "qualcomm/Whisper-Small", "openai/whisper-small.en"],
        "medium.en" => ["qualcomm/Whisper-Medium-En", "openai/whisper-medium.en"],
        _ => ["openai/whisper-base.en"],
    };

    /// <summary>
    /// Asks a repository what it contains.
    /// <para>
    /// The two ways this fails are kept apart deliberately. A repository that does not exist is
    /// an ordinary outcome when walking a preference list and means try the next one. A network
    /// that cannot be reached means every remaining entry will fail the same way, and the person
    /// running setup needs to hear about their connection rather than about our model catalogue.
    /// </para>
    /// </summary>
    public async Task<RepositoryContents> LookUpAsync(
        string repository,
        CancellationToken cancellationToken = default)
    {
        try
        {
            using var response = await _http
                .GetAsync(ApiRoot + repository, cancellationToken)
                .ConfigureAwait(false);

            // A gated repository answers 401 or 403. From here that is indistinguishable in
            // remedy from one that does not exist: either way this build is not available to us.
            if (response.StatusCode is HttpStatusCode.NotFound
                or HttpStatusCode.Unauthorized
                or HttpStatusCode.Forbidden)
            {
                return RepositoryContents.NotFound;
            }

            response.EnsureSuccessStatusCode();

            var info = await response.Content
                .ReadFromJsonAsync<RepositoryInfo>(cancellationToken)
                .ConfigureAwait(false);

            var files = info?.Siblings?
                .Select(s => s.FileName)
                .Where(n => n.Length > 0)
                .ToList() ?? [];

            return new RepositoryContents(RepositoryLookup.Found, files);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            // The user stopped setup. That is not a verdict on the network.
            throw;
        }
        catch (Exception exception)
            when (exception is HttpRequestException or TaskCanceledException or JsonException)
        {
            // JsonException belongs here rather than with a malformed repository: a captive
            // portal or proxy answering with an HTML error page is a connectivity problem
            // wearing a 200.
            return RepositoryContents.Unreachable;
        }
    }

    /// <summary>Builds the download URL for one file in a repository.</summary>
    public static string DownloadUrl(string repository, string path) =>
        $"{DownloadRoot}{repository}/resolve/main/{path}";

    /// <summary>
    /// Picks the set of files to download for a chipset and model size.
    /// <para>
    /// Repositories that publish for several chipsets keep them in separate folders, and a
    /// binary from the wrong folder will not load. So a chipset-specific folder wins outright
    /// when one exists; otherwise the repository is assumed to hold a single build.
    /// </para>
    /// </summary>
    /// <param name="paths">Every path in the repository.</param>
    /// <param name="chipsetSlug">Folder name for the target chipset, e.g. <c>snapdragon-x-elite</c>.</param>
    /// <returns>Paths to download, or empty when no usable set was found.</returns>
    public static IReadOnlyList<string> SelectAssets(IEnumerable<string> paths, string chipsetSlug)
    {
        ArgumentNullException.ThrowIfNull(paths);

        var all = paths.ToList();
        var candidates = all.Where(IsInteresting).ToList();

        if (candidates.Count == 0)
        {
            return [];
        }

        var chipsetFolder = FindChipsetFolder(candidates, chipsetSlug);
        if (chipsetFolder is not null)
        {
            var inFolder = candidates
                .Where(p => p.StartsWith(chipsetFolder + "/", StringComparison.OrdinalIgnoreCase))
                .ToList();

            // The vocabulary usually sits at the repository root rather than beside the graphs.
            var rootVocab = candidates.Where(p => IsVocab(p) && !p.Contains('/'));

            return [.. inFolder.Concat(rootVocab).Distinct(StringComparer.OrdinalIgnoreCase)];
        }

        return candidates;
    }

    /// <summary>
    /// Locates a directory whose name identifies the target chipset. Comparison ignores the
    /// separators publishers vary on: <c>snapdragon-x-elite</c>, <c>snapdragon_x_elite</c>, and
    /// <c>SnapdragonXElite</c> all describe the same silicon.
    /// </summary>
    internal static string? FindChipsetFolder(IEnumerable<string> paths, string chipsetSlug)
    {
        var wanted = Simplify(chipsetSlug);
        if (wanted.Length == 0)
        {
            return null;
        }

        var folders = paths
            .Where(p => p.Contains('/'))
            .Select(p => p[..p.LastIndexOf('/')])
            .Distinct(StringComparer.OrdinalIgnoreCase);

        return folders
            .Where(folder => folder
                .Split('/')
                .Any(segment => Simplify(segment).Equals(wanted, StringComparison.Ordinal)))
            .OrderByDescending(f => f.Length)
            .FirstOrDefault();
    }

    private static string Simplify(string value) =>
        new([.. value.Where(char.IsLetterOrDigit).Select(char.ToLowerInvariant)]);

    private static bool IsInteresting(string path) =>
        path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase)
        || path.EndsWith(".onnx_data", StringComparison.OrdinalIgnoreCase)
        || path.EndsWith(".bin", StringComparison.OrdinalIgnoreCase)
        || IsVocab(path);

    private static bool IsVocab(string path) =>
        Path.GetFileName(path).Equals("vocab.json", StringComparison.OrdinalIgnoreCase);

    public void Dispose()
    {
        if (_ownsHttpClient)
        {
            _http.Dispose();
        }
    }

    private sealed record RepositorySibling(
        [property: JsonPropertyName("rfilename")] string FileName);

    private sealed record RepositoryInfo(
        [property: JsonPropertyName("siblings")] IReadOnlyList<RepositorySibling>? Siblings);
}
