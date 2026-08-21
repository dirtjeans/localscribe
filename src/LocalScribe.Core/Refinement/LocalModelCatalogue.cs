using System.Net.Http.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace LocalScribe.Core.Refinement;

/// <summary>
/// Asks a local OpenAI-compatible service what models it has, and picks one for the cleanup
/// stage.
/// <para>
/// Every backend here used to name a model as a constant, which is a guess about someone else's
/// machine. Both guesses were wrong on the first machine they met: GenieX had a Gemma build and
/// no Qwen, Foundry had three Phi builds and no Qwen, and each service answered "yes, I am
/// running" to a question about availability that never mentioned the model — so the pipeline
/// picked a backend that then failed on the very first window.
/// </para>
/// <para>
/// Choosing from what is actually cached is the whole fix. Nothing is downloaded to make a
/// choice possible: a service with no models is treated exactly like a service that is not
/// running, and the transcript comes back raw.
/// </para>
/// </summary>
public static partial class LocalModelCatalogue
{
    /// <summary>
    /// The best model this service already has, or null if it has none or cannot be reached.
    /// <para>
    /// Preference goes to builds named for the NPU, then to smaller ones, then to the longest
    /// context. Cleanup is a mechanical rewrite repeated over many short windows, so throughput
    /// decides how long the user waits far more than reasoning power decides quality, and a
    /// longer context means fewer windows and so fewer seams to stitch.
    /// </para>
    /// </summary>
    public static async Task<string?> ChooseAsync(
        Uri endpoint,
        HttpClient? httpClient = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(endpoint);

        var http = httpClient ?? Shared;

        try
        {
            var catalogue = await http
                .GetFromJsonAsync<ModelList>(new Uri(endpoint, "v1/models"), cancellationToken)
                .ConfigureAwait(false);

            if (catalogue?.Data is not { Count: > 0 } models)
            {
                return null;
            }

            return models
                .Where(model => !string.IsNullOrWhiteSpace(model.Id))
                .OrderByDescending(model => RunsOnNpu(model.Id))
                .ThenBy(model => BillionsOfParameters(model.Id))
                .ThenByDescending(model => model.MaxInputTokens)
                .FirstOrDefault()
                ?.Id;
        }
        catch (Exception exception) when (exception is HttpRequestException or TaskCanceledException or NotSupportedException or System.Text.Json.JsonException)
        {
            return null;
        }
    }

    /// <summary>True for builds named for the Hexagon NPU.</summary>
    internal static bool RunsOnNpu(string modelId) =>
        modelId.Contains("qnn", StringComparison.OrdinalIgnoreCase)
        || modelId.Contains("npu", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Parameter count read off the model id, which is where these runtimes put it. An
    /// unparseable name sorts as mid-range rather than as zero, so that a model we cannot
    /// measure does not win by appearing to be the smallest thing on offer.
    /// </summary>
    internal static double BillionsOfParameters(string modelId)
    {
        if (SizeInName().Match(modelId) is { Success: true } match
            && double.TryParse(
                match.Groups[1].Value,
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture,
                out var billions))
        {
            return billions;
        }

        // "mini" is the usual name for the 3-4B tier and is common enough to be worth knowing.
        return modelId.Contains("mini", StringComparison.OrdinalIgnoreCase) ? 3.8 : 8.0;
    }

    [GeneratedRegex(@"(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", RegexOptions.IgnoreCase)]
    private static partial Regex SizeInName();

    private static readonly HttpClient Shared = new() { Timeout = TimeSpan.FromSeconds(10) };

    private sealed record ModelList(
        [property: JsonPropertyName("data")] IReadOnlyList<ModelEntry> Data);

    private sealed record ModelEntry(
        [property: JsonPropertyName("id")] string Id,
        [property: JsonPropertyName("maxInputTokens")] int MaxInputTokens);
}
