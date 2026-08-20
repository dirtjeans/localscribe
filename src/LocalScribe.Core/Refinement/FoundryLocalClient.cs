using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace LocalScribe.Core.Refinement;

/// <summary>
/// Talks to a locally running Foundry Local service over its OpenAI-compatible HTTP API.
/// <para>
/// Foundry Local is the practical way to reach the Hexagon NPU for text generation on these
/// machines: it ships QNN model variants and picks the right one for the hardware when given a
/// model alias rather than a fully qualified id. Nothing here leaves the machine — the endpoint
/// is on localhost.
/// </para>
/// </summary>
public sealed class FoundryLocalClient : ILanguageModel, IDisposable
{
    /// <summary>
    /// A fallback port, used only when the service could not be asked where it is listening.
    /// <para>
    /// Foundry Local binds to a <em>dynamic</em> loopback port, so this constant is a last
    /// resort rather than a default. Prefer passing the endpoint from
    /// <c>FoundryLocalManager.DiscoverEndpointAsync</c>, which asks the CLI for the real one.
    /// </para>
    /// </summary>
    public const int FallbackPort = 5273;

    /// <summary>
    /// An alias rather than a specific model id, on purpose. Passing an alias lets Foundry pick
    /// the NPU build on Snapdragon and fall back to CPU elsewhere, without us hard-coding either.
    /// </summary>
    public const string DefaultModelAlias = "qwen2.5-1.5b-instruct";

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    private readonly HttpClient _http;
    private readonly bool _ownsHttpClient;
    private readonly string _model;

    public FoundryLocalClient(
        string model = DefaultModelAlias,
        Uri? endpoint = null,
        HttpClient? httpClient = null)
    {
        _model = model;
        _ownsHttpClient = httpClient is null;
        _http = httpClient ?? new HttpClient();
        _http.BaseAddress ??= NormaliseEndpoint(endpoint) ?? new Uri($"http://localhost:{FallbackPort}");

        // Cleanup on a small local model is quick, but a cold start has to load weights first.
        if (_http.Timeout == TimeSpan.FromSeconds(100))
        {
            _http.Timeout = TimeSpan.FromMinutes(3);
        }
    }

    public string Description => $"{_model} via Foundry Local";

    public async Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            using var response = await _http
                .GetAsync("v1/models", cancellationToken)
                .ConfigureAwait(false);
            return response.IsSuccessStatusCode;
        }
        catch (Exception exception) when (exception is HttpRequestException or TaskCanceledException)
        {
            // A missing service is the normal case on a machine that has not set one up, so this
            // is an expected answer rather than an error worth propagating.
            return false;
        }
    }

    public async Task<string> CompleteAsync(
        string systemPrompt,
        string userPrompt,
        int maxTokens = 1024,
        CancellationToken cancellationToken = default)
    {
        var request = new ChatRequest(
            _model,
            [new ChatMessage("system", systemPrompt), new ChatMessage("user", userPrompt)],
            maxTokens,
            // Cleanup is a transformation, not a creative task. Near-zero temperature keeps the
            // model from paraphrasing the speaker.
            Temperature: 0.1);

        using var response = await _http
            .PostAsJsonAsync("v1/chat/completions", request, JsonOptions, cancellationToken)
            .ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            throw new InvalidOperationException(
                $"Foundry Local returned {(int)response.StatusCode}: {Truncate(body, 400)}");
        }

        var parsed = await response.Content
            .ReadFromJsonAsync<ChatResponse>(JsonOptions, cancellationToken)
            .ConfigureAwait(false);

        return parsed?.Choices is { Count: > 0 } choices
            ? choices[0].Message.Content
            : string.Empty;
    }

    private static string Truncate(string value, int limit) =>
        value.Length <= limit ? value : value[..limit] + "…";

    /// <summary>
    /// Trims a discovered endpoint back to its origin and gives it a trailing slash.
    /// <para>
    /// The CLI's status text sometimes includes an API path such as <c>/v1</c>. Left in place
    /// that produces requests to <c>/v1/v1/chat/completions</c>, because a relative request URI
    /// is resolved against the base address rather than appended to it.
    /// </para>
    /// </summary>
    internal static Uri? NormaliseEndpoint(Uri? endpoint) =>
        endpoint is null ? null : new Uri(endpoint.GetLeftPart(UriPartial.Authority) + "/");

    public void Dispose()
    {
        if (_ownsHttpClient)
        {
            _http.Dispose();
        }
    }

    private sealed record ChatMessage(
        [property: JsonPropertyName("role")] string Role,
        [property: JsonPropertyName("content")] string Content);

    private sealed record ChatRequest(
        [property: JsonPropertyName("model")] string Model,
        [property: JsonPropertyName("messages")] IReadOnlyList<ChatMessage> Messages,
        [property: JsonPropertyName("max_tokens")] int MaxTokens,
        [property: JsonPropertyName("temperature")] double Temperature);

    private sealed record ChatChoice(
        [property: JsonPropertyName("message")] ChatMessage Message);

    private sealed record ChatResponse(
        [property: JsonPropertyName("choices")] IReadOnlyList<ChatChoice> Choices);
}
