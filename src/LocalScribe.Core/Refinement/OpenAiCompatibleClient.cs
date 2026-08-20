using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace LocalScribe.Core.Refinement;

/// <summary>
/// Talks to a local text-generation service over an OpenAI-compatible HTTP API.
/// <para>
/// Every local runtime worth using on these machines speaks this dialect, so the transport is
/// written once and the backends differ only in a port, a model name, and what to call the
/// thing in an error message. Nothing here leaves the machine: the endpoint is on loopback.
/// </para>
/// </summary>
public sealed class OpenAiCompatibleClient : ILanguageModel, IDisposable
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    private readonly HttpClient _http;
    private readonly bool _ownsHttpClient;
    private readonly string _model;
    private readonly string _serviceName;

    /// <param name="serviceName">How to name this backend in descriptions and errors.</param>
    /// <param name="model">Model id or alias to request.</param>
    /// <param name="endpoint">Base address. Must be loopback.</param>
    /// <param name="httpClient">Supply one to share a pool, or leave null to own a private one.</param>
    public OpenAiCompatibleClient(
        string serviceName,
        string model,
        Uri endpoint,
        HttpClient? httpClient = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(serviceName);
        ArgumentException.ThrowIfNullOrEmpty(model);
        ArgumentNullException.ThrowIfNull(endpoint);

        if (!endpoint.IsLoopback)
        {
            // The promise this app makes is that nothing leaves the machine. A cleanup backend
            // pointed at a remote host would break that quietly, in the one stage that sees the
            // whole transcript at once.
            throw new ArgumentException(
                $"The cleanup backend must be local, but {endpoint} is not a loopback address.",
                nameof(endpoint));
        }

        _serviceName = serviceName;
        _model = model;
        _ownsHttpClient = httpClient is null;
        _http = httpClient ?? new HttpClient();
        _http.BaseAddress ??= endpoint;

        // Cleanup on a small local model is quick, but a cold start has to load weights first.
        if (_http.Timeout == TimeSpan.FromSeconds(100))
        {
            _http.Timeout = TimeSpan.FromMinutes(3);
        }
    }

    public string Description => $"{_model} via {_serviceName}";

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
                $"{_serviceName} returned {(int)response.StatusCode}: {Truncate(body, 400)}");
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
