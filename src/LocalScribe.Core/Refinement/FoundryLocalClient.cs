namespace LocalScribe.Core.Refinement;

/// <summary>
/// Talks to a locally running Foundry Local service over its OpenAI-compatible HTTP API.
/// <para>
/// Foundry Local ships QNN model variants and picks the right one for the hardware when given a
/// model alias rather than a fully qualified id. Nothing here leaves the machine — the endpoint
/// is on localhost.
/// </para>
/// <para>
/// Foundry has had open bugs affecting NPU generation on Snapdragon X Elite, so
/// <see cref="LocalLanguageModel"/> tries GenieX first and falls back to this.
/// </para>
/// </summary>
public sealed class FoundryLocalClient : ILanguageModel, IDisposable
{
    /// <summary>Foundry Local's default port.</summary>
    public const int DefaultPort = 5273;

    /// <summary>
    /// An alias rather than a specific model id, on purpose. Passing an alias lets Foundry pick
    /// the NPU build on Snapdragon and fall back to CPU elsewhere, without us hard-coding either.
    /// </summary>
    public const string DefaultModelAlias = "qwen2.5-1.5b-instruct";

    private readonly OpenAiCompatibleClient _inner;

    public FoundryLocalClient(
        string model = DefaultModelAlias,
        Uri? endpoint = null,
        HttpClient? httpClient = null) =>
        _inner = new OpenAiCompatibleClient(
            "Foundry Local",
            model,
            endpoint ?? new Uri($"http://localhost:{DefaultPort}"),
            httpClient);

    public string Description => _inner.Description;

    public Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default) =>
        _inner.IsAvailableAsync(cancellationToken);

    public Task<string> CompleteAsync(
        string systemPrompt,
        string userPrompt,
        int maxTokens = 1024,
        CancellationToken cancellationToken = default) =>
        _inner.CompleteAsync(systemPrompt, userPrompt, maxTokens, cancellationToken);

    public void Dispose() => _inner.Dispose();
}
