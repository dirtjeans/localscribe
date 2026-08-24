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
    /// <summary>
    /// The port older builds listened on.
    /// <para>
    /// Kept only as the first thing <see cref="FoundryLocalDiscovery"/> probes. Current builds
    /// take an ephemeral port at startup, so anything that assumes this one finds nothing and
    /// concludes, wrongly, that no cleanup backend is installed.
    /// </para>
    /// </summary>
    public const int DefaultPort = FoundryLocalDiscovery.LegacyPort;

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
            NormaliseEndpoint(endpoint) ?? new Uri($"http://localhost:{DefaultPort}"),
            httpClient);

    /// <summary>
    /// Cuts a discovered endpoint back to its origin.
    /// <para>
    /// The service reports where it is listening including the API path, and every request here
    /// is made relative to the base address. Left in place, a base of <c>/v1</c> produces
    /// requests to <c>/v1/v1/chat/completions</c> — a 404 that reads exactly like the service not
    /// being there, which is the failure this whole discovery path exists to stop misreporting.
    /// </para>
    /// </summary>
    public static Uri? NormaliseEndpoint(Uri? endpoint) =>
        endpoint is null ? null : new Uri(endpoint.GetLeftPart(UriPartial.Authority) + "/");

    /// <summary>
    /// Finds a running service and a model it already has, or null when there is none.
    /// <para>
    /// Preferred over the constructor, which can only guess at both. Nothing is downloaded: a
    /// machine with the service running but no models cached returns null and the pipeline
    /// produces a raw transcript, exactly as it does on a machine with no service at all.
    /// </para>
    /// </summary>
    public static async Task<FoundryLocalClient?> DiscoverAsync(
        HttpClient? httpClient = null,
        CancellationToken cancellationToken = default)
    {
        if (await FoundryLocalDiscovery.FindEndpointAsync(httpClient, cancellationToken)
                .ConfigureAwait(false) is not { } endpoint)
        {
            return null;
        }

        if (await LocalModelCatalogue.ChooseAsync(endpoint, httpClient, cancellationToken)
                .ConfigureAwait(false) is not { } model)
        {
            return null;
        }

        return new FoundryLocalClient(model, endpoint, httpClient);
    }

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
