namespace LocalScribe.Core.Refinement;

/// <summary>
/// Talks to a local GenieX service, Qualcomm's on-device generative runtime.
/// <para>
/// GenieX runs LLMs across the Hexagon NPU, the Adreno GPU, and the Oryon cores, and serves an
/// OpenAI-compatible API on loopback, so it needs nothing here beyond a port and a model name.
/// It is preferred over Foundry Local on Snapdragon because Foundry has had open bugs affecting
/// NPU generation on X Elite, and a cleanup stage that crashes is worse than one that is merely
/// slower.
/// </para>
/// <para>
/// It covers this stage only. GenieX runs language and vision-language models; it has no
/// speech-to-text path, so Whisper still goes through ONNX Runtime.
/// </para>
/// </summary>
public sealed class GenieXClient : ILanguageModel, IDisposable
{
    /// <summary>GenieX's default local server port.</summary>
    public const int DefaultPort = 18181;

    /// <summary>
    /// A small instruct model, which is all this stage needs. Punctuation repair and a summary
    /// are transformations of text that is already there, not open-ended generation.
    /// </summary>
    public const string DefaultModel = "qwen2.5-1.5b-instruct";

    private readonly OpenAiCompatibleClient _inner;

    public GenieXClient(
        string model = DefaultModel,
        Uri? endpoint = null,
        HttpClient? httpClient = null) =>
        _inner = new OpenAiCompatibleClient(
            "GenieX",
            model,
            endpoint ?? new Uri($"http://127.0.0.1:{DefaultPort}"),
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
