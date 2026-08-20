namespace LocalScribe.Core.Refinement;

/// <summary>
/// A local text-generation backend used for the cleanup pass. Deliberately narrow: the pipeline
/// only ever needs a prompt in and text out, so a different runtime can be swapped in without
/// touching anything above it.
/// </summary>
public interface ILanguageModel
{
    /// <summary>What is running, e.g. "qwen2.5-1.5b-instruct-qnn-npu via Foundry Local".</summary>
    string Description { get; }

    /// <summary>True when the service answered a health check. False disables the cleanup stage.</summary>
    Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default);

    /// <summary>Runs one completion.</summary>
    /// <param name="systemPrompt">Instructions describing the task.</param>
    /// <param name="userPrompt">The transcript text to work on.</param>
    /// <param name="maxTokens">Upper bound on the reply length.</param>
    Task<string> CompleteAsync(
        string systemPrompt,
        string userPrompt,
        int maxTokens = 1024,
        CancellationToken cancellationToken = default);
}
