namespace LocalScribe.Core.Refinement;

/// <summary>
/// Finds a local text-generation service for the cleanup stage.
/// <para>
/// There is more than one way to run a small model on these machines and no reason to make the
/// user care which. This probes the known backends and returns the first that can do the job,
/// or null when none can — in which case the pipeline still produces a transcript, just a raw
/// one.
/// </para>
/// </summary>
public static class LocalLanguageModel
{
    /// <summary>
    /// The backends we know how to reach, in preference order.
    /// <para>
    /// GenieX comes first on purpose. Both can reach the NPU, but Foundry Local has had open
    /// bugs affecting NPU generation on Snapdragon X Elite, which is the machine this app is
    /// built for. Where both are running, the one less likely to crash mid-summary wins.
    /// </para>
    /// </summary>
    public static IReadOnlyList<string> BackendNames { get; } = new[] { "GenieX", "Foundry Local" };

    /// <summary>
    /// Returns a backend that can actually do the job, or null. The caller owns the result and
    /// should dispose it.
    /// <para>
    /// Each candidate is discovered rather than assumed: its endpoint is confirmed to answer,
    /// and the model is chosen from what that service already holds rather than named here in
    /// advance. Naming it in advance is a guess about someone else's machine, and both guesses
    /// this file used to make were wrong on the first machine they met — GenieX had a Gemma
    /// build and no Qwen, Foundry had three Phi builds and no Qwen.
    /// </para>
    /// <para>
    /// That mattered more than it sounds. Availability used to mean "is listening", so a service
    /// holding nothing we could use still answered yes, won the race against the backend that
    /// would have worked, and failed on the first window with a finished transcript in hand.
    /// </para>
    /// </summary>
    /// <param name="httpClient">Shared client, or null to let each candidate own one.</param>
    public static async Task<ILanguageModel?> ResolveAsync(
        HttpClient? httpClient = null,
        CancellationToken cancellationToken = default)
    {
        if (await GenieXClient.DiscoverAsync(httpClient, cancellationToken).ConfigureAwait(false)
            is { } genie)
        {
            return genie;
        }

        // A shared HttpClient carries a single BaseAddress, so it cannot be handed to two
        // backends pointing at different ports. The preferred one gets it; the fallback owns
        // its own.
        return await FoundryLocalClient
            .DiscoverAsync(cancellationToken: cancellationToken)
            .ConfigureAwait(false);
    }
}
