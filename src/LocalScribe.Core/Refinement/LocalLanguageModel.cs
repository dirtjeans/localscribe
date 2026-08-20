namespace LocalScribe.Core.Refinement;

/// <summary>
/// Finds a local text-generation service for the cleanup stage.
/// <para>
/// There is more than one way to run a small model on these machines and no reason to make the
/// user care which. This probes the known backends and returns the first that answers, or null
/// when none does — in which case the pipeline still produces a transcript, just a raw one.
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
    /// Returns a reachable backend, or null. The caller owns the result and should dispose it.
    /// </summary>
    /// <param name="httpClient">Shared client, or null to let each candidate own one.</param>
    public static async Task<ILanguageModel?> ResolveAsync(
        HttpClient? httpClient = null,
        CancellationToken cancellationToken = default)
    {
        foreach (var candidate in Candidates(httpClient))
        {
            bool available;
            try
            {
                available = await candidate.IsAvailableAsync(cancellationToken).ConfigureAwait(false);
            }
            catch
            {
                (candidate as IDisposable)?.Dispose();
                throw;
            }

            if (available)
            {
                return candidate;
            }

            (candidate as IDisposable)?.Dispose();
        }

        return null;
    }

    private static IEnumerable<ILanguageModel> Candidates(HttpClient? httpClient)
    {
        // A shared HttpClient carries a single BaseAddress, so it cannot be handed to two
        // backends pointing at different ports. When one is supplied we honour it for the
        // preferred backend only, and let the fallback own its own.
        yield return new GenieXClient(httpClient: httpClient);
        yield return new FoundryLocalClient();
    }
}
