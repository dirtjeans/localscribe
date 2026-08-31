using System.Diagnostics;
using System.Text.RegularExpressions;

namespace LocalScribe.Core.Refinement;

/// <summary>
/// Finds the port a running Foundry Local service is listening on.
/// <para>
/// Needed because there isn't a fixed one. Foundry Local takes an ephemeral port at startup, so
/// the 5273 this app used to try was right only for old builds, and every machine since has
/// looked to us like a machine with no cleanup backend installed at all.
/// </para>
/// <para>
/// Which model to then ask it for is <see cref="LocalModelCatalogue"/>'s problem.
/// </para>
/// </summary>
public static partial class FoundryLocalDiscovery
{
    /// <summary>
    /// The port older builds used, tried first because probing it costs one refused connection
    /// and succeeding avoids launching a process.
    /// </summary>
    public const int LegacyPort = 5273;

    /// <summary>Where the service says it is, or null when it is not running.</summary>
    /// <param name="httpClient">Shared client for the probe, or null to use a private one.</param>
    public static async Task<Uri?> FindEndpointAsync(
        HttpClient? httpClient = null,
        CancellationToken cancellationToken = default)
    {
        var http = httpClient ?? SharedProbe;

        var legacy = new Uri($"http://127.0.0.1:{LegacyPort}");
        if (await RespondsAsync(http, legacy, cancellationToken).ConfigureAwait(false))
        {
            return legacy;
        }

        // Ask the CLI. It is the supported way to learn the endpoint and it is installed
        // wherever the service is, being the thing that starts it.
        if (await AskTheCliAsync(cancellationToken).ConfigureAwait(false) is not { } reported)
        {
            return null;
        }

        return await RespondsAsync(http, reported, cancellationToken).ConfigureAwait(false)
            ? reported
            : null;
    }

    [GeneratedRegex(@"https?://(?:127\.0\.0\.1|localhost):(\d+)", RegexOptions.IgnoreCase)]
    private static partial Regex EndpointInOutput();

    private static async Task<bool> RespondsAsync(
        HttpClient http,
        Uri endpoint,
        CancellationToken cancellationToken)
    {
        try
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(ProbeTimeout);

            using var response = await http
                .GetAsync(new Uri(endpoint, "v1/models"), timeout.Token)
                .ConfigureAwait(false);

            return response.IsSuccessStatusCode;
        }
        catch (Exception exception) when (exception is HttpRequestException or TaskCanceledException or OperationCanceledException)
        {
            return false;
        }
    }

    /// <summary>
    /// Runs <c>foundry service status</c> and reads the endpoint out of what it prints.
    /// <para>
    /// Parsing console output is not how one would choose to learn this, but the alternatives
    /// are worse: the port lives in a packaged app's settings store, and the official SDK is a
    /// dependency an order of magnitude larger than the thing it would tell us. This runs once
    /// per session, off the UI thread, and treats every failure as "not running".
    /// </para>
    /// </summary>
    private static async Task<Uri?> AskTheCliAsync(CancellationToken cancellationToken)
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo(Provisioning.FoundryCli.Path, "service status")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            });

            if (process is null)
            {
                return null;
            }

            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(CliTimeout);

            var output = await process.StandardOutput.ReadToEndAsync(timeout.Token).ConfigureAwait(false);
            await process.WaitForExitAsync(timeout.Token).ConfigureAwait(false);

            return EndpointInOutput().Match(output) is { Success: true } match
                && int.TryParse(match.Groups[1].Value, out var port)
                ? new Uri($"http://127.0.0.1:{port}")
                : null;
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            // Not installed, not on PATH, or refusing to run. All mean the same thing here.
            return null;
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return null;
        }
    }

    /// <summary>Long enough for a loopback round trip, short enough not to delay startup.</summary>
    private static readonly TimeSpan ProbeTimeout = TimeSpan.FromSeconds(2);

    /// <summary>The CLI is a .NET app and takes a moment to start before it answers.</summary>
    private static readonly TimeSpan CliTimeout = TimeSpan.FromSeconds(20);

    private static readonly HttpClient SharedProbe = new() { Timeout = TimeSpan.FromSeconds(5) };
}
