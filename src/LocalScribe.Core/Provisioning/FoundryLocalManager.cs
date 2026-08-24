using System.Text.RegularExpressions;

namespace LocalScribe.Core.Provisioning;

/// <summary>
/// Detects, installs, and starts Foundry Local, which provides the cleanup model.
/// <para>
/// Foundry is the practical way to reach the Hexagon NPU for text generation: it ships QNN
/// model variants and picks the right build for the hardware when given a model <em>alias</em>
/// rather than a fully qualified id. That last detail matters — passing a specific id pins you
/// to one hardware variant and quietly gives up the NPU.
/// </para>
/// </summary>
public sealed partial class FoundryLocalManager
{
    /// <summary>The winget package identifier.</summary>
    public const string WingetPackageId = "Microsoft.FoundryLocal";

    /// <summary>
    /// A small instruct model, given as an alias so Foundry picks the NPU build where one exists.
    /// Cleanup is a transformation rather than a reasoning task, so a 1.5B model is ample.
    /// </summary>
    public const string DefaultModelAlias = "qwen2.5-1.5b-instruct";

    private readonly IProcessRunner _runner;

    public FoundryLocalManager(IProcessRunner? runner = null)
    {
        _runner = runner ?? new ProcessRunner();
    }

    /// <summary>True when the foundry CLI is on the PATH.</summary>
    public async Task<bool> IsInstalledAsync(CancellationToken cancellationToken = default)
    {
        var result = await _runner
            .RunAsync("foundry", "--version", TimeSpan.FromSeconds(30), cancellationToken)
            .ConfigureAwait(false);

        return result.Succeeded;
    }

    /// <summary>
    /// Installs Foundry Local through winget.
    /// </summary>
    public async Task<InstallResult> InstallAsync(
        IProgress<InstallProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress?.Report(new InstallProgress("foundry", "Installing Foundry Local via winget…"));

        var result = await _runner.RunAsync(
            "winget",
            $"install --id {WingetPackageId} --accept-package-agreements --accept-source-agreements "
            + "--disable-interactivity --silent",
            TimeSpan.FromMinutes(20),
            cancellationToken).ConfigureAwait(false);

        if (result.NotFound)
        {
            return new InstallResult(
                "foundry",
                false,
                "winget is not available. Install App Installer from the Microsoft Store, or "
                + "install Foundry Local by hand from https://aka.ms/foundry-local-install");
        }

        if (!result.Succeeded)
        {
            return new InstallResult("foundry", false, $"winget failed: {Summarise(result.CombinedOutput)}");
        }

        return new InstallResult("foundry", true, "Foundry Local installed.");
    }

    /// <summary>Starts the service if it is not already running.</summary>
    public async Task<InstallResult> StartServiceAsync(
        IProgress<InstallProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress?.Report(new InstallProgress("foundry", "Starting the Foundry Local service…"));

        var result = await _runner
            .RunAsync("foundry", "service start", TimeSpan.FromMinutes(5), cancellationToken)
            .ConfigureAwait(false);

        return result.Succeeded
            ? new InstallResult("foundry", true, "Foundry Local service started.")
            : new InstallResult("foundry", false, $"Could not start the service: {Summarise(result.CombinedOutput)}");
    }

    /// <summary>
    /// Downloads a model into Foundry's cache.
    /// <para>
    /// This can take a while and moves a lot of data, so it reports progress by line rather than
    /// appearing to hang.
    /// </para>
    /// </summary>
    public async Task<InstallResult> DownloadModelAsync(
        string alias = DefaultModelAlias,
        IProgress<InstallProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress?.Report(new InstallProgress("foundry-model", $"Downloading {alias}. This may take a while…"));

        var result = await _runner.RunAsync(
            "foundry",
            $"model download {alias}",
            TimeSpan.FromMinutes(60),
            cancellationToken).ConfigureAwait(false);

        return result.Succeeded
            ? new InstallResult("foundry-model", true, $"{alias} is in the local cache.")
            : new InstallResult("foundry-model", false, $"Download failed: {Summarise(result.CombinedOutput)}");
    }

    /// <summary>
    /// Finds the endpoint the service is actually listening on.
    /// <para>
    /// Foundry Local binds to a <em>dynamic</em> port on loopback, so a hard-coded port works
    /// until the day it does not. The documented way to learn the real one is to ask the CLI.
    /// </para>
    /// </summary>
    /// <returns>The base URI, or <c>null</c> when the service is not running.</returns>
    public async Task<Uri?> DiscoverEndpointAsync(CancellationToken cancellationToken = default)
    {
        var result = await _runner
            .RunAsync("foundry", "service status", TimeSpan.FromSeconds(30), cancellationToken)
            .ConfigureAwait(false);

        return result.Succeeded ? ParseEndpoint(result.StandardOutput) : null;
    }

    /// <summary>
    /// Pulls the endpoint out of the CLI's status text, which is prose rather than a stable
    /// format. Matching a loopback URL anywhere in the output is more durable than parsing lines.
    /// </summary>
    internal static Uri? ParseEndpoint(string statusOutput)
    {
        if (string.IsNullOrWhiteSpace(statusOutput))
        {
            return null;
        }

        var match = EndpointPattern().Match(statusOutput);
        if (!match.Success)
        {
            return null;
        }

        return Uri.TryCreate(match.Value, UriKind.Absolute, out var uri) ? uri : null;
    }

    /// <summary>Keeps command output short enough to read in a console or a status bar.</summary>
    private static string Summarise(string output)
    {
        var lines = output
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(line => line.Length > 0)
            .ToList();

        if (lines.Count == 0)
        {
            return "no output";
        }

        // The last few lines carry the error; everything before is progress noise.
        var tail = lines.TakeLast(3);
        return string.Join(" / ", tail);
    }

    [GeneratedRegex(@"https?://(?:localhost|127\.0\.0\.1)(?::\d+)?(?:/v\d+)?", RegexOptions.IgnoreCase)]
    private static partial Regex EndpointPattern();
}
