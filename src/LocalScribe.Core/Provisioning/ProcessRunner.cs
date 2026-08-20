using System.Diagnostics;

namespace LocalScribe.Core.Provisioning;

/// <param name="ExitCode">Process exit code, or -1 when the executable could not be started.</param>
/// <param name="StandardOutput">Everything written to stdout.</param>
/// <param name="StandardError">Everything written to stderr.</param>
public sealed record ProcessResult(int ExitCode, string StandardOutput, string StandardError)
{
    public bool Succeeded => ExitCode == 0;

    /// <summary>True when the executable itself was not found, as opposed to running and failing.</summary>
    public bool NotFound => ExitCode == -1;

    /// <summary>Both streams together, for error messages.</summary>
    public string CombinedOutput =>
        string.Join(Environment.NewLine, new[] { StandardOutput, StandardError }
            .Where(s => !string.IsNullOrWhiteSpace(s)));

    /// <summary>The result used when an executable is absent.</summary>
    public static ProcessResult Missing { get; } = new(-1, string.Empty, string.Empty);
}

/// <summary>
/// Runs external commands. Abstracted because the installer's logic — deciding what to run and
/// what to make of the output — is worth testing without actually installing anything.
/// </summary>
public interface IProcessRunner
{
    Task<ProcessResult> RunAsync(
        string fileName,
        string arguments,
        TimeSpan? timeout = null,
        CancellationToken cancellationToken = default);
}

/// <summary>Runs commands as child processes.</summary>
public sealed class ProcessRunner : IProcessRunner
{
    private static readonly TimeSpan DefaultTimeout = TimeSpan.FromMinutes(15);

    public async Task<ProcessResult> RunAsync(
        string fileName,
        string arguments,
        TimeSpan? timeout = null,
        CancellationToken cancellationToken = default)
    {
        using var process = new Process
        {
            StartInfo = new ProcessStartInfo(fileName, arguments)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            },
        };

        try
        {
            if (!process.Start())
            {
                return ProcessResult.Missing;
            }
        }
        catch (Exception exception)
            when (exception is System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            // The command is not installed. That is a normal finding here, not an error.
            return ProcessResult.Missing;
        }

        // Read both streams concurrently. Reading them in sequence deadlocks as soon as a
        // command writes more than a pipe buffer to the stream we are not reading yet, and
        // package managers are chatty enough to hit that.
        var stdoutTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
        var stderrTask = process.StandardError.ReadToEndAsync(cancellationToken);

        using var timeoutSource = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutSource.CancelAfter(timeout ?? DefaultTimeout);

        try
        {
            await process.WaitForExitAsync(timeoutSource.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            TryKill(process);
            throw new TimeoutException($"'{fileName} {arguments}' did not finish in time.");
        }

        return new ProcessResult(
            process.ExitCode,
            await stdoutTask.ConfigureAwait(false),
            await stderrTask.ConfigureAwait(false));
    }

    private static void TryKill(Process process)
    {
        try
        {
            process.Kill(entireProcessTree: true);
        }
        catch (Exception exception) when (exception is InvalidOperationException or NotSupportedException)
        {
            // Already gone.
        }
    }
}
