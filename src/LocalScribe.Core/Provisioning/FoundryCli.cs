namespace LocalScribe.Core.Provisioning;

/// <summary>
/// Where the Foundry Local CLI actually is.
/// <para>
/// Invoking a bare "foundry" works from a terminal and fails from Finder: a GUI-launched app
/// gets the minimal system PATH, which does not include /usr/local/bin or /opt/homebrew/bin —
/// exactly where the CLI lives on a Mac. Every Foundry feature in the app quietly worked from
/// development shells and failed for the installed bundle, which read as "I have to set this
/// up every time". Resolving to an absolute path when one exists makes launch context stop
/// mattering; the bare name remains the fallback, and on Windows it is simply correct.
/// </para>
/// </summary>
public static class FoundryCli
{
    private static readonly string[] KnownHomes =
    [
        "/usr/local/bin/foundry",
        "/opt/homebrew/bin/foundry",
    ];

    public static string Path { get; } = Find();

    private static string Find()
    {
        if (OperatingSystem.IsWindows())
        {
            return "foundry";
        }

        return KnownHomes.FirstOrDefault(File.Exists) ?? "foundry";
    }
}
