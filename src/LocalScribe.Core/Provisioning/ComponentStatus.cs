namespace LocalScribe.Core.Provisioning;

/// <summary>What sort of thing a prerequisite is. Determines who can install it.</summary>
public enum ComponentKind
{
    /// <summary>Whisper weights and vocabulary. Downloadable.</summary>
    ModelAssets,

    /// <summary>A local inference service such as Foundry Local. Installable through a package manager.</summary>
    InferenceEngine,

    /// <summary>A hardware driver. Needs a human.</summary>
    Driver,

    /// <summary>Something decided at build time, such as which architecture the app was compiled for.</summary>
    BuildConfiguration,
}

/// <summary>One prerequisite, whether it is satisfied, and what to do if not.</summary>
/// <param name="Id">Stable identifier, used to select components on the command line.</param>
/// <param name="Title">Short human-readable name.</param>
/// <param name="Kind">What sort of prerequisite this is.</param>
/// <param name="Installed">True when the component is present and usable.</param>
/// <param name="CanInstallAutomatically">
/// True when this tool can install it unattended. False for anything needing a licence
/// acceptance, an account, or a reboot.
/// </param>
/// <param name="Detail">What was found, or what is missing.</param>
/// <param name="ManualInstructions">
/// What a person must do. Populated whenever <paramref name="CanInstallAutomatically"/> is false
/// and the component is missing.
/// </param>
/// <param name="Required">
/// False for components the app works without, such as the cleanup model. Optional components
/// are reported but never block.
/// </param>
public sealed record ComponentStatus(
    string Id,
    string Title,
    ComponentKind Kind,
    bool Installed,
    bool CanInstallAutomatically,
    string Detail,
    string? ManualInstructions = null,
    bool Required = true)
{
    /// <summary>True when this component is missing and something can be done about it here.</summary>
    public bool NeedsAutomaticInstall => !Installed && CanInstallAutomatically;

    /// <summary>True when this component is missing and only a person can fix it.</summary>
    public bool NeedsManualAction => !Installed && !CanInstallAutomatically;

    /// <summary>True when the app cannot transcribe at all until this is resolved.</summary>
    public bool IsBlocking => !Installed && Required;
}

/// <summary>The full prerequisite picture for a machine.</summary>
/// <param name="Components">Every prerequisite, in the order they should be addressed.</param>
public sealed record ProvisioningPlan(IReadOnlyList<ComponentStatus> Components)
{
    /// <summary>Components this tool can install without help.</summary>
    public IReadOnlyList<ComponentStatus> Installable =>
        [.. Components.Where(c => c.NeedsAutomaticInstall)];

    /// <summary>Components needing a person.</summary>
    public IReadOnlyList<ComponentStatus> ManualActions =>
        [.. Components.Where(c => c.NeedsManualAction)];

    /// <summary>True when transcription is possible right now.</summary>
    public bool CanTranscribe => !Components.Any(c => c.IsBlocking);

    /// <summary>True when running the installer would change anything.</summary>
    public bool HasWorkToDo => Installable.Count > 0;
}

/// <summary>Progress during an install, so a long download does not look like a hang.</summary>
/// <param name="ComponentId">Which component is being worked on.</param>
/// <param name="Message">What is happening.</param>
/// <param name="Fraction">Completion between 0 and 1, or <c>null</c> when not measurable.</param>
public sealed record InstallProgress(string ComponentId, string Message, double? Fraction = null);

/// <summary>The outcome of installing one component.</summary>
/// <param name="ComponentId">Which component this refers to.</param>
/// <param name="Succeeded">Whether it is now installed.</param>
/// <param name="Message">What happened, including the failure reason when unsuccessful.</param>
public sealed record InstallResult(string ComponentId, bool Succeeded, string Message);
