using System.Runtime.InteropServices;
using LocalScribe.Core.Hardware;

namespace LocalScribe.Core.Provisioning;

/// <summary>
/// Works out what is missing, and installs what can be installed.
/// <para>
/// The split between automatic and manual is the point of this class. Model weights and a local
/// inference service are ordinary downloads. A signed kernel driver behind an account wall is
/// not, and pretending otherwise would produce a tool that appears to work and then leaves the
/// user with a silently CPU-bound app. Anything needing a human says so, precisely.
/// </para>
/// </summary>
public sealed class Provisioner
{
    /// <summary>
    /// Where to get the Hexagon driver. Named explicitly because this is the step people miss,
    /// and the symptom — everything works, just slowly — points nowhere near the cause.
    /// </summary>
    public const string HexagonDriverUrl = "https://softwarecenter.qualcomm.com/";

    private readonly WhisperModelInstaller _models;
    private readonly FoundryLocalManager _foundry;
    private readonly DiarizationModelInstaller _diarizationModels;

    public Provisioner(
        WhisperModelInstaller? models = null,
        FoundryLocalManager? foundry = null,
        DiarizationModelInstaller? diarizationModels = null)
    {
        _models = models ?? new WhisperModelInstaller();
        _foundry = foundry ?? new FoundryLocalManager();
        _diarizationModels = diarizationModels ?? new DiarizationModelInstaller();
    }

    /// <summary>
    /// Assembles the prerequisite picture. Pure apart from the file-existence checks its inputs
    /// already performed, so the ordering and classification can be tested directly.
    /// </summary>
    /// <param name="capabilities">What the machine was observed to have.</param>
    /// <param name="plan">The execution plan, which determines which model size is needed.</param>
    /// <param name="modelDirectory">Directory for the chipset and size in <paramref name="plan"/>.</param>
    /// <param name="foundryInstalled">Whether the foundry CLI was found.</param>
    /// <param name="processIsArm64">Whether this process is running natively on arm64.</param>
    /// <param name="diarizationDirectory">
    /// Where the speaker diarization models live. Pass <c>null</c> to leave diarization out of
    /// the plan entirely.
    /// </param>
    public static ProvisioningPlan BuildPlan(
        DeviceCapabilities capabilities,
        ExecutionPlan plan,
        string modelDirectory,
        bool foundryInstalled,
        bool processIsArm64,
        string? diarizationDirectory = null)
    {
        ArgumentNullException.ThrowIfNull(capabilities);
        ArgumentNullException.ThrowIfNull(plan);

        var isQualcomm = capabilities.Family
            is SocFamily.SnapdragonXElite or SocFamily.SnapdragonXPlus or SocFamily.SnapdragonX2;

        var components = new List<ComponentStatus>
        {
            // Listed first because it invalidates everything below it. Under emulation the QNN
            // provider cannot load at all, and the resulting symptoms look like a bad driver.
            new(
                Id: "arm64",
                Title: "Native arm64 process",
                Kind: ComponentKind.BuildConfiguration,
                Installed: processIsArm64 || !isQualcomm,
                CanInstallAutomatically: false,
                Detail: !isQualcomm
                    // Emulation is beside the point on hardware that has no NPU to reach.
                    ? "Not applicable: this processor has no Hexagon NPU."
                    : processIsArm64
                        ? "Running natively."
                        : "Running under x64 emulation, where the NPU is unreachable.",
                ManualInstructions: "Rebuild and run with -r win-arm64. Nothing else on this list "
                    + "will help until this is fixed.",
                Required: false),

            new(
                Id: "whisper-model",
                Title: $"Whisper {plan.WhisperModel} weights",
                Kind: ComponentKind.ModelAssets,
                Installed: ModelLayout.Discover(modelDirectory) is not null,
                CanInstallAutomatically: true,
                Detail: ModelLayout.Discover(modelDirectory) is not null
                    ? $"Found in {modelDirectory}."
                    : $"Not present in {modelDirectory}.",
                Required: true),
        };

        if (isQualcomm)
        {
            components.Add(new ComponentStatus(
                Id: "hexagon-driver",
                Title: "Hexagon NPU runtime driver",
                Kind: ComponentKind.Driver,
                Installed: capabilities.HexagonDriverPresent,
                CanInstallAutomatically: false,
                Detail: capabilities.HexagonDriverPresent
                    ? "Installed."
                    : "Missing. The NPU cannot be used without it.",
                ManualInstructions:
                    $"Download the Hexagon NPU Runtime Driver from {HexagonDriverUrl} (a free "
                    + "developer account is required) and install it. This is a separate package "
                    + "from the driver Windows ships with, and cannot be installed unattended.",
                Required: false));
        }

        components.Add(new ComponentStatus(
            Id: "foundry",
            Title: "Foundry Local",
            Kind: ComponentKind.InferenceEngine,
            Installed: foundryInstalled,
            CanInstallAutomatically: true,
            Detail: foundryInstalled ? "Installed." : "Not installed.",
            Required: false));

        components.Add(new ComponentStatus(
            Id: "foundry-model",
            Title: $"Cleanup model ({FoundryLocalManager.DefaultModelAlias})",
            Kind: ComponentKind.ModelAssets,
            Installed: capabilities.LocalLanguageModelPresent,
            CanInstallAutomatically: foundryInstalled,
            Detail: capabilities.LocalLanguageModelPresent
                ? "Service is answering."
                : "Service is not answering, so punctuation repair and summaries are unavailable.",
            ManualInstructions: foundryInstalled
                ? null
                : "Install Foundry Local first.",
            Required: false));

        if (diarizationDirectory is not null)
        {
            var installed = DiarizationModelInstaller.IsInstalled(diarizationDirectory);

            components.Add(new ComponentStatus(
                Id: "diarization-models",
                Title: "Speaker diarization models (pyannote via sherpa-onnx)",
                Kind: ComponentKind.ModelAssets,
                Installed: installed,
                CanInstallAutomatically: true,
                Detail: installed
                    ? $"Found in {diarizationDirectory}. Runs on CPU; the NPU is not involved."
                    : "Not present, so the transcript will have no speaker labels.",
                Required: false));
        }

        return new ProvisioningPlan(components);
    }

    /// <summary>True when this process is running natively on arm64 rather than emulated.</summary>
    public static bool ProcessIsArm64 => RuntimeInformation.ProcessArchitecture == Architecture.Arm64;

    /// <summary>
    /// Installs every component that can be installed unattended.
    /// <para>
    /// Failures do not stop the run. The components are independent, and a machine that cannot
    /// reach the model repository can still install the cleanup service, so it is better to make
    /// what progress is possible and report the rest than to abort at the first problem.
    /// </para>
    /// </summary>
    public async Task<IReadOnlyList<InstallResult>> InstallAsync(
        ProvisioningPlan provisioningPlan,
        string modelDirectory,
        string whisperModel,
        string chipsetSlug,
        IProgress<InstallProgress>? progress = null,
        CancellationToken cancellationToken = default,
        string? diarizationDirectory = null)
    {
        ArgumentNullException.ThrowIfNull(provisioningPlan);

        var results = new List<InstallResult>();

        foreach (var component in provisioningPlan.Installable)
        {
            cancellationToken.ThrowIfCancellationRequested();

            try
            {
                results.Add(component.Id switch
                {
                    "whisper-model" => await _models.EnsureInstalledAsync(
                        modelDirectory, whisperModel, chipsetSlug, progress, cancellationToken)
                        .ConfigureAwait(false),

                    "foundry" => await InstallFoundryAsync(progress, cancellationToken).ConfigureAwait(false),

                    "foundry-model" => await InstallFoundryModelAsync(progress, cancellationToken)
                        .ConfigureAwait(false),

                    "diarization-models" when diarizationDirectory is not null =>
                        await _diarizationModels
                            .EnsureInstalledAsync(diarizationDirectory, progress, cancellationToken)
                            .ConfigureAwait(false),

                    _ => new InstallResult(component.Id, false, "No installer is defined for this component."),
                });
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception exception)
            {
                results.Add(new InstallResult(component.Id, false, exception.Message));
            }
        }

        return results;
    }

    private async Task<InstallResult> InstallFoundryAsync(
        IProgress<InstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        var install = await _foundry.InstallAsync(progress, cancellationToken).ConfigureAwait(false);
        if (!install.Succeeded)
        {
            return install;
        }

        return await _foundry.StartServiceAsync(progress, cancellationToken).ConfigureAwait(false);
    }

    private async Task<InstallResult> InstallFoundryModelAsync(
        IProgress<InstallProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Starting the service is idempotent, and the model download needs it running.
        await _foundry.StartServiceAsync(progress, cancellationToken).ConfigureAwait(false);

        return await _foundry
            .DownloadModelAsync(FoundryLocalManager.DefaultModelAlias, progress, cancellationToken)
            .ConfigureAwait(false);
    }
}
