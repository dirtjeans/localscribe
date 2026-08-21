using System.ComponentModel;
using System.Runtime.CompilerServices;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Provisioning;
using LocalScribe.Core.Refinement;
using LocalScribe.Onnx;

namespace LocalScribe.App;

/// <summary>
/// Works out what this machine is missing, and downloads what it can.
/// <para>
/// Holds no UI types. The window renders what this exposes; the sequencing — probe, plan,
/// install, probe again — lives here so it stays in one readable place rather than spread
/// through event handlers.
/// </para>
/// </summary>
public sealed class SetupViewModel : INotifyPropertyChanged
{
    private readonly string _modelRoot;
    private readonly FoundryLocalManager _foundry = new();
    private readonly Provisioner _provisioner = new();

    private ProvisioningPlan? _provisioning;
    private bool _installing;
    private string _status = string.Empty;
    private double _progress;

    public SetupViewModel(string modelRoot)
    {
        _modelRoot = modelRoot;
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>What the probe found. Conservative defaults until the first refresh completes.</summary>
    public DeviceCapabilities Capabilities { get; private set; } = DeviceCapabilities.Unknown;

    /// <summary>Where each stage will run, given what the probe found.</summary>
    public ExecutionPlan? Plan { get; private set; }

    /// <summary>
    /// Where Foundry Local is actually listening, discovered by asking its CLI.
    /// <para>
    /// Kept here and handed to every client we construct because the service binds a dynamic
    /// loopback port. Assuming a fixed one makes a perfectly healthy service look absent, and
    /// the only visible symptom is that transcripts quietly stop being cleaned up.
    /// </para>
    /// </summary>
    public Uri? FoundryEndpoint { get; private set; }

    /// <summary>Directory holding the weights for this chipset and model size.</summary>
    public string ModelDirectory { get; private set; } = string.Empty;

    /// <summary>Every prerequisite and its state, in the order they should be addressed.</summary>
    public IReadOnlyList<ComponentStatus> Components => _provisioning?.Components ?? [];

    /// <summary>Prerequisites only a person can resolve, such as the Hexagon driver.</summary>
    public IReadOnlyList<ComponentStatus> ManualActions => _provisioning?.ManualActions ?? [];

    /// <summary>True when transcription is possible right now.</summary>
    public bool CanTranscribe => _provisioning?.CanTranscribe ?? false;

    /// <summary>True when there is something the app can download without help.</summary>
    public bool HasWorkToDo => _provisioning?.HasWorkToDo ?? false;

    /// <summary>True while a download is running, so the UI can disable its own controls.</summary>
    public bool IsInstalling
    {
        get => _installing;
        private set => Set(ref _installing, value);
    }

    /// <summary>What setup is doing right now.</summary>
    public string Status
    {
        get => _status;
        private set => Set(ref _status, value);
    }

    /// <summary>Download progress between 0 and 1, or 0 when not measurable.</summary>
    public double Progress
    {
        get => _progress;
        private set => Set(ref _progress, value);
    }

    private string ChipsetSlug { get; set; } = string.Empty;

    /// <summary>
    /// Probes the machine and rebuilds the prerequisite picture. Safe to call repeatedly; it is
    /// how the app learns that a download, or a driver install done outside it, took effect.
    /// </summary>
    public async Task RefreshAsync(CancellationToken cancellationToken = default)
    {
        Status = "Checking this machine…";

        try
        {
            await ProbeAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception exception)
        {
            // A probe that throws must not take the window with it. Reporting nothing is a
            // worse outcome than reporting that we could not look.
            Status = $"Could not check this machine: {exception.Message}";
        }
    }

    private async Task ProbeAsync(CancellationToken cancellationToken)
    {
        var foundryInstalled = await _foundry.IsInstalledAsync(cancellationToken).ConfigureAwait(false);

        // Ask the CLI where the service is before probing for it, so a running service on a
        // dynamic port is found rather than reported missing.
        FoundryEndpoint = await _foundry.DiscoverEndpointAsync(cancellationToken).ConfigureAwait(false);

        var capabilities = await Task.Run(() => DeviceProbe.Probe(_modelRoot), cancellationToken)
            .ConfigureAwait(false);

        using (var client = new FoundryLocalClient(endpoint: FoundryEndpoint))
        {
            capabilities = capabilities with
            {
                FoundryLocalPresent = await client.IsAvailableAsync(cancellationToken).ConfigureAwait(false),
            };
        }

        Capabilities = capabilities;
        Plan = AcceleratorPlanner.Plan(capabilities);

        ChipsetSlug = DeviceProbe.AssetFolderFor(capabilities.Family);
        ModelDirectory = Path.Combine(_modelRoot, ChipsetSlug, Plan.WhisperModel);

        _provisioning = Provisioner.BuildPlan(
            capabilities,
            Plan,
            ModelDirectory,
            foundryInstalled,
            Provisioner.ProcessIsArm64);

        RaisePlanChanged();
        Status = CanTranscribe ? "Ready." : "Setup needed.";
    }

    /// <summary>
    /// Downloads everything that can be installed unattended, then re-probes so the reported
    /// state reflects what is now on disk rather than what was there when the window opened.
    /// </summary>
    public async Task<IReadOnlyList<InstallResult>> InstallAsync(CancellationToken cancellationToken = default)
    {
        if (_provisioning is null || Plan is null || IsInstalling)
        {
            return [];
        }

        IsInstalling = true;
        Progress = 0;

        try
        {
            var progress = new Progress<InstallProgress>(update =>
            {
                Status = update.Message;
                Progress = update.Fraction ?? Progress;
            });

            var results = await _provisioner.InstallAsync(
                _provisioning,
                ModelDirectory,
                Plan.WhisperModel,
                ChipsetSlug,
                progress,
                cancellationToken).ConfigureAwait(false);

            await RefreshAsync(cancellationToken).ConfigureAwait(false);

            // The per-component messages carry the diagnosis — a blocked network reads very
            // differently from a model that was never published — so a failure surfaces the
            // text rather than a count.
            var failed = results.Where(r => !r.Succeeded).ToList();
            Status = failed.Count == 0
                ? "Setup complete."
                : string.Join(" ", failed.Select(r => r.Message));

            return results;
        }
        catch (OperationCanceledException)
        {
            Status = "Setup cancelled. Partial downloads were discarded.";
            return [];
        }
        catch (Exception exception)
        {
            Status = $"Setup failed: {exception.Message}";
            return [];
        }
        finally
        {
            IsInstalling = false;
            Progress = 0;
        }
    }

    /// <summary>
    /// The readable summary of what will run where, or why nothing will. Shown once setup has
    /// something to say rather than left for the user to infer from the component list.
    /// </summary>
    public string Verdict
    {
        get
        {
            if (Plan is null)
            {
                return "Checking…";
            }

            if (!CanTranscribe)
            {
                return "LocalScribe cannot transcribe yet. The items above need attention first.";
            }

            return Plan.IsCpuOnly
                ? "Everything will run on the processor. That works, but it is slower and the "
                    + "machine will feel busier. The items above explain why."
                : Plan.Summary;
        }
    }

    private void RaisePlanChanged()
    {
        Raise(nameof(Components));
        Raise(nameof(ManualActions));
        Raise(nameof(CanTranscribe));
        Raise(nameof(HasWorkToDo));
        Raise(nameof(Verdict));
    }

    private void Raise([CallerMemberName] string? name = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));

    private void Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return;
        }

        field = value;
        Raise(name);
    }
}
