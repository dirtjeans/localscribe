using LocalScribe.Core.Hardware;
using LocalScribe.Core.Provisioning;
using LocalScribe.Core.Refinement;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>Creates and cleans up a scratch directory for tests that touch the filesystem.</summary>
internal sealed class TemporaryDirectory : IDisposable
{
    public TemporaryDirectory()
    {
        Path = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            "localscribe-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path);
    }

    public string Path { get; }

    public string Touch(string fileName, string content = "x")
    {
        var full = System.IO.Path.Combine(Path, fileName);
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(full)!);
        File.WriteAllText(full, content);
        return full;
    }

    public void Dispose()
    {
        try
        {
            Directory.Delete(Path, recursive: true);
        }
        catch (IOException)
        {
            // Test cleanup is best effort.
        }
    }
}

public sealed class ModelLayoutTests
{
    [Fact]
    public void ConventionalNamesWorkWithoutAManifest()
    {
        // Someone assembling a directory by hand should not need to write a manifest first.
        using var directory = new TemporaryDirectory();
        directory.Touch("encoder.onnx");
        directory.Touch("decoder.onnx");
        directory.Touch("vocab.json");

        var layout = ModelLayout.Discover(directory.Path);

        Assert.NotNull(layout);
        Assert.Equal("encoder.onnx", layout.Encoder);
    }

    [Fact]
    public void AnIncompleteDirectoryIsNotAModel()
    {
        using var directory = new TemporaryDirectory();
        directory.Touch("encoder.onnx");

        Assert.Null(ModelLayout.Discover(directory.Path));
    }

    [Fact]
    public void AMissingDirectoryIsNotAModel()
    {
        Assert.Null(ModelLayout.Discover(Path.Combine(Path.GetTempPath(), "definitely-not-here-" + Guid.NewGuid())));
    }

    [Fact]
    public void AManifestOverridesTheConventionalNames()
    {
        using var directory = new TemporaryDirectory();
        directory.Touch("HfWhisperEncoder.onnx");
        directory.Touch("HfWhisperDecoder.onnx");
        directory.Touch("vocab.json");

        new ModelLayout
        {
            Encoder = "HfWhisperEncoder.onnx",
            Decoder = "HfWhisperDecoder.onnx",
            Vocab = "vocab.json",
        }.Save(directory.Path);

        var layout = ModelLayout.Discover(directory.Path);

        Assert.NotNull(layout);
        Assert.Equal("HfWhisperEncoder.onnx", layout.Encoder);
    }

    [Fact]
    public void ACorruptManifestFallsBackRatherThanThrowing()
    {
        using var directory = new TemporaryDirectory();
        directory.Touch("encoder.onnx");
        directory.Touch("decoder.onnx");
        directory.Touch("vocab.json");
        directory.Touch(ModelLayout.FileName, "{ not json");

        Assert.NotNull(ModelLayout.Discover(directory.Path));
    }

    [Fact]
    public void AManifestPointingAtAbsentFilesIsIgnored()
    {
        // A manifest left behind by a failed download must not make the directory look usable.
        using var directory = new TemporaryDirectory();
        new ModelLayout { Encoder = "gone.onnx", Decoder = "gone2.onnx", Vocab = "vocab.json" }
            .Save(directory.Path);

        Assert.Null(ModelLayout.Discover(directory.Path));
    }

    [Theory]
    [InlineData("encoder.onnx", "decoder.onnx")]
    [InlineData("model_encoder.onnx", "model_decoder.onnx")]
    [InlineData("HfWhisperEncoder.onnx", "HfWhisperDecoder.onnx")]
    [InlineData("whisper_base_en-whisperencoderinf.onnx", "whisper_base_en-whisperdecoderinf.onnx")]
    public void RolesAreInferredAcrossTheNamingConventionsPublishersUse(string encoder, string decoder)
    {
        var layout = ModelLayout.Infer([encoder, decoder, "vocab.json"]);

        Assert.NotNull(layout);
        Assert.Equal(encoder, layout.Encoder);
        Assert.Equal(decoder, layout.Decoder);
    }

    [Fact]
    public void ThePlainDecoderIsPreferredOverTheWithPastVariant()
    {
        // Optimum exports ship both. Our decode loop is written against the plain one.
        var layout = ModelLayout.Infer(["encoder.onnx", "decoder.onnx", "decoder_with_past.onnx", "vocab.json"]);

        Assert.NotNull(layout);
        Assert.Equal("decoder.onnx", layout.Decoder);
    }

    [Fact]
    public void InferenceFailsRatherThanGuessingWhenARoleIsAbsent()
    {
        Assert.Null(ModelLayout.Infer(["encoder.onnx", "vocab.json"]));
        Assert.Null(ModelLayout.Infer(["encoder.onnx", "decoder.onnx"]));
    }

    [Fact]
    public void ManifestSurvivesASaveAndLoadRoundTrip()
    {
        using var directory = new TemporaryDirectory();
        directory.Touch("a.onnx");
        directory.Touch("b.onnx");
        directory.Touch("vocab.json");

        var original = new ModelLayout
        {
            Encoder = "a.onnx",
            Decoder = "b.onnx",
            Vocab = "vocab.json",
            Source = "qualcomm/Whisper-Base-En",
        };
        original.Save(directory.Path);

        var loaded = ModelLayout.Discover(directory.Path);

        Assert.NotNull(loaded);
        Assert.Equal("qualcomm/Whisper-Base-En", loaded.Source);
    }
}

public sealed class HuggingFaceCatalogTests
{
    private static readonly string[] MultiChipsetRepository =
    [
        "README.md",
        "vocab.json",
        "snapdragon-x-elite/WhisperEncoder.onnx",
        "snapdragon-x-elite/WhisperDecoder.onnx",
        "snapdragon-x-plus/WhisperEncoder.onnx",
        "snapdragon-x-plus/WhisperDecoder.onnx",
    ];

    [Fact]
    public void AssetsAreTakenFromTheMatchingChipsetFolderOnly()
    {
        // A binary built for another Snapdragon will not load, so this must not blend folders.
        var selected = HuggingFaceCatalog.SelectAssets(MultiChipsetRepository, "snapdragon-x-elite");

        Assert.Contains("snapdragon-x-elite/WhisperEncoder.onnx", selected);
        Assert.DoesNotContain("snapdragon-x-plus/WhisperEncoder.onnx", selected);
    }

    [Fact]
    public void TheRootVocabularyIsIncludedAlongsideChipsetSpecificGraphs()
    {
        // QNN exports frequently omit the vocabulary; it lives at the repository root instead.
        var selected = HuggingFaceCatalog.SelectAssets(MultiChipsetRepository, "snapdragon-x-elite");

        Assert.Contains("vocab.json", selected);
    }

    [Fact]
    public void SelectedAssetsFormACompleteLayout()
    {
        var selected = HuggingFaceCatalog.SelectAssets(MultiChipsetRepository, "snapdragon-x-elite");

        Assert.NotNull(ModelLayout.Infer(selected));
    }

    [Theory]
    [InlineData("snapdragon_x_elite")]
    [InlineData("SnapdragonXElite")]
    [InlineData("Snapdragon-X-Elite")]
    public void ChipsetFoldersMatchRegardlessOfSeparatorsOrCasing(string folderName)
    {
        var paths = new[] { $"{folderName}/encoder.onnx", $"{folderName}/decoder.onnx", "vocab.json" };

        var selected = HuggingFaceCatalog.SelectAssets(paths, "snapdragon-x-elite");

        Assert.Contains($"{folderName}/encoder.onnx", selected);
    }

    [Fact]
    public void AFlatRepositoryIsUsedWholeWhenThereAreNoChipsetFolders()
    {
        var paths = new[] { "encoder.onnx", "decoder.onnx", "vocab.json", "README.md" };

        var selected = HuggingFaceCatalog.SelectAssets(paths, "snapdragon-x-elite");

        Assert.Equal(3, selected.Count);
        Assert.DoesNotContain("README.md", selected);
    }

    [Fact]
    public void WeightSidecarsAreIncluded()
    {
        // Models above 2 GB keep their weights beside the graph. Missing these gives a model
        // that loads and then fails at the first inference.
        var paths = new[] { "encoder.onnx", "encoder.onnx_data", "decoder.onnx", "vocab.json" };

        Assert.Contains("encoder.onnx_data", HuggingFaceCatalog.SelectAssets(paths, "cpu"));
    }

    [Fact]
    public void DocumentationAndImagesAreNotDownloaded()
    {
        var paths = new[] { "encoder.onnx", "decoder.onnx", "vocab.json", "README.md", "demo.png", ".gitattributes" };

        var selected = HuggingFaceCatalog.SelectAssets(paths, "cpu");

        Assert.All(selected, p => Assert.DoesNotContain(".png", p, StringComparison.Ordinal));
        Assert.Equal(3, selected.Count);
    }

    [Fact]
    public void AnEmptyRepositorySelectsNothing()
    {
        Assert.Empty(HuggingFaceCatalog.SelectAssets([], "snapdragon-x-elite"));
    }

    [Fact]
    public void EveryModelSizeHasAtLeastOneRepositoryToTry()
    {
        foreach (var model in new[] { "tiny.en", "base.en", "small.en", "medium.en" })
        {
            Assert.NotEmpty(HuggingFaceCatalog.RepositoriesFor(model));
        }
    }

    [Fact]
    public void DownloadUrlsPointAtTheResolveEndpoint()
    {
        var url = HuggingFaceCatalog.DownloadUrl("qualcomm/Whisper-Base-En", "snapdragon-x-elite/encoder.onnx");

        Assert.Equal(
            "https://huggingface.co/qualcomm/Whisper-Base-En/resolve/main/snapdragon-x-elite/encoder.onnx",
            url);
    }
}

public sealed class FoundryEndpointTests
{
    [Theory]
    [InlineData("Model management service is running on http://localhost:52144/openai/status", 52144)]
    [InlineData("Service is running at http://127.0.0.1:5273", 5273)]
    [InlineData("endpoint: http://localhost:61234/v1", 61234)]
    public void ThePortIsReadFromTheStatusTextRatherThanAssumed(string output, int expectedPort)
    {
        // Foundry Local binds a dynamic port, so a hard-coded guess reports a running service
        // as absent.
        var endpoint = FoundryLocalManager.ParseEndpoint(output);

        Assert.NotNull(endpoint);
        Assert.Equal(expectedPort, endpoint.Port);
    }

    [Theory]
    [InlineData("")]
    [InlineData("Service is not running.")]
    public void NoEndpointIsFoundWhenTheServiceIsDown(string output)
    {
        Assert.Null(FoundryLocalManager.ParseEndpoint(output));
    }

    [Fact]
    public void ADiscoveredApiPathIsStrippedBackToTheOrigin()
    {
        // Left in place, "/v1" would produce requests to /v1/v1/chat/completions, because a
        // relative request URI resolves against the base address rather than appending to it.
        var normalised = FoundryLocalClient.NormaliseEndpoint(new Uri("http://localhost:61234/v1"));

        Assert.Equal("http://localhost:61234/", normalised!.ToString());
    }
}

public sealed class ProvisioningPlanTests
{
    private static DeviceCapabilities Snapdragon(bool driver = false, bool foundryRunning = false) => new()
    {
        SocName = "Snapdragon X Elite X1E-78-100",
        Family = SocFamily.SnapdragonXElite,
        PerformanceCoreCount = 12,
        TotalCoreCount = 12,
        TotalMemoryBytes = 32L * 1024 * 1024 * 1024,
        HexagonDriverPresent = driver,
        FoundryLocalPresent = foundryRunning,
    };

    private static ProvisioningPlan Build(
        DeviceCapabilities capabilities,
        string modelDirectory,
        bool foundryInstalled = false,
        bool arm64 = true) =>
        Provisioner.BuildPlan(
            capabilities,
            AcceleratorPlanner.Plan(capabilities),
            modelDirectory,
            foundryInstalled,
            arm64);

    [Fact]
    public void TheHexagonDriverIsNeverInstalledAutomatically()
    {
        // It is a signed kernel driver behind an account wall. Claiming otherwise would produce
        // a tool that appears to succeed and leaves the app silently CPU-bound.
        using var directory = new TemporaryDirectory();

        var plan = Build(Snapdragon(), directory.Path);
        var driver = plan.Components.Single(c => c.Id == "hexagon-driver");

        Assert.False(driver.CanInstallAutomatically);
        Assert.Contains("softwarecenter.qualcomm.com", driver.ManualInstructions!, StringComparison.Ordinal);
    }

    [Fact]
    public void ModelWeightsAreInstallableAutomatically()
    {
        using var directory = new TemporaryDirectory();

        var plan = Build(Snapdragon(), directory.Path);

        Assert.Contains(plan.Installable, c => c.Id == "whisper-model");
    }

    [Fact]
    public void AnAlreadyInstalledModelIsNotReinstalled()
    {
        using var directory = new TemporaryDirectory();
        directory.Touch("encoder.onnx");
        directory.Touch("decoder.onnx");
        directory.Touch("vocab.json");

        var plan = Build(Snapdragon(), directory.Path);

        Assert.DoesNotContain(plan.Installable, c => c.Id == "whisper-model");
        Assert.True(plan.CanTranscribe);
    }

    [Fact]
    public void MissingWeightsBlockTranscriptionButAMissingDriverDoesNot()
    {
        // Without weights there is nothing to run. Without the driver it still runs, just slower.
        using var empty = new TemporaryDirectory();
        Assert.False(Build(Snapdragon(driver: true), empty.Path).CanTranscribe);

        using var stocked = new TemporaryDirectory();
        stocked.Touch("encoder.onnx");
        stocked.Touch("decoder.onnx");
        stocked.Touch("vocab.json");
        Assert.True(Build(Snapdragon(driver: false), stocked.Path).CanTranscribe);
    }

    [Fact]
    public void TheCleanupModelCannotBeInstalledBeforeItsEngine()
    {
        using var directory = new TemporaryDirectory();

        var plan = Build(Snapdragon(), directory.Path, foundryInstalled: false);
        var model = plan.Components.Single(c => c.Id == "foundry-model");

        Assert.False(model.CanInstallAutomatically);
        Assert.Contains("Install Foundry Local first", model.ManualInstructions!, StringComparison.Ordinal);
    }

    [Fact]
    public void OnceTheEngineIsPresentItsModelBecomesInstallable()
    {
        using var directory = new TemporaryDirectory();

        var plan = Build(Snapdragon(), directory.Path, foundryInstalled: true);

        Assert.Contains(plan.Installable, c => c.Id == "foundry-model");
    }

    [Fact]
    public void EmulationIsReportedFirstBecauseItInvalidatesEverythingElse()
    {
        using var directory = new TemporaryDirectory();

        var plan = Build(Snapdragon(), directory.Path, arm64: false);

        Assert.Equal("arm64", plan.Components[0].Id);
        Assert.False(plan.Components[0].Installed);
        Assert.Contains("win-arm64", plan.Components[0].ManualInstructions!, StringComparison.Ordinal);
    }

    [Fact]
    public void NonQualcommMachinesAreNotToldToInstallADriver()
    {
        using var directory = new TemporaryDirectory();
        var capabilities = DeviceCapabilities.Unknown with { Family = SocFamily.NonQualcomm };

        var plan = Build(capabilities, directory.Path);

        Assert.DoesNotContain(plan.Components, c => c.Id == "hexagon-driver");
    }

    [Fact]
    public void NonQualcommMachinesAreNotAccusedOfRunningEmulated()
    {
        // An x64 desktop is not "running under emulation"; it is simply not a Snapdragon.
        using var directory = new TemporaryDirectory();
        var capabilities = DeviceCapabilities.Unknown with { Family = SocFamily.NonQualcomm };

        var plan = Build(capabilities, directory.Path, arm64: false);

        Assert.True(plan.Components.Single(c => c.Id == "arm64").Installed);
    }

    [Fact]
    public void EveryMissingComponentOffersEitherAnInstallerOrInstructions()
    {
        using var directory = new TemporaryDirectory();

        var plan = Build(Snapdragon(), directory.Path);

        Assert.All(
            plan.Components.Where(c => !c.Installed),
            c => Assert.True(
                c.CanInstallAutomatically || !string.IsNullOrWhiteSpace(c.ManualInstructions),
                $"{c.Id} is missing but offers neither an installer nor instructions."));
    }
}
