using Xunit;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Provisioning;
using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Tests;

/// <summary>
/// Covers the reconciliation between transcription and diarization, which is where this
/// approach is weakest and therefore where the behaviour most needs pinning down.
/// </summary>
public class SpeakerAssignerTests
{
    private static TranscriptSegment Segment(double start, double end, string text = "hello") =>
        new(text, start, end);

    [Fact]
    public void Assign_LeavesSegmentsAloneWhenThereAreNoTurns()
    {
        var segments = new[] { Segment(0, 5) };

        var result = new SpeakerAssigner().Assign(segments, []);

        // No diarization is not the same claim as "one speaker throughout".
        Assert.Null(result[0].Speaker);
    }

    [Fact]
    public void Assign_LabelsSegmentFullyInsideOneTurn()
    {
        var segments = new[] { Segment(1, 4) };
        var turns = new[] { new SpeakerTurn("Speaker 1", 0, 10) };

        var result = new SpeakerAssigner().Assign(segments, turns);

        Assert.Equal("Speaker 1", result[0].Speaker);
        Assert.Equal(1.0, result[0].SpeakerOverlapFraction, precision: 6);
        Assert.False(result[0].SpeakerIsUncertain());
    }

    [Fact]
    public void Assign_PicksTheSpeakerWithTheMostOverlap()
    {
        // Two thirds Speaker 2, one third Speaker 1.
        var segments = new[] { Segment(2, 5) };
        var turns = new[]
        {
            new SpeakerTurn("Speaker 1", 0, 3),
            new SpeakerTurn("Speaker 2", 3, 10),
        };

        var result = new SpeakerAssigner().Assign(segments, turns);

        Assert.Equal("Speaker 2", result[0].Speaker);
        Assert.Equal(2.0 / 3.0, result[0].SpeakerOverlapFraction, precision: 6);
    }

    [Fact]
    public void Assign_FlagsASegmentThatStraddlesASpeakerChange()
    {
        // Exactly half each. This is the case that word-level timestamps would fix and we
        // cannot: the segment keeps a speaker, but says the attribution is thin.
        var segments = new[] { Segment(0, 4) };
        var turns = new[]
        {
            new SpeakerTurn("Speaker 1", 0, 2),
            new SpeakerTurn("Speaker 2", 2, 4),
        };

        var result = new SpeakerAssigner().Assign(segments, turns);

        Assert.Equal(0.5, result[0].SpeakerOverlapFraction, precision: 6);
        Assert.True(result[0].SpeakerIsUncertain());
    }

    [Fact]
    public void Assign_IsDeterministicOnAnExactTie()
    {
        var segments = new[] { Segment(0, 4) };
        var turns = new[]
        {
            new SpeakerTurn("Speaker 1", 0, 2),
            new SpeakerTurn("Speaker 2", 2, 4),
        };

        var assigner = new SpeakerAssigner();

        // Running twice must not produce two different transcripts.
        var first = assigner.Assign(segments, turns)[0].Speaker;
        var second = assigner.Assign(segments, turns)[0].Speaker;

        Assert.Equal(first, second);
        Assert.Equal("Speaker 1", first);
    }

    [Fact]
    public void Assign_LeavesSegmentUnlabelledWhenItOverlapsNothing()
    {
        var segments = new[] { Segment(20, 25) };
        var turns = new[] { new SpeakerTurn("Speaker 1", 0, 10) };

        var result = new SpeakerAssigner().Assign(segments, turns);

        Assert.Null(result[0].Speaker);
    }

    [Fact]
    public void Assign_LeavesAZeroLengthSegmentUnlabelled()
    {
        var segments = new[] { Segment(5, 5) };
        var turns = new[] { new SpeakerTurn("Speaker 1", 0, 10) };

        var result = new SpeakerAssigner().Assign(segments, turns);

        // A zero-length segment overlaps nothing by measure, so there is no speaker to claim
        // it. This is what keeps the overlap fraction's division safe.
        Assert.Null(result[0].Speaker);
    }

    [Fact]
    public void Consolidate_JoinsShortGapsInTheSameSpeaker()
    {
        var turns = new[]
        {
            new SpeakerTurn("Speaker 1", 0, 3),
            new SpeakerTurn("Speaker 1", 3.2, 6),
        };

        var result = new SpeakerAssigner(new DiarizationOptions { MinimumGapSeconds = 0.5f })
            .Consolidate(turns);

        Assert.Single(result);
        Assert.Equal(0, result[0].StartSeconds);
        Assert.Equal(6, result[0].EndSeconds);
    }

    [Fact]
    public void Consolidate_KeepsDifferentSpeakersApartAcrossAShortGap()
    {
        var turns = new[]
        {
            new SpeakerTurn("Speaker 1", 0, 3),
            new SpeakerTurn("Speaker 2", 3.1, 6),
        };

        var result = new SpeakerAssigner().Consolidate(turns);

        Assert.Equal(2, result.Count);
    }

    [Fact]
    public void Consolidate_KeepsALongGapInTheSameSpeakerApart()
    {
        var turns = new[]
        {
            new SpeakerTurn("Speaker 1", 0, 3),
            new SpeakerTurn("Speaker 1", 30, 33),
        };

        var result = new SpeakerAssigner().Consolidate(turns);

        Assert.Equal(2, result.Count);
    }

    [Fact]
    public void Consolidate_DropsShortTurnsOnlyAfterMerging()
    {
        // Two fragments, each below the minimum on its own, together above it.
        var turns = new[]
        {
            new SpeakerTurn("Speaker 1", 0, 0.2),
            new SpeakerTurn("Speaker 1", 0.3, 0.5),
        };

        var result = new SpeakerAssigner(new DiarizationOptions
        {
            MinimumTurnSeconds = 0.4f,
            MinimumGapSeconds = 0.5f,
        }).Consolidate(turns);

        Assert.Single(result);
    }

    [Fact]
    public void Consolidate_SortsOutOfOrderTurns()
    {
        var turns = new[]
        {
            new SpeakerTurn("Speaker 2", 10, 15),
            new SpeakerTurn("Speaker 1", 0, 5),
        };

        var result = new SpeakerAssigner().Consolidate(turns);

        Assert.Equal("Speaker 1", result[0].Speaker);
        Assert.Equal("Speaker 2", result[1].Speaker);
    }

    [Fact]
    public void FormatAsDialogue_StartsANewBlockOnEachSpeakerChange()
    {
        var segments = new[]
        {
            Segment(0, 2, "hello there") with { Speaker = "Speaker 1" },
            Segment(2, 4, "and how are you") with { Speaker = "Speaker 1" },
            Segment(4, 6, "very well") with { Speaker = "Speaker 2" },
        };

        var text = SpeakerAssigner.FormatAsDialogue(segments);

        Assert.Contains("Speaker 1: hello there and how are you", text);
        Assert.Contains("Speaker 2: very well", text);
    }

    [Fact]
    public void FormatAsDialogue_CarriesAnUnlabelledSegmentIntoTheCurrentSpeaker()
    {
        var segments = new[]
        {
            Segment(0, 2, "first") with { Speaker = "Speaker 1" },
            Segment(2, 4, "second"),
        };

        var text = SpeakerAssigner.FormatAsDialogue(segments);

        // A gap mid-sentence is a boundary artefact, not a third person entering the room.
        Assert.Equal("Speaker 1: first second", text);
    }

    [Fact]
    public void Transcript_ReportsSpeakersInOrderOfFirstAppearance()
    {
        var transcript = new Transcript(
        [
            Segment(0, 2) with { Speaker = "Speaker 2" },
            Segment(2, 4) with { Speaker = "Speaker 1" },
            Segment(4, 6) with { Speaker = "Speaker 2" },
        ]);

        Assert.True(transcript.HasSpeakers);
        Assert.Equal(["Speaker 2", "Speaker 1"], transcript.Speakers);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Options_RejectAnImpossibleSpeakerCount(int count)
    {
        var options = new DiarizationOptions { SpeakerCount = count };

        Assert.Throws<ArgumentOutOfRangeException>(options.Validate);
    }

    [Fact]
    public void Options_RejectAFractionOutsideZeroToOne()
    {
        var options = new DiarizationOptions { UncertainBelowFraction = 1.5 };

        Assert.Throws<ArgumentOutOfRangeException>(options.Validate);
    }
}

/// <summary>Covers picking the right asset out of a release listing.</summary>
public class GitHubReleaseCatalogTests
{
    private static ReleaseAsset Asset(string name, long size = 1000) =>
        new(name, $"https://example.test/{name}", size);

    [Fact]
    public void PickByPreference_TakesTheFirstPreferenceThatMatches()
    {
        var assets = new[]
        {
            Asset("3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"),
            Asset("nemo_en_titanet_small.onnx"),
        };

        var picked = GitHubReleaseCatalog.PickByPreference(
            assets,
            DiarizationModelInstaller.EmbeddingPreference,
            ".onnx");

        // English first: several published extractors are Mandarin-trained and separate
        // English voices noticeably less well.
        Assert.Equal("nemo_en_titanet_small.onnx", picked?.Name);
    }

    [Fact]
    public void PickByPreference_FallsThroughToALaterPreference()
    {
        var assets = new[] { Asset("3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx") };

        var picked = GitHubReleaseCatalog.PickByPreference(
            assets,
            DiarizationModelInstaller.EmbeddingPreference,
            ".onnx");

        Assert.NotNull(picked);
    }

    [Fact]
    public void PickByPreference_HonoursTheRequiredExtension()
    {
        var assets = new[] { Asset("nemo_en_titanet_small.tar.bz2") };

        var picked = GitHubReleaseCatalog.PickByPreference(
            assets,
            DiarizationModelInstaller.EmbeddingPreference,
            ".onnx");

        Assert.Null(picked);
    }

    [Fact]
    public void PickByPreference_PrefersTheShortestNameAmongVariants()
    {
        var assets = new[]
        {
            Asset("sherpa-onnx-pyannote-segmentation-3-0-int8-quantised.onnx"),
            Asset("sherpa-onnx-pyannote-segmentation-3-0.onnx"),
        };

        var picked = GitHubReleaseCatalog.PickByPreference(
            assets,
            DiarizationModelInstaller.SegmentationPreference,
            ".onnx");

        Assert.Equal("sherpa-onnx-pyannote-segmentation-3-0.onnx", picked?.Name);
    }

    [Fact]
    public void PickByPreference_ReturnsNullWhenNothingMatches()
    {
        var assets = new[] { Asset("unrelated-model.onnx") };

        var picked = GitHubReleaseCatalog.PickByPreference(
            assets,
            DiarizationModelInstaller.SegmentationPreference,
            ".onnx");

        Assert.Null(picked);
    }
}
