using System.Runtime.InteropServices;
using LocalScribe.Core.Diarization;
using LocalScribe.Core.Models;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// Meeting the user's speaker count with an end-to-end diarizer that cannot be told one: the
/// extras are folded in, smallest first, into whoever they sound like, and nobody is invented.
/// </summary>
public class SpeakerMergingTests
{
    private static readonly float[] Alice = [1f, 0f, 0f];
    private static readonly float[] Bob = [0f, 1f, 0f];
    private static readonly float[] LikeAlice = [0.9f, 0.1f, 0f];
    private static readonly float[] LikeBob = [0.1f, 0.9f, 0f];

    private static SpeakerTurn Turn(int speaker, double start, double end) => new(speaker, start, end);

    [Fact]
    public void AnExtraSpeakerJoinsWhoeverTheySoundLike()
    {
        // Speaker 2 is short and sounds like speaker 1, though they talk right after speaker 0.
        IReadOnlyList<SpeakerTurn> turns = [Turn(0, 0, 30), Turn(2, 30, 33), Turn(1, 40, 70)];
        var prints = new Dictionary<int, float[]> { [0] = Alice, [1] = Bob, [2] = LikeBob };

        var fitted = SpeakerMerging.ToCount(turns, prints, 2);

        Assert.Equal(2, fitted.Select(t => t.Speaker).Distinct().Count());
        Assert.Equal(fitted[2].Speaker, fitted[1].Speaker);
        Assert.NotEqual(fitted[0].Speaker, fitted[1].Speaker);
    }

    [Fact]
    public void TheSmallestSpeakerIsFoldedFirst()
    {
        // Three people; one must go. The one heard for two seconds goes, though the large one
        // sounds no less like someone else.
        IReadOnlyList<SpeakerTurn> turns = [Turn(0, 0, 60), Turn(1, 60, 62), Turn(2, 70, 130)];
        var prints = new Dictionary<int, float[]> { [0] = Alice, [1] = LikeAlice, [2] = LikeBob };

        var fitted = SpeakerMerging.ToCount(turns, prints, 2);

        Assert.Equal(fitted[0].Speaker, fitted[1].Speaker);
        Assert.NotEqual(fitted[0].Speaker, fitted[2].Speaker);
    }

    [Fact]
    public void NobodyIsInventedWhenFewerWereHeard()
    {
        IReadOnlyList<SpeakerTurn> turns = [Turn(0, 0, 10), Turn(1, 10, 20)];

        var fitted = SpeakerMerging.ToCount(turns, new Dictionary<int, float[]>(), 5);

        Assert.Equal(turns, fitted);
    }

    [Fact]
    public void ASpeakerTooShortToMeasureJoinsTheirNeighbourInTime()
    {
        IReadOnlyList<SpeakerTurn> turns = [Turn(0, 0, 30), Turn(1, 40, 70), Turn(2, 70.5, 71)];
        var prints = new Dictionary<int, float[]> { [0] = Alice, [1] = Bob };

        var fitted = SpeakerMerging.ToCount(turns, prints, 2);

        Assert.Equal(fitted[1].Speaker, fitted[2].Speaker);
    }

    [Fact]
    public void LabelsStayDenseAndInOrderOfAppearance()
    {
        IReadOnlyList<SpeakerTurn> turns = [Turn(0, 0, 1), Turn(1, 1, 40), Turn(2, 40, 80)];
        var prints = new Dictionary<int, float[]> { [0] = LikeBob, [1] = Alice, [2] = Bob };

        var fitted = SpeakerMerging.ToCount(turns, prints, 2);

        // Speaker 0 folded into speaker 2, who now opens the recording and is renumbered 0.
        Assert.Equal([0, 1, 0], fitted.Select(t => t.Speaker));
    }

    [Fact]
    public void OnlyTwoDifferentPeopleTalkingAtOnceIsCrosstalk()
    {
        IReadOnlyList<SpeakerTurn> turns = [Turn(0, 0, 10), Turn(0, 8, 12), Turn(1, 11, 15), Turn(1, 14, 16)];

        var (start, end) = Assert.Single(SpeakerMerging.Overlaps(turns));

        Assert.Equal(11, start, 6);
        Assert.Equal(12, end, 6);
    }
}

public class SpeakerEngineTests
{
    [Fact]
    public void Arm64DiarizesWithNemotron() =>
        Assert.Equal(SpeakerEngine.Sortformer, SpeakerEngines.For(Architecture.Arm64));

    [Theory]
    [InlineData(Architecture.X64)]
    [InlineData(Architecture.X86)]
    public void OtherMachinesKeepThePipelineUntilItIsMeasuredThere(Architecture architecture) =>
        Assert.Equal(SpeakerEngine.Pyannote, SpeakerEngines.For(architecture));

    [Fact]
    public void NemotronIsNeverOfferedMoreSpeakersThanItHasChannels() =>
        Assert.Equal(ArrivalOrderSpeakerCache.Speakers, SpeakerEngines.MostSpeakers(SpeakerEngine.Sortformer));
}

public class SortformerModelSourceTests
{
    [Fact]
    public void EveryFileIsPinnedToItsExactContents()
    {
        Assert.All(SortformerModelSource.Files, file =>
        {
            Assert.Matches("^[0-9a-f]{64}$", file.Sha256);

            // A commit in the address, never a branch: a branch moves.
            Assert.Matches("/resolve/[0-9a-f]{40}/", file.Source.AbsoluteUri);
            Assert.False(file.Optional);
        });
    }

    [Fact]
    public void TheLicenceIsFetchedButNotRequiredToRun()
    {
        var directory = Path.Combine(Path.GetTempPath(), $"localscribe-sortformer-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);

        try
        {
            Assert.Contains(SortformerModelSource.Files, file => file.FileName == "LICENSE");
            Assert.False(SortformerModelSource.IsInstalled(directory));

            foreach (var file in SortformerModelSource.Files.Where(f => f.FileName != "LICENSE"))
            {
                File.WriteAllBytes(Path.Combine(directory, file.FileName), [1]);
            }

            Assert.True(SortformerModelSource.IsInstalled(directory));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
