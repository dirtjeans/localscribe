using LocalScribe.Core.Diarization;
using Xunit;

namespace LocalScribe.Core.Tests;

public class SpeakerOverlapTests
{
    private static IReadOnlyList<(double Start, double End)> Talks(params double[] pairs)
    {
        var spans = new List<(double, double)>();
        for (var i = 0; i + 1 < pairs.Length; i += 2)
        {
            spans.Add((pairs[i], pairs[i + 1]));
        }

        return spans;
    }

    private static IReadOnlyList<IReadOnlyList<(double Start, double End)>> Window(
        params IReadOnlyList<(double Start, double End)>[] speakers) => speakers;

    /// <summary>Several windows agreeing, as the real scan produces.</summary>
    private static IReadOnlyList<IReadOnlyList<(double Start, double End)>>[] Repeated(
        IReadOnlyList<IReadOnlyList<(double Start, double End)>> window, int times = 8) =>
        [.. Enumerable.Repeat(window, times)];

    [Fact]
    public void TwoPeopleTalkingAtOnceAreReported()
    {
        // One from 0 to 6, the other from 4 to 10: they share four seconds.
        var windows = Repeated(Window(Talks(0, 6), Talks(4, 10)));
        var tracks = SpeakerTracks.Link(windows);

        var overlaps = SpeakerTracks.Overlaps(windows, tracks, 10);

        Assert.Single(overlaps);
        Assert.Equal(4, overlaps[0].Start, 1);
        Assert.Equal(6, overlaps[0].End, 1);
    }

    [Fact]
    public void TakingTurnsIsNotOverlap()
    {
        var windows = Repeated(Window(Talks(0, 5), Talks(5, 10)));
        var tracks = SpeakerTracks.Link(windows);

        Assert.Empty(SpeakerTracks.Overlaps(windows, tracks, 10));
    }

    /// <summary>
    /// One window disagreeing with seven others is a boundary drawn a moment early, not two
    /// people speaking. Reporting it as crosstalk would mark every turn in the recording.
    /// </summary>
    [Fact]
    public void OneDissentingWindowIsNotCrosstalk()
    {
        var agreeing = Enumerable.Repeat(Window(Talks(0, 5), Talks(5, 10)), 7);
        var dissenting = new[] { Window(Talks(0, 6), Talks(4, 10)) };

        var windows = agreeing.Concat(dissenting).ToArray();
        var tracks = SpeakerTracks.Link(windows);

        Assert.Empty(SpeakerTracks.Overlaps(windows, tracks, 10));
    }

    /// <summary>A brush of a boundary is not people talking over each other.</summary>
    [Fact]
    public void AVeryBriefBrushIsIgnored()
    {
        var windows = Repeated(Window(Talks(0, 5.2), Talks(5.0, 10)));
        var tracks = SpeakerTracks.Link(windows);

        Assert.Empty(SpeakerTracks.Overlaps(windows, tracks, 10));
    }

    [Fact]
    public void SeveralSeparateStretchesAreReportedSeparately()
    {
        var windows = Repeated(Window(Talks(0, 3, 20, 24), Talks(2, 6, 22, 30)));
        var tracks = SpeakerTracks.Link(windows);

        var overlaps = SpeakerTracks.Overlaps(windows, tracks, 30);

        Assert.Equal(2, overlaps.Count);
        Assert.True(overlaps[0].End < overlaps[1].Start, "the stretches must not run together");
    }

    [Fact]
    public void OneSpeakerAloneOverlapsNobody()
    {
        var windows = Repeated(Window(Talks(0, 10)));
        var tracks = SpeakerTracks.Link(windows);

        Assert.Empty(SpeakerTracks.Overlaps(windows, tracks, 10));
    }

    [Fact]
    public void NothingAtAllIsNotAFailure() =>
        Assert.Empty(SpeakerTracks.Overlaps([], [], 10));
}
