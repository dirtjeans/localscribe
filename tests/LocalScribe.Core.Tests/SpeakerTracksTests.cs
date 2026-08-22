using LocalScribe.Core.Diarization;
using Xunit;

namespace LocalScribe.Core.Tests;

public class SpeakerTracksTests
{
    /// <summary>Spans for one local speaker.</summary>
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

    /// <summary>
    /// The case the whole thing exists for. A window numbers its speakers locally, so the person
    /// called 0 in one window can be 1 in the next; only the clock says they are the same.
    /// </summary>
    [Fact]
    public void LocalNumberingThatSwapsIsFollowedAnyway()
    {
        var windows = new[]
        {
            Window(Talks(0, 5), Talks(6, 10)),      // A talks 0-5, B talks 6-10
            Window(Talks(6, 10), Talks(1, 5)),      // same two people, numbered the other way
        };

        var tracks = SpeakerTracks.Link(windows);

        Assert.Equal(tracks[0][0], tracks[1][1]);
        Assert.Equal(tracks[0][1], tracks[1][0]);
        Assert.NotEqual(tracks[0][0], tracks[0][1]);
    }

    [Fact]
    public void SpeakersAreCarriedAcrossManyWindows()
    {
        // One speaker throughout, seen by six overlapping windows.
        var windows = Enumerable.Range(0, 6)
            .Select(w => Window(Talks(w, w + 10)))
            .ToArray();

        var tracks = SpeakerTracks.Link(windows);

        Assert.All(tracks, window => Assert.Equal(tracks[0][0], window[0]));
    }

    /// <summary>Two people who never share a moment must not be merged.</summary>
    [Fact]
    public void PeopleWhoDoNotOverlapStaySeparate()
    {
        var windows = new[]
        {
            Window(Talks(0, 4)),
            Window(Talks(20, 24)),
        };

        var tracks = SpeakerTracks.Link(windows);

        Assert.NotEqual(tracks[0][0], tracks[1][0]);
    }

    /// <summary>
    /// A glancing overlap at a turn boundary is not evidence of anything. Without a floor, one
    /// frame of shared audio would weld two speakers into one.
    /// </summary>
    [Fact]
    public void AGlancingOverlapDoesNotLinkTwoPeople()
    {
        var windows = new[]
        {
            Window(Talks(0, 5.05)),
            Window(Talks(5.0, 10)),
        };

        var tracks = SpeakerTracks.Link(windows);

        Assert.NotEqual(tracks[0][0], tracks[1][0]);
    }

    /// <summary>
    /// Each local speaker belongs to exactly one person. Letting two of them claim one track
    /// would merge speakers the window had just separated.
    /// </summary>
    [Fact]
    public void TwoLocalSpeakersCannotClaimTheSameTrack()
    {
        var windows = new[]
        {
            Window(Talks(0, 10)),
            Window(Talks(0, 6), Talks(6, 10)),
        };

        var tracks = SpeakerTracks.Link(windows);

        Assert.NotEqual(tracks[1][0], tracks[1][1]);
    }

    [Fact]
    public void TurnsComeOutOfTheVote()
    {
        // Three windows agreeing that A talks 0-5 and B talks 5-10.
        var windows = Enumerable.Range(0, 3)
            .Select(_ => Window(Talks(0, 5), Talks(5, 10)))
            .ToArray();

        var tracks = SpeakerTracks.Link(windows);
        var turns = SpeakerTracks.ToTurns(windows, tracks, 10);

        Assert.Equal(2, turns.Count);
        Assert.Equal(0, turns[0].StartSeconds, 1);
        Assert.Equal(5, turns[0].EndSeconds, 1);
        Assert.Equal(5, turns[1].StartSeconds, 1);
        Assert.NotEqual(turns[0].Speaker, turns[1].Speaker);
    }

    /// <summary>
    /// The redundancy that used to be waste. One window disagreeing with nine others should not
    /// carve a hole in somebody's turn.
    /// </summary>
    [Fact]
    public void OneDissentingWindowIsOutvoted()
    {
        var agreeing = Enumerable.Range(0, 5).Select(_ => Window(Talks(0, 10), Talks(20, 21)));
        var dissenting = new[] { Window(Talks(0, 4), Talks(4, 10)) };

        var windows = agreeing.Concat(dissenting).ToArray();
        var tracks = SpeakerTracks.Link(windows);
        var turns = SpeakerTracks.ToTurns(windows, tracks, 22);

        var first = turns.Where(t => t.StartSeconds < 10).ToList();
        Assert.Single(first);
        Assert.Equal(10, first[0].EndSeconds, 1);
    }

    [Fact]
    public void SilenceIsNotAttributedToAnyone()
    {
        var windows = new[] { Window(Talks(0, 2), Talks(8, 10)) };

        var turns = SpeakerTracks.ToTurns(windows, SpeakerTracks.Link(windows), 10);

        Assert.DoesNotContain(turns, turn => turn.StartSeconds >= 2 && turn.EndSeconds <= 8);
    }

    /// <summary>
    /// Two local speakers in one window are two different people, whatever they sound like.
    /// </summary>
    [Fact]
    public void OverlapInAWindowSeparatesTwoPeople()
    {
        var windows = Enumerable.Range(0, 4)
            .Select(_ => Window(Talks(0, 5), Talks(4, 10)))
            .ToArray();

        var tracks = SpeakerTracks.Link(windows);
        var separated = SpeakerTracks.SeparateTwo(windows, tracks);

        Assert.NotNull(separated);
        Assert.NotEqual(separated![0][0], separated[0][1]);
        Assert.All(separated, w => Assert.All(w, t => Assert.InRange(t, 0, 1)));
    }

    /// <summary>
    /// Islands with no constraint between them are joined on the assumption that consecutive
    /// turns are different people, which is what carries the colouring across a quiet patch.
    /// </summary>
    [Fact]
    public void IslandsAreJoinedByWhoSpokeNext()
    {
        var windows = new[]
        {
            // A and B overlap here, so they are known to differ.
            Window(Talks(0, 6), Talks(5, 10)),
            Window(Talks(0, 6), Talks(5, 10)),
            // A gap, then two more speakers overlapping — a separate island.
            Window(Talks(40, 46), Talks(45, 50)),
            Window(Talks(40, 46), Talks(45, 50)),
        };

        var tracks = SpeakerTracks.Link(windows);
        var separated = SpeakerTracks.SeparateTwo(windows, tracks);

        Assert.NotNull(separated);
        Assert.All(separated!, w => Assert.All(w, t => Assert.True(t == SpeakerTracks.Silent || t is 0 or 1)));
    }

    [Fact]
    public void WithoutAnyOverlapThereIsNothingToSeparate()
    {
        var windows = Enumerable.Range(0, 3).Select(w => Window(Talks(w, w + 5))).ToArray();

        Assert.Null(SpeakerTracks.SeparateTwo(windows, SpeakerTracks.Link(windows)));
    }

    /// <summary>
    /// Reading order, not talking time. A transcript that opens with "Speaker 2" asks the reader
    /// to wonder where Speaker 1 went; numbering by who talks most did precisely that.
    /// </summary>
    [Fact]
    public void WhoeverSpeaksFirstIsSpeakerOne()
    {
        // The second local speaker does nearly all the talking, but starts later.
        var windows = Enumerable.Range(0, 3)
            .Select(_ => Window(Talks(0, 2), Talks(3, 40)))
            .ToArray();

        var turns = SpeakerTracks.ToTurns(windows, SpeakerTracks.Link(windows), 40);

        Assert.Equal(0, turns[0].Speaker);
        Assert.Equal(0, turns.OrderBy(t => t.StartSeconds).First().Speaker);
    }

    [Fact]
    public void NothingAtAllIsNotAFailure()
    {
        Assert.Empty(SpeakerTracks.Link([]));
        Assert.Empty(SpeakerTracks.ToTurns([], [], 10));
    }
}
