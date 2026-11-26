using System.Buffers.Binary;
using System.Text;
using LocalScribe.Core.Audio;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// Several recordings of one event, put in the order they were made: the strongest clue all of
/// them carry decides, anything that disagrees is said, and each container's own record of when
/// recording began is read where recorders actually write it.
/// </summary>
public class RecordingOrderTests
{
    private static readonly DateTimeOffset Nine = new(2026, 11, 20, 9, 0, 0, TimeSpan.Zero);

    private static RecordingFacts File(string name, DateTimeOffset? recorded = null, int? track = null,
        double? seconds = null, DateTimeOffset? saved = null) =>
        new($"/r/{name}.m4a", recorded, track, seconds, saved ?? Nine.AddDays(1));

    private static string[] Names(RecordingOrder.Proposal proposal) => [.. proposal.Ordered.Select(f => f.Name)];

    /* ---- deciding ----------------------------------------------------------------------- */

    [Fact]
    public void RecordedTimesBeatEverythingElse()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("b", recorded: Nine.AddMinutes(30)),
            File("c", recorded: Nine.AddMinutes(60)),
            File("a", recorded: Nine),
        ]);

        Assert.Equal(["a", "b", "c"], Names(proposal));
        Assert.Equal(OrderBasis.RecordedTimes, proposal.Basis);
        Assert.Empty(proposal.Doubts);
    }

    [Fact]
    public void ATimeOnlySomeFilesCarryCannotPlaceTheRest()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("Part 2", recorded: Nine, track: 2),
            File("Part 1", track: 1),
        ]);

        Assert.Equal(OrderBasis.TrackNumbers, proposal.Basis);
        Assert.Equal(["Part 1", "Part 2"], Names(proposal));
    }

    [Fact]
    public void TwoFilesClaimingTheSameMomentSayNothingAboutOrder()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("Part 10", recorded: Nine),
            File("Part 2", recorded: Nine),
        ]);

        Assert.Equal(OrderBasis.FileNames, proposal.Basis);
        Assert.Equal(["Part 2", "Part 10"], Names(proposal));
    }

    [Fact]
    public void NumbersInNamesAreReadAsNumbers()
    {
        var proposal = RecordingOrder.Decide([File("Meeting 10"), File("Meeting 9"), File("Meeting 1")]);

        Assert.Equal(["Meeting 1", "Meeting 9", "Meeting 10"], Names(proposal));
    }

    [Fact]
    public void TheCopyWithoutANumberComesBeforeTheNumberedOnes()
    {
        // What the Finder calls the second and third recordings made under one name.
        var proposal = RecordingOrder.Decide([File("Recording 3"), File("Recording"), File("Recording 2")]);

        Assert.Equal(OrderBasis.FileNames, proposal.Basis);
        Assert.Equal(["Recording", "Recording 2", "Recording 3"], Names(proposal));
    }

    [Fact]
    public void PartLettersInNamesAreASequence()
    {
        var proposal = RecordingOrder.Decide([File("Session C"), File("Session A"), File("Session B")]);

        Assert.Equal(OrderBasis.FileNames, proposal.Basis);
        Assert.Equal(["Session A", "Session B", "Session C"], Names(proposal));
    }

    [Fact]
    public void ALetterInsideAWordIsNotAPartLetter()
    {
        // "Plan" and "Planb" differ by a letter, but not by a part letter standing alone.
        var proposal = RecordingOrder.Decide([File("Planb", saved: Nine), File("Plan", saved: Nine.AddHours(1))]);

        Assert.Equal(OrderBasis.FileTimes, proposal.Basis);
    }

    [Fact]
    public void PartLettersThatDisagreeWithTheRecordedTimesAreRaised()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("Session A", recorded: Nine.AddMinutes(1)),
            File("Session C", recorded: Nine),
        ]);

        Assert.Equal(["Session C", "Session A"], Names(proposal));
        Assert.Contains(proposal.Doubts, doubt => doubt.Contains("file names", StringComparison.Ordinal));
    }

    [Fact]
    public void UnrelatedNamesThatHappenToHoldNumbersAreNotASequence()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("ken-mindy_2023-jul-31-0642pm_forecast", saved: Nine.AddHours(2)),
            File("Mindy", saved: Nine),
            File("Mindy 1", saved: Nine.AddHours(1)),
        ]);

        Assert.Equal(OrderBasis.FileTimes, proposal.Basis);
        Assert.Equal(["Mindy", "Mindy 1", "ken-mindy_2023-jul-31-0642pm_forecast"], Names(proposal));
    }

    [Fact]
    public void NamesStampedWithTheirTimeAreASequence()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("Meeting 2026-11-20 10.30"),
            File("Meeting 2026-11-20 09.45"),
        ]);

        Assert.Equal(OrderBasis.FileNames, proposal.Basis);
        Assert.Equal(["Meeting 2026-11-20 09.45", "Meeting 2026-11-20 10.30"], Names(proposal));
    }

    [Fact]
    public void WithNothingElseTheSaveTimesDecideAndSaySo()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("Later", saved: Nine.AddHours(2)),
            File("Earlier", saved: Nine.AddHours(1)),
        ]);

        Assert.Equal(OrderBasis.FileTimes, proposal.Basis);
        Assert.Equal(["Earlier", "Later"], Names(proposal));
        Assert.Single(proposal.Doubts);
    }

    [Fact]
    public void SaveTimesCountFromWhenEachFileWasBegun()
    {
        // The long one was saved last but started first: an hour long, saved at 10:05.
        var proposal = RecordingOrder.Decide(
        [
            File("Short", seconds: 300, saved: Nine.AddMinutes(30)),
            File("Long", seconds: 3600, saved: Nine.AddMinutes(65)),
        ]);

        Assert.Equal(["Long", "Short"], Names(proposal));
    }

    [Fact]
    public void NamesThatDisagreeWithTheRecordedTimesAreRaised()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("Part 1", recorded: Nine.AddMinutes(30)),
            File("Part 2", recorded: Nine),
        ]);

        Assert.Equal(["Part 2", "Part 1"], Names(proposal));
        Assert.Contains(proposal.Doubts, doubt => doubt.Contains("file names", StringComparison.Ordinal));
    }

    [Fact]
    public void GapsAreMeasuredFromTheEndOfTheFileBefore()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("a", recorded: Nine, seconds: 600),
            File("b", recorded: Nine.AddSeconds(600)),
            File("c", recorded: Nine.AddSeconds(1500)),
        ]);

        Assert.Null(proposal.GapsBefore[0]);
        Assert.Equal(0, proposal.GapsBefore[1]);
        Assert.Null(proposal.GapsBefore[2]);
    }

    [Fact]
    public void GapsAreNotShownWhenSomethingElseDecided()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("Part 1", recorded: Nine, seconds: 60),
            File("Part 2", recorded: Nine.AddDays(50)),
            File("Part 3"),
        ]);

        Assert.Equal(OrderBasis.FileNames, proposal.Basis);
        Assert.All(proposal.GapsBefore, gap => Assert.Null(gap));
    }

    [Fact]
    public void PartsRecordedHoursApartAreRaised()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("a", recorded: Nine, seconds: 600),
            File("b", recorded: Nine.AddHours(3)),
        ]);

        Assert.Contains(proposal.Doubts, doubt => doubt.Contains("half an hour", StringComparison.Ordinal));
    }

    [Fact]
    public void OverlappingRecordingsAreRaised()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("Mic A", recorded: Nine, seconds: 600),
            File("Mic B", recorded: Nine.AddSeconds(5), seconds: 600),
        ]);

        Assert.True(proposal.GapsBefore[1] < 0);
        Assert.Contains(proposal.Doubts, doubt => doubt.Contains("overlap", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData(new[] { "Interview part 1", "Interview part 2" }, "Interview")]
    [InlineData(new[] { "ZOOM0001", "ZOOM0002" }, "ZOOM")]
    [InlineData(new[] { "Recording", "Recording 2" }, "Recording")]
    [InlineData(new[] { "Board meeting (1)", "Board meeting (2)" }, "Board meeting")]
    [InlineData(new[] { "Alpha", "Beta", "Gamma" }, "Alpha and 2 more")]
    [InlineData(new[] { "Only" }, "Only")]
    public void TogetherTheyAreCalledWhatTheirNamesShare(string[] names, string expected)
    {
        Assert.Equal(expected, RecordingOrder.CombinedName(names));
    }

    [Fact]
    public void TwoMicrophonesOnOneSessionAreRaisedNotJoinedSilently()
    {
        // A Zoom H6 take: both tracks stamped with the same start, so the names decide the order.
        var proposal = RecordingOrder.Decide(
        [
            File("ZOOM0001_Tr2", recorded: Nine, seconds: 600),
            File("ZOOM0001_Tr1", recorded: Nine, seconds: 600),
        ]);

        Assert.Equal(OrderBasis.FileNames, proposal.Basis);
        Assert.Contains(proposal.Doubts, doubt => doubt.Contains("overlap", StringComparison.Ordinal));
    }

    [Fact]
    public void TiedTimesAreNotCalledMissing()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("Lapel", recorded: Nine, saved: Nine.AddHours(2)),
            File("Boom", recorded: Nine, saved: Nine.AddHours(1)),
        ]);

        Assert.Equal(OrderBasis.FileTimes, proposal.Basis);
        Assert.DoesNotContain(proposal.Doubts, doubt => doubt.StartsWith("Nothing in these files", StringComparison.Ordinal));
    }

    [Fact]
    public void DatesWrittenDayFirstAreNotASequence()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("Board meeting 05.12.2026", saved: Nine.AddDays(7)),
            File("Board meeting 28.11.2026", saved: Nine),
        ]);

        Assert.Equal(OrderBasis.FileTimes, proposal.Basis);
        Assert.Equal(["Board meeting 28.11.2026", "Board meeting 05.12.2026"], Names(proposal));
    }

    [Fact]
    public void DatesWrittenMonthFirstAreNotASequence()
    {
        var proposal = RecordingOrder.Decide([File("Call 12-31-2025"), File("Call 01-02-2026")]);

        Assert.NotEqual(OrderBasis.FileNames, proposal.Basis);
    }

    [Fact]
    public void PhoneNumbersAreNotPartNumbers()
    {
        var proposal = RecordingOrder.Decide([File("Call with 07700900123"), File("Call with 02079460000")]);

        Assert.NotEqual(OrderBasis.FileNames, proposal.Basis);
    }

    [Fact]
    public void PartNumbersBesideAFixedTotalAreASequence()
    {
        var proposal = RecordingOrder.Decide([File("Part 2 of 3"), File("Part 1 of 3"), File("Part 3 of 3")]);

        Assert.Equal(OrderBasis.FileNames, proposal.Basis);
        Assert.Equal(["Part 1 of 3", "Part 2 of 3", "Part 3 of 3"], Names(proposal));
    }

    [Fact]
    public void SaveTimesThatDisagreeWithTheNamesAreRaised()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("Part 1", saved: Nine.AddHours(2)),
            File("Part 2", saved: Nine.AddHours(1)),
        ]);

        Assert.Equal(["Part 1", "Part 2"], Names(proposal));
        Assert.Contains(proposal.Doubts, doubt => doubt.Contains("saved", StringComparison.Ordinal));
    }

    [Fact]
    public void ATiedPairIsRaisedEvenBesideAFileWithNoTime()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("ZOOM0004_Tr1", recorded: Nine),
            File("ZOOM0004_Tr2", recorded: Nine),
            File("ZOOM0004_Tr3"),
        ]);

        Assert.Contains(proposal.Doubts, doubt => doubt.Contains("overlap", StringComparison.Ordinal));
    }

    [Fact]
    public void PartlyTimedFilesAreNotCalledUntimed()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("Lapel", recorded: Nine, saved: Nine.AddHours(2)),
            File("Notes", saved: Nine.AddHours(1)),
        ]);

        Assert.Equal(OrderBasis.FileTimes, proposal.Basis);
        Assert.Contains(proposal.Doubts, doubt => doubt.StartsWith("Only some", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("Meeting 2811", "Meeting 0512")]
    [InlineData("Rec 1430", "Rec 0915")]
    [InlineData("Call 555-1234", "Call 555-0987")]
    [InlineData("Call 020 7946 0991", "Call 020 7946 0018")]
    [InlineData("Call", "Call 1234")]
    public void FourDigitNumbersAndPhoneGroupsAreNotPartNumbers(string first, string second)
    {
        Assert.NotEqual(OrderBasis.FileNames, RecordingOrder.Decide([File(first), File(second)]).Basis);
    }

    [Theory]
    [InlineData("ZOOM0001", "ZOOM0002")]
    [InlineData("Recording 10", "Recording 9")]
    [InlineData("Part 1 of 3", "Part 2 of 3")]
    public void PaddedCountersAndShortNumbersArePartNumbers(string first, string second)
    {
        Assert.Equal(OrderBasis.FileNames, RecordingOrder.Decide([File(first), File(second)]).Basis);
    }

    [Fact]
    public void OneStrictDisagreementInSaveTimesIsRaisedDespiteATie()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("Part 1", saved: Nine.AddHours(3)),
            File("Part 2", saved: Nine.AddHours(1)),
            File("Part 3", saved: Nine.AddHours(1)),
        ]);

        Assert.Contains(proposal.Doubts, doubt => doubt.Contains("saved", StringComparison.Ordinal));
    }

    [Fact]
    public void AnImpossibleLengthDoesNotBreakTheSaveTimeOrder()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("b", seconds: double.MaxValue, saved: Nine.AddHours(2)),
            File("a", seconds: double.NaN, saved: Nine.AddHours(1)),
        ]);

        Assert.Equal(["a", "b"], Names(proposal));
    }

    /// <summary>
    /// Nine clips from a Pixel, as they came off the phone on 2026-11-26: each named for when it
    /// started, in UTC, and each stamped inside with when it stopped — the name's time plus its
    /// length. Read as starts, the stamps made four of the joins overlap by minutes.
    /// </summary>
    private static readonly (string Name, string Stamp, double Seconds)[] PixelClips =
    [
        ("PXL_20261126_180535720", "18:05:40", 2.1),
        ("PXL_20261126_180604104", "18:13:48", 463.7),
        ("PXL_20261126_181403955", "18:14:21", 16.6),
        ("PXL_20261126_181429650", "18:14:50", 20.4),
        ("PXL_20261126_181516986", "18:31:08", 951.2),
        ("PXL_20261126_183118679", "18:40:12", 533.3),
        ("PXL_20261126_184019273", "18:40:58", 38.4),
        ("PXL_20261126_184135912", "18:43:35", 119.3),
        ("PXL_20261126_184438574", "18:44:58", 19.5),
    ];

    private static List<RecordingFacts> Pixel() =>
    [
        // Enumerable by name: from C# 14 an array's .Reverse() binds to the in-place span method.
        .. Enumerable.Reverse(PixelClips).Select(clip => File(
            clip.Name,
            recorded: DateTimeOffset.Parse($"2026-11-26T{clip.Stamp}Z", System.Globalization.CultureInfo.InvariantCulture),
            seconds: clip.Seconds)),
    ];

    [Fact]
    public void APixelsStampsAreReadAsWhenEachClipStopped()
    {
        var proposal = RecordingOrder.Decide(Pixel());

        Assert.True(proposal.StampsMarkEnds);
        Assert.Equal(PixelClips.Select(c => c.Name), Names(proposal));
        Assert.DoesNotContain(proposal.Doubts, doubt => doubt.Contains("overlap", StringComparison.Ordinal));
    }

    [Fact]
    public void APixelsClipsFollowOnWithTheGapsTheirNamesSay()
    {
        var proposal = RecordingOrder.Decide(Pixel());

        // Between 7 and 63 seconds between clips, by their names' start times and lengths.
        Assert.All(proposal.GapsBefore.Skip(1), gap => Assert.InRange(gap!.Value, 5, 65));
    }

    [Fact]
    public void APixelsClipStartsWhenItsNameSays()
    {
        var proposal = RecordingOrder.Decide(Pixel());
        var clip = proposal.Ordered.Single(f => f.Name == "PXL_20261126_180604104");

        var start = proposal.StartOf(clip)!.Value;

        Assert.Equal(new DateTimeOffset(2026, 11, 26, 18, 6, 4, TimeSpan.Zero), start.AddMilliseconds(-start.Millisecond), TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void APixelsNamesAgreeWithItsStamps()
    {
        var proposal = RecordingOrder.Decide(Pixel());

        Assert.DoesNotContain(proposal.Doubts, doubt => doubt.Contains("file names", StringComparison.Ordinal));
    }

    [Fact]
    public void StampsAreReadAsStartsWhenThatFitsAsWell()
    {
        // Ten minutes apart, two minutes long: no overlap either way, so the standard reading.
        var proposal = RecordingOrder.Decide(
        [
            File("a", recorded: Nine, seconds: 120),
            File("b", recorded: Nine.AddMinutes(10), seconds: 120),
        ]);

        Assert.False(proposal.StampsMarkEnds);
        Assert.Equal(480, proposal.GapsBefore[1]);
    }

    [Fact]
    public void RecordingsThatOverlapEitherWayAreStillRaised()
    {
        // Two microphones on one ten-minute session, started a few seconds apart.
        var proposal = RecordingOrder.Decide(
        [
            File("Mic A", recorded: Nine, seconds: 600),
            File("Mic B", recorded: Nine.AddSeconds(4), seconds: 596),
        ]);

        Assert.False(proposal.StampsMarkEnds);
        Assert.Contains(proposal.Doubts, doubt => doubt.Contains("overlap", StringComparison.Ordinal));
    }

    [Fact]
    public void WithoutEveryLengthTheStampsAreReadAsTheStandardMeansThem()
    {
        var proposal = RecordingOrder.Decide(
        [
            File("a", recorded: Nine.AddSeconds(463), seconds: 463),
            File("b", recorded: Nine.AddSeconds(480)),
        ]);

        Assert.False(proposal.StampsMarkEnds);
    }

    [Fact]
    public void ASecondMicrophoneInsideTheFirstIsStillRaised()
    {
        // The room mic ran the whole hour; the lapel started half an hour in and ran twenty
        // minutes. Read as stops, these would follow on, which is why stamps alone cannot decide.
        var proposal = RecordingOrder.Decide(
        [
            File("Room", recorded: Nine, seconds: 3600),
            File("Lapel", recorded: Nine.AddMinutes(30), seconds: 1200),
        ]);

        Assert.False(proposal.StampsMarkEnds);
        Assert.Equal(Nine, proposal.StartOf(proposal.Ordered[0]));
        Assert.Contains(proposal.Doubts, doubt => doubt.Contains("overlap", StringComparison.Ordinal));
    }

    [Fact]
    public void ClipsStampedAsTheyStoppedButRenamedAreRaisedWithBothReadings()
    {
        // The Pixel's clips with their names gone: nothing vouches for reading the stamps as stops.
        var renamed = Pixel().Select((clip, i) => clip with { Path = $"/r/Clip {i}.m4a" }).ToList();

        var proposal = RecordingOrder.Decide(renamed);

        Assert.False(proposal.StampsMarkEnds);
        Assert.Contains(proposal.Doubts, doubt => doubt.Contains("stopped instead", StringComparison.Ordinal));
    }

    [Fact]
    public void AStartStampedMemoAmongPixelClipsKeepsItsOwnStart()
    {
        var clips = Pixel();
        var memoStart = DateTimeOffset.Parse("2026-11-26T18:14:55Z", System.Globalization.CultureInfo.InvariantCulture);
        clips.Add(File("Memo", recorded: memoStart, seconds: 15));

        var proposal = RecordingOrder.Decide(clips);
        var memo = proposal.Ordered.Single(f => f.Name == "Memo");

        Assert.Equal(memoStart, proposal.StartOf(memo));
        Assert.DoesNotContain(proposal.Doubts, doubt => doubt.Contains("overlap", StringComparison.Ordinal));
    }

    [Fact]
    public void NamesInLocalTimeStillVouchForStampsInUtc()
    {
        // A Samsung names its clips in local time; the stamp inside is UTC, eight hours on.
        var proposal = RecordingOrder.Decide(
        [
            File("20261126_100604", recorded: DateTimeOffset.Parse("2026-11-26T18:13:48Z", System.Globalization.CultureInfo.InvariantCulture), seconds: 464),
            File("20261126_101403", recorded: DateTimeOffset.Parse("2026-11-26T18:14:21Z", System.Globalization.CultureInfo.InvariantCulture), seconds: 18),
        ]);

        Assert.True(proposal.StampsMarkEnds);
        Assert.DoesNotContain(proposal.Doubts, doubt => doubt.Contains("overlap", StringComparison.Ordinal));
    }

    [Fact]
    public void UnpaddedFieldTimesInNamesAreStillASequence()
    {
        var proposal = RecordingOrder.Decide([File("Meeting 2026-11-20 10.30"), File("Meeting 2026-11-20 9.45")]);

        Assert.Equal(OrderBasis.FileNames, proposal.Basis);
        Assert.Equal(["Meeting 2026-11-20 9.45", "Meeting 2026-11-20 10.30"], Names(proposal));
    }

    [Theory]
    [InlineData("REC_20261126_1230PM", "REC_20261126_0115PM")]
    [InlineData("Recording 2026-11-26 12.30 PM", "Recording 2026-11-26 01.15 PM")]
    [InlineData("Rec 12 PM", "Rec 1 PM")]
    public void TwelveHourTimesInNamesAreNotASequence(string first, string second)
    {
        Assert.NotEqual(OrderBasis.FileNames, RecordingOrder.Decide([File(first), File(second)]).Basis);
    }

    [Fact]
    public void NamesSortOnTheirNumbersNotTheirPunctuation()
    {
        var proposal = RecordingOrder.Decide([File("PXL 20261126 181403955"), File("PXL_20261126_180604104")]);

        Assert.Equal(["PXL_20261126_180604104", "PXL 20261126 181403955"], Names(proposal));
    }

    [Theory]
    [InlineData("PXL_20261126_184438574", "PXL_20261126_180535720")]
    [InlineData("20261126_184438", "20261126_180535")]
    [InlineData("VID_20261127_0910", "VID_20261126_2245")]
    public void PhoneStyleTimestampsInNamesAreASequence(string later, string earlier)
    {
        var proposal = RecordingOrder.Decide([File(later), File(earlier)]);

        Assert.Equal(OrderBasis.FileNames, proposal.Basis);
        Assert.Equal([earlier, later], Names(proposal));
    }

    [Theory]
    [InlineData("PXL_20261126_180535", "PXL_20261126_180604104")]
    [InlineData("REC_20261399_1200", "REC_20261126_1200")]
    [InlineData("Clip 12345678 1200", "Clip 12345679 1200")]
    public void CompactRunsThatAreNotATimestampAreNotASequence(string first, string second)
    {
        Assert.NotEqual(OrderBasis.FileNames, RecordingOrder.Decide([File(first), File(second)]).Basis);
    }

    /* ---- reading ------------------------------------------------------------------------ */

    private static RecordingFacts Read(byte[] bytes) =>
        RecordingFacts.Read(new MemoryStream(bytes), new RecordingFacts("/r/x", null, null, null, null));

    [Fact]
    public void AMovieHeaderSaysWhenRecordingBeganAndHowLong()
    {
        var facts = Read(Mp4(created: Nine, timescale: 1000, duration: 754_000, track: 3));

        Assert.Equal(Nine, facts.RecordedAt);
        Assert.Equal(754, facts.DurationSeconds);
        Assert.Equal(3, facts.Track);
    }

    [Fact]
    public void AnUnsetMovieHeaderTimeIsNotATime()
    {
        Assert.Null(Read(Mp4(created: null, timescale: 1000, duration: 1000)).RecordedAt);
    }

    [Fact]
    public void ABroadcastWaveSaysWhenItsRecorderStarted()
    {
        var facts = Read(Wave(bext: ("2026-11-20", "09:15:30"), seconds: 2));

        Assert.Equal(new DateTimeOffset(new DateTime(2026, 11, 20, 9, 15, 30, DateTimeKind.Local)), facts.RecordedAt);
        Assert.Equal(2, facts.DurationSeconds);
    }

    [Fact]
    public void ABroadcastWaveMayUseOtherSeparators()
    {
        var facts = Read(Wave(bext: ("2026:11:20", "09-15-30"), seconds: 1));

        Assert.Equal(new DateTime(2026, 11, 20, 9, 15, 30), facts.RecordedAt?.LocalDateTime);
    }

    [Fact]
    public void APlainWaveStillKnowsItsLength()
    {
        var facts = Read(Wave(bext: null, seconds: 3));

        Assert.Null(facts.RecordedAt);
        Assert.Equal(3, facts.DurationSeconds);
    }

    [Fact]
    public void Id3v24CarriesTheWholeMomentInOneFrame()
    {
        var facts = Read(Id3(4, ("TDRC", "2026-11-20T09:15:30"), ("TRCK", "2/5")));

        Assert.Equal(new DateTime(2026, 11, 20, 9, 15, 30), facts.RecordedAt?.LocalDateTime);
        Assert.Equal(2, facts.Track);
    }

    [Fact]
    public void Id3v23SplitsItIntoYearDayAndTime()
    {
        var facts = Read(Id3(3, ("TYER", "2026"), ("TDAT", "2011"), ("TIME", "0915")));

        Assert.Equal(new DateTime(2026, 11, 20, 9, 15, 0), facts.RecordedAt?.LocalDateTime);
    }

    [Fact]
    public void AYearAloneIsNotARecordingTime()
    {
        Assert.Null(Read(Id3(4, ("TDRC", "2026"))).RecordedAt);
    }

    [Fact]
    public void FlacSaysItsLengthAndItsComments()
    {
        var facts = Read(Flac(rate: 44_100, samples: 44_100 * 90, ("DATE", "2026-11-20 09:15"), ("TRACKNUMBER", "4")));

        Assert.Equal(90, facts.DurationSeconds);
        Assert.Equal(4, facts.Track);
        Assert.Equal(new DateTime(2026, 11, 20, 9, 15, 0), facts.RecordedAt?.LocalDateTime);
    }

    [Fact]
    public void SomethingElseEntirelyKeepsWhatWasKnown()
    {
        var known = new RecordingFacts("/r/x", null, null, null, Nine);

        Assert.Same(known, RecordingFacts.Read(new MemoryStream("not audio at all"u8.ToArray()), known));
    }

    [Fact]
    public void AFileCutShortIsNotAnError()
    {
        var whole = Mp4(created: Nine, timescale: 1000, duration: 1000);

        Assert.Null(Read(whole[..^20]).DurationSeconds);
    }

    [Fact]
    public async Task AHugeBoxSizeEndsTheWalkRatherThanLoopingForever()
    {
        // moov holding an 8-byte free box, then a free box whose 64-bit size is near the maximum:
        // added to the position it wrapped negative, passed the check, and walked backwards.
        var huge = new byte[16];
        BinaryPrimitives.WriteUInt32BigEndian(huge, 1);
        Encoding.ASCII.GetBytes("free").CopyTo(huge, 4);
        BinaryPrimitives.WriteUInt64BigEndian(huge.AsSpan(8), 0x7FFFFFFFFFFFFFF8);
        byte[] model = [.. Box("ftyp", "M4A \0\0\0\0"u8.ToArray()), .. Box("moov", Box("free"), huge)];

        var read = Task.Run(() => Read(model));
        var first = await Task.WhenAny(read, Task.Delay(TimeSpan.FromSeconds(5)));

        Assert.True(first == read, "the box walk did not end");
    }

    [Fact]
    public void AnEmptyMovieHeaderIsNotAnError()
    {
        byte[] model = [.. Box("ftyp", "M4A \0\0\0\0"u8.ToArray()), .. Box("moov", Box("mvhd"))];

        Assert.Null(Read(model).RecordedAt);
    }

    [Fact]
    public void AnUnknownMovieLengthIsNotALength()
    {
        // Version 1, all ones: what a writer that does not know the duration is told to write.
        var mvhd = new byte[112];
        mvhd[0] = 1;
        BinaryPrimitives.WriteUInt32BigEndian(mvhd.AsSpan(20), 1000);
        BinaryPrimitives.WriteUInt64BigEndian(mvhd.AsSpan(24), ulong.MaxValue);
        byte[] model = [.. Box("ftyp", "M4A \0\0\0\0"u8.ToArray()), .. Box("moov", Box("mvhd", mvhd))];

        Assert.Null(Read(model).DurationSeconds);
    }

    [Fact]
    public void AnId3FrameSizeNearTheLimitEndsTheWalk()
    {
        var frame = new byte[10];
        Encoding.ASCII.GetBytes("APIC").CopyTo(frame, 0);
        BinaryPrimitives.WriteUInt32BigEndian(frame.AsSpan(4), 0x7FFFFFF8);
        byte[] tag = [.. frame, .. new byte[54]];
        byte[] mp3 = [.. "ID3"u8.ToArray(), 3, 0, 0, .. Synchsafe(tag.Length), .. tag];

        Assert.Null(Read(mp3).Track);
    }

    [Fact]
    public void AnId3ExtendedHeaderIsSkipped()
    {
        var plain = Id3(3, ("TRCK", "3"));

        // Flag 0x40 and a six-byte extended header (its size field, then six bytes) before the frames.
        byte[] extended = [0, 0, 0, 6, 0, 0, 0, 0, 0, 0];
        var body = plain[10..];
        byte[] withExtended = [.. "ID3"u8.ToArray(), 3, 0, 0x40, .. Synchsafe(body.Length + extended.Length), .. extended, .. body];

        Assert.Equal(3, Read(withExtended).Track);
    }

    [Fact]
    public void AnUnsynchronisedId3TagIsRestoredBeforeItsFramesAreRead()
    {
        // A frame holding 0xFF 0x00 0xE0 as written for 0xFF 0xE0, ahead of the track number.
        var priv = new byte[] { (byte)'P', (byte)'R', (byte)'I', (byte)'V', 0, 0, 0, 3, 0, 0, 0xFF, 0xE0, 0x41 };
        var unsynced = new List<byte>();
        for (var i = 0; i < priv.Length; i++)
        {
            unsynced.Add(priv[i]);
            if (priv[i] == 0xFF)
            {
                unsynced.Add(0x00);
            }
        }

        var track = Id3(3, ("TRCK", "7"))[10..];
        byte[] tag = [.. unsynced, .. track];
        byte[] mp3 = [.. "ID3"u8.ToArray(), 3, 0, 0x80, .. Synchsafe(tag.Length), .. tag];

        Assert.Equal(7, Read(mp3).Track);
    }

    [Fact]
    public void AChunkCutShortKeepsWhatWasReadBeforeIt()
    {
        var whole = Wave(bext: ("2026-11-20", "09:15:30"), seconds: 1);
        byte[] cut = [.. whole, .. "LIST"u8.ToArray(), .. BitConverter.GetBytes(4000), .. new byte[8]];

        var facts = Read(cut);

        Assert.NotNull(facts.RecordedAt);
        Assert.Equal(1, facts.DurationSeconds);
    }

    [Fact]
    public void AFlacCommentBlockCutShortKeepsTheLength()
    {
        var whole = Flac(rate: 8_000, samples: 8_000 * 5, ("DATE", "2026-11-20 09:15"));

        Assert.Equal(5, Read(whole[..60]).DurationSeconds);
    }

    [Theory]
    [InlineData(44_100)]
    [InlineData(48_000)]
    public void AnUnknownVersion0MovieLengthIsNotALengthAtAudioTimescales(uint timescale)
    {
        Assert.Null(Read(Mp4(created: Nine, timescale: timescale, duration: uint.MaxValue)).DurationSeconds);
    }

    [Fact]
    public async Task AnUnsizedDataChunkIsTheRestOfTheFileAndIsNotWalked()
    {
        var wave = Wave(bext: ("2026-11-20", "09:15:30"), seconds: 64);
        var data = wave.AsSpan().IndexOf("data"u8);
        BinaryPrimitives.WriteUInt32LittleEndian(wave.AsSpan(data + 4), 0);

        var read = Task.Run(() => Read(wave));
        var first = await Task.WhenAny(read, Task.Delay(TimeSpan.FromSeconds(2)));

        Assert.True(first == read, "the silence was walked as chunks");
        Assert.Equal(64, Math.Round((await read).DurationSeconds ?? 0));
        Assert.NotNull((await read).RecordedAt);
    }

    [Fact]
    public void AnInfoSubchunkSizeNearTheLimitKeepsTheTrackReadBeforeIt()
    {
        var info = new List<byte>();
        info.AddRange("INFO"u8.ToArray());
        info.AddRange("ITRK"u8.ToArray());
        info.AddRange(BitConverter.GetBytes(2));
        info.AddRange("3\0"u8.ToArray());
        info.AddRange("ICMT"u8.ToArray());
        info.AddRange(BitConverter.GetBytes(0x7FFFFFF4));
        info.AddRange(new byte[8]);

        byte[] wave = [.. Wave(bext: null, seconds: 1), .. Chunk("LIST", [.. info])];

        Assert.Equal(3, Read(wave).Track);
    }

    [Theory]
    [InlineData(0x7FFFFFFF)]
    [InlineData(1000)]
    public void ABadVorbisVendorLengthKeepsTheFlacLength(int vendorLength)
    {
        var flac = Flac(rate: 8_000, samples: 8_000 * 5, ("TRACKNUMBER", "2"));
        var vorbis = 4 + 4 + 34 + 4;
        BinaryPrimitives.WriteInt32LittleEndian(flac.AsSpan(vorbis), vendorLength);

        Assert.Equal(5, Read(flac).DurationSeconds);
    }

    [Fact]
    public void AnId3v24TagUnsynchronisedAsAWholeIsRestored()
    {
        // UTF-16 with a byte-order mark FF FE, written FF 00 FE under unsynchronisation.
        byte[] value = [1, 0xFF, 0x00, 0xFE, (byte)'7', 0];
        var frame = new byte[10];
        Encoding.ASCII.GetBytes("TRCK").CopyTo(frame, 0);
        Synchsafe(value.Length).CopyTo(frame, 4);
        byte[] tag = [.. frame, .. value, .. new byte[16]];
        byte[] mp3 = [.. "ID3"u8.ToArray(), 4, 0, 0x80, .. Synchsafe(tag.Length), .. tag];

        Assert.Equal(7, Read(mp3).Track);
    }

    [Fact]
    public void AGroupedId3FrameIsReadPastItsGroupByte()
    {
        byte[] value = [9, 3, (byte)'4'];
        var frame = new byte[10];
        Encoding.ASCII.GetBytes("TRCK").CopyTo(frame, 0);
        BinaryPrimitives.WriteUInt32BigEndian(frame.AsSpan(4), (uint)value.Length);
        frame[9] = 0x20;
        byte[] tag = [.. frame, .. value, .. new byte[16]];
        byte[] mp3 = [.. "ID3"u8.ToArray(), 3, 0, 0, .. Synchsafe(tag.Length), .. tag];

        Assert.Equal(4, Read(mp3).Track);
    }

    [Theory]
    [InlineData("2026-11-20")]
    [InlineData("05/12/2026 10:00")]
    [InlineData("28.11.2026 10:00")]
    public void ADateWithoutATimeOrWrittenDayFirstIsNotARecordingTime(string date)
    {
        Assert.Null(Read(Id3(4, ("TDRC", date))).RecordedAt);
    }

    [Fact]
    public void Id3v23WithoutATimeOfDayIsNotARecordingTime()
    {
        Assert.Null(Read(Id3(3, ("TYER", "2026"), ("TDAT", "2011"))).RecordedAt);
    }

    /* ---- writers: just the fields that are read ----------------------------------------- */

    private static byte[] Box(string type, params byte[][] content)
    {
        var body = content.SelectMany(b => b).ToArray();
        var box = new byte[8 + body.Length];
        BinaryPrimitives.WriteUInt32BigEndian(box, (uint)box.Length);
        Encoding.ASCII.GetBytes(type).CopyTo(box, 4);
        body.CopyTo(box, 8);
        return box;
    }

    private static byte[] Mp4(DateTimeOffset? created, uint timescale, uint duration, int? track = null)
    {
        var mvhd = new byte[100];
        var seconds = created is { } at ? (uint)(at - new DateTimeOffset(1904, 1, 1, 0, 0, 0, TimeSpan.Zero)).TotalSeconds : 0;
        BinaryPrimitives.WriteUInt32BigEndian(mvhd.AsSpan(4), seconds);
        BinaryPrimitives.WriteUInt32BigEndian(mvhd.AsSpan(8), seconds);
        BinaryPrimitives.WriteUInt32BigEndian(mvhd.AsSpan(12), timescale);
        BinaryPrimitives.WriteUInt32BigEndian(mvhd.AsSpan(16), duration);

        var moov = new List<byte[]> { Box("mvhd", mvhd) };

        if (track is { } number)
        {
            var data = new byte[16];
            BinaryPrimitives.WriteUInt16BigEndian(data.AsSpan(10), (ushort)number);
            moov.Add(Box("udta", Box("meta", new byte[4], Box("ilst", Box("trkn", Box("data", data))))));
        }

        return [.. Box("ftyp", "M4A \0\0\0\0"u8.ToArray()), .. Box("mdat", new byte[32]), .. Box("moov", [.. moov])];
    }

    private static byte[] Chunk(string id, byte[] body)
    {
        var chunk = new byte[8 + body.Length + (body.Length % 2)];
        Encoding.ASCII.GetBytes(id).CopyTo(chunk, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(chunk.AsSpan(4), (uint)body.Length);
        body.CopyTo(chunk, 8);
        return chunk;
    }

    private static byte[] Wave((string Date, string Time)? bext, int seconds)
    {
        var format = new byte[16];
        BinaryPrimitives.WriteUInt16LittleEndian(format, 1);
        BinaryPrimitives.WriteUInt16LittleEndian(format.AsSpan(2), 1);
        BinaryPrimitives.WriteUInt32LittleEndian(format.AsSpan(4), 16_000);
        BinaryPrimitives.WriteUInt32LittleEndian(format.AsSpan(8), 32_000);
        BinaryPrimitives.WriteUInt16LittleEndian(format.AsSpan(12), 2);
        BinaryPrimitives.WriteUInt16LittleEndian(format.AsSpan(14), 16);

        var chunks = new List<byte> { };
        chunks.AddRange(Chunk("fmt ", format));

        if (bext is { } stamp)
        {
            var body = new byte[602];
            Encoding.ASCII.GetBytes(stamp.Date).CopyTo(body, 320);
            Encoding.ASCII.GetBytes(stamp.Time).CopyTo(body, 330);
            chunks.AddRange(Chunk("bext", body));
        }

        chunks.AddRange(Chunk("data", new byte[32_000 * seconds]));

        var riff = new List<byte>();
        riff.AddRange("RIFF"u8.ToArray());
        riff.AddRange(BitConverter.GetBytes(4 + chunks.Count));
        riff.AddRange("WAVE"u8.ToArray());
        riff.AddRange(chunks);
        return [.. riff];
    }

    private static byte[] Id3(int version, params (string Id, string Text)[] frames)
    {
        var body = new List<byte>();

        foreach (var (id, text) in frames)
        {
            var value = new byte[] { 3 }.Concat(Encoding.UTF8.GetBytes(text)).ToArray();
            var header = new byte[10];
            Encoding.ASCII.GetBytes(id).CopyTo(header, 0);

            if (version == 4)
            {
                Synchsafe(value.Length).CopyTo(header, 4);
            }
            else
            {
                BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(4), (uint)value.Length);
            }

            body.AddRange(header);
            body.AddRange(value);
        }

        body.AddRange(new byte[16]);
        return [.. "ID3"u8.ToArray(), (byte)version, 0, 0, .. Synchsafe(body.Count), .. body, .. new byte[64]];
    }

    private static byte[] Synchsafe(int value) =>
        [(byte)((value >> 21) & 0x7F), (byte)((value >> 14) & 0x7F), (byte)((value >> 7) & 0x7F), (byte)(value & 0x7F)];

    private static byte[] Flac(int rate, long samples, params (string Key, string Value)[] comments)
    {
        var info = new byte[34];
        info[10] = (byte)(rate >> 12);
        info[11] = (byte)(rate >> 4);
        info[12] = (byte)((rate & 0x0F) << 4);
        info[13] = (byte)((samples >> 32) & 0x0F);
        BinaryPrimitives.WriteUInt32BigEndian(info.AsSpan(14), (uint)(samples & 0xFFFFFFFF));

        var vorbis = new List<byte>();
        vorbis.AddRange(BitConverter.GetBytes(4));
        vorbis.AddRange("test"u8.ToArray());
        vorbis.AddRange(BitConverter.GetBytes(comments.Length));

        foreach (var (key, value) in comments)
        {
            var line = Encoding.UTF8.GetBytes($"{key}={value}");
            vorbis.AddRange(BitConverter.GetBytes(line.Length));
            vorbis.AddRange(line);
        }

        static byte[] Block(int type, bool last, byte[] body) =>
            [(byte)(type | (last ? 0x80 : 0)), (byte)(body.Length >> 16), (byte)(body.Length >> 8), (byte)body.Length, .. body];

        return [.. "fLaC"u8.ToArray(), .. Block(0, false, info), .. Block(4, true, [.. vorbis]), .. new byte[16]];
    }
}
