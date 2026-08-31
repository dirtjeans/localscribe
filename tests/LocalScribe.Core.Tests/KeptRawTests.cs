using LocalScribe.Core.Refinement;
using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class KeptRawTests
{
    private static TranscriptSegment Line(int index, string text) =>
        new(text, index * 5.0, (index * 5.0) + 4.0);

    /// <summary>
    /// The refiner remembers which segments were kept raw, as the exact instances the cleaned
    /// result carries — which is what lets a retry re-clean only the failed passage instead of
    /// the whole transcript.
    /// </summary>
    [Fact]
    public async Task AFailedWindowIsRememberedByInstance()
    {
        // Enough words that the transcript splits into more than one window, with a marker in
        // the late segments the model refuses to answer about.
        var filler = string.Join(' ', Enumerable.Repeat("word", 80));
        var segments = Enumerable.Range(0, 10)
            .Select(i => Line(i, i >= 8 ? $"unlucky {filler}" : filler))
            .ToList();

        var model = new FakeLanguageModel((_, user) =>
            user.Contains("unlucky", StringComparison.Ordinal)
                ? throw new HttpRequestException("timed out")
                : user);

        var refiner = new TranscriptRefiner(model);
        var result = await refiner.RefineAsync(new Transcript(segments));

        Assert.NotEmpty(refiner.KeptRaw);
        Assert.All(refiner.KeptRaw, kept => Assert.Contains("unlucky", kept.Text));

        // By reference, in the cleaned output, so a retry can find them there.
        Assert.All(
            refiner.KeptRaw,
            kept => Assert.Contains(result.CleanedSegments!, s => ReferenceEquals(s, kept)));
    }

    [Fact]
    public async Task ACleanRunRemembersNothing()
    {
        var filler = string.Join(' ', Enumerable.Repeat("word", 60));
        var segments = Enumerable.Range(0, 4).Select(i => Line(i, filler)).ToList();

        var refiner = new TranscriptRefiner(new FakeLanguageModel((_, user) => user));
        await refiner.RefineAsync(new Transcript(segments));

        Assert.Empty(refiner.KeptRaw);
    }
}
