using LocalScribe.Core.Diarization;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// The memory a Sortformer diarizer is shown ahead of each chunk. These check what the memory
/// is for — every speaker stays findable, in the order they arrived, by their clearest frames —
/// rather than which frames one particular run happened to keep.
/// </summary>
public class ArrivalOrderSpeakerCacheTests
{
    private const int Chunk = 340;
    private const int Lookahead = 40;
    private const int CacheFrames = 264;
    private const int QueueFrames = 40;

    private static readonly float[] Silence = [-1f];

    /// <summary>A frame that carries nothing but its own position, so it can be recognised later.</summary>
    private static List<float[]> Frames(int count) =>
        [.. Enumerable.Range(0, count).Select(i => new float[] { i })];

    private static float[] Heard(params (int Speaker, float Probability)[] talking)
    {
        var row = new float[ArrivalOrderSpeakerCache.Speakers];

        foreach (var (speaker, probability) in talking)
        {
            row[speaker] = probability;
        }

        return row;
    }

    /// <summary>The cache part of the context, cut into the blocks the silence frames close.</summary>
    private static List<List<int>> Blocks(IReadOnlyList<float[]> context)
    {
        var blocks = new List<List<int>> { new() };

        foreach (var frame in context.Take(CacheFrames))
        {
            if (ReferenceEquals(frame, Silence))
            {
                blocks.Add([]);
            }
            else
            {
                blocks[^1].Add((int)frame[0]);
            }
        }

        return blocks;
    }

    [Fact]
    public void NothingIsRememberedBeforeAnythingIsHeard()
    {
        Assert.Empty(new ArrivalOrderSpeakerCache(Silence).Context());
    }

    [Fact]
    public void AShortRecordingIsRememberedWhole()
    {
        var cache = new ArrivalOrderSpeakerCache(Silence);
        var input = Frames(30);

        cache.Update(input, [.. input.Select(_ => Heard((0, 0.9f)))], 30);

        Assert.Equal(input, cache.Context());
    }

    [Fact]
    public void LookaheadNeverEntersTheMemory()
    {
        var cache = new ArrivalOrderSpeakerCache(Silence);
        var input = Frames(Chunk + Lookahead);

        cache.Update(input, [.. input.Select(_ => Heard((0, 0.9f)))], Chunk);

        // The lookahead will be heard again as the next chunk, with more to go on. Remembering
        // it now would put the same audio in front of the model twice.
        Assert.DoesNotContain(cache.Context(), frame => !ReferenceEquals(frame, Silence) && frame[0] >= Chunk);
    }

    [Fact]
    public void TheMemoryIsBoundedHoweverLongTheRecording()
    {
        var cache = new ArrivalOrderSpeakerCache(Silence);

        for (var step = 0; step < 5; step++)
        {
            var context = cache.Context();
            List<float[]> input = [.. context, .. Frames(Chunk + Lookahead)];

            cache.Update(input, [.. input.Select(_ => Heard((step % 2, 0.9f)))], Chunk);

            Assert.True(cache.Context().Count <= CacheFrames + QueueFrames);
        }

        Assert.Equal(CacheFrames + QueueFrames, cache.Context().Count);
    }

    [Fact]
    public void TheQueueIsTheMostRecentAudioInOrder()
    {
        var cache = new ArrivalOrderSpeakerCache(Silence);
        var input = Frames(Chunk + Lookahead);

        cache.Update(input, [.. input.Select(_ => Heard((0, 0.9f)))], Chunk);

        var queue = cache.Context().Skip(CacheFrames).Select(frame => (int)frame[0]);

        Assert.Equal(Enumerable.Range(Chunk - QueueFrames, QueueFrames), queue);
    }

    [Fact]
    public void SpeakersAreLaidOutInTheOrderTheyArrived()
    {
        var cache = new ArrivalOrderSpeakerCache(Silence);
        var input = Frames(Chunk + Lookahead);

        // Speaker 0 for the first half, speaker 1 for the second.
        var heard = input.Select((_, i) => i < 170 ? Heard((0, 0.9f)) : Heard((1, 0.9f))).ToList();

        cache.Update(input, heard, Chunk);

        var blocks = Blocks(cache.Context());

        // One block per channel, each closed by silence, whether or not anyone used it.
        Assert.Equal(ArrivalOrderSpeakerCache.Speakers + 1, blocks.Count);

        Assert.NotEmpty(blocks[0]);
        Assert.NotEmpty(blocks[1]);
        Assert.All(blocks[0], frame => Assert.True(frame < 170));
        Assert.All(blocks[1], frame => Assert.True(frame >= 170));
        Assert.All(blocks.Skip(2), Assert.Empty);
    }

    [Fact]
    public void AQuietSpeakerIsNotForgotten()
    {
        var cache = new ArrivalOrderSpeakerCache(Silence);
        var input = Frames(Chunk + Lookahead);

        // One person talks confidently for nearly the whole chunk; another gets ten hesitant
        // frames in. Ranked on score alone those ten are the worst frames in the chunk and the
        // first to go — and with them the only record that a second person exists.
        var heard = input
            .Select((_, i) => i is >= 100 and < 110 ? Heard((1, 0.6f)) : Heard((0, 0.99f)))
            .ToList();

        cache.Update(input, heard, Chunk);

        Assert.Equal(Enumerable.Range(100, 10), Blocks(cache.Context())[1]);
    }

    [Fact]
    public void ASpeakerWithCleanFramesIsNotRememberedByMuddyOnes()
    {
        var cache = new ArrivalOrderSpeakerCache(Silence);
        var input = Frames(Chunk + Lookahead);

        // Speaker 0 alone for a hundred frames, then talked over by speaker 1 for the rest.
        var heard = input
            .Select((_, i) => i < 100 ? Heard((0, 0.9f)) : Heard((0, 0.9f), (1, 0.9f)))
            .ToList();

        cache.Update(input, heard, Chunk);

        var blocks = Blocks(cache.Context());

        // Speaker 0 has plenty of clean examples, so the crosstalk is no use as a picture of
        // them. Speaker 1 has nothing else, and keeps what there is.
        Assert.Equal(Enumerable.Range(0, 100), blocks[0]);
        Assert.NotEmpty(blocks[1]);
        Assert.All(blocks[1], frame => Assert.True(frame >= 100));
    }
}
