using LocalScribe.Core.Alignment;
using Xunit;

namespace LocalScribe.Core.Tests;

public class GlobalCtcAlignmentTests
{
    private const int Blank = 0;
    private const int Alphabet = 4;

    /// <summary>A recording where each stretch of frames strongly favours one token.</summary>
    private static AlignmentScores Recording(params (int Token, int Frames)[] stretches)
    {
        var total = stretches.Sum(s => s.Frames);
        var scores = new AlignmentScores(total, Alphabet, 0.02);
        var window = new float[Alphabet];
        var at = 0;

        foreach (var (token, frames) in stretches)
        {
            for (var f = 0; f < frames; f++)
            {
                for (var k = 0; k < Alphabet; k++)
                {
                    window[k] = k == token ? 0f : -12f;
                }

                // The blank stays plausible during speech, as it does in the real model.
                if (token != Blank)
                {
                    window[Blank] = -4f;
                }

                scores.Fill(at++, window);
            }
        }

        return scores;
    }

    private static int[] Diagonal(int frames, int states) =>
        [.. Enumerable.Range(0, frames).Select(t => (int)((long)t * states / frames))];

    [Fact]
    public void LettersLandOnTheirOwnStretches()
    {
        var scores = Recording((1, 20), (2, 20), (3, 20));
        var targets = new[] { 1, 2, 3 };

        var placed = GlobalCtcAlignment.Align(
            scores, targets, Blank, Diagonal(60, 7), halfBand: 50);

        Assert.NotNull(placed);
        Assert.InRange(placed[0].FirstFrame, 0, 19);
        Assert.InRange(placed[1].FirstFrame, 20, 39);
        Assert.InRange(placed[2].FirstFrame, 40, 59);
    }

    /// <summary>
    /// The failure the whole pass exists to make unrepresentable. A repeated phrase gives a
    /// windowed aligner two plausible locks and it can take the wrong one; a single path over
    /// the whole text must spend the first occurrence before the second.
    /// </summary>
    [Fact]
    public void ARepeatedPhraseKeepsItsOccurrencesInOrder()
    {
        var scores = Recording((1, 10), (2, 10), (Blank, 10), (1, 10), (2, 10));
        var targets = new[] { 1, 2, 1, 2 };

        var placed = GlobalCtcAlignment.Align(
            scores, targets, Blank, Diagonal(50, 9), halfBand: 50);

        Assert.NotNull(placed);
        Assert.InRange(placed[0].LastFrame, 0, 12);
        Assert.InRange(placed[1].LastFrame, 8, 22);
        Assert.InRange(placed[2].FirstFrame, 28, 42);
        Assert.InRange(placed[3].FirstFrame, 38, 50);
        Assert.True(placed[1].LastFrame < placed[2].FirstFrame, "the twins must not swap");
    }

    /// <summary>The corridor only rises, so the stamps can never force the text backwards.</summary>
    [Fact]
    public void ACorridorThatCannotReachTheEndRefusesRatherThanLies()
    {
        var scores = Recording((1, 10), (2, 10));
        var targets = new[] { 1, 2, 3, 1, 2, 3, 1, 2, 3 };

        var placed = GlobalCtcAlignment.Align(
            scores, targets, Blank, [.. Enumerable.Repeat(0, 20)], halfBand: 3);

        Assert.Null(placed);
    }

    [Fact]
    public void MoreTextThanTheRecordingCanCarryIsRefused()
    {
        var scores = Recording((1, 3));

        var placed = GlobalCtcAlignment.Align(
            scores, [1, 2, 3, 1, 2, 3], Blank, Diagonal(3, 13), halfBand: 20);

        Assert.Null(placed);
    }

    [Fact]
    public void SilenceBetweenSpeechBelongsToNobody()
    {
        var scores = Recording((1, 10), (Blank, 20), (2, 10));

        var placed = GlobalCtcAlignment.Align(
            scores, [1, 2], Blank, Diagonal(40, 5), halfBand: 40);

        Assert.NotNull(placed);
        Assert.True(placed[0].LastFrame < 15, "the first letter ends before the silence");
        Assert.True(placed[1].FirstFrame > 25, "the second begins after it");
    }
}
