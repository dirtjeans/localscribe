using LocalScribe.Core.Audio;
using LocalScribe.Core.Diarization;
using Xunit;

namespace LocalScribe.Core.Tests;

public class PowersetDecoderTests
{
    /// <summary>
    /// The published model's shape: three local speakers, at most two talking at once, seven
    /// classes. The mapping is not stored in the model, so it has to be built in the order
    /// pyannote builds it — silence, each speaker alone, then each pair.
    /// </summary>
    [Fact]
    public void TheMappingMatchesThePublishedModel()
    {
        var mapping = PowersetDecoder.Mapping(speakers: 3, maxOverlap: 2);

        Assert.Equal(7, mapping.Count);
        Assert.Empty(mapping[0]);
        Assert.Equal([0], mapping[1]);
        Assert.Equal([1], mapping[2]);
        Assert.Equal([2], mapping[3]);
        Assert.Equal([0, 1], mapping[4]);
        Assert.Equal([0, 2], mapping[5]);
        Assert.Equal([1, 2], mapping[6]);
    }

    [Fact]
    public void SilenceActivatesNobody()
    {
        var mapping = PowersetDecoder.Mapping(3, 2);
        float[] scores = [9, 0, 0, 0, 0, 0, 0];

        var active = PowersetDecoder.Decode(scores, frames: 1, mapping, speakers: 3);

        Assert.All(active, Assert.False);
    }

    [Fact]
    public void ASingleSpeakerClassActivatesOnlyThatSpeaker()
    {
        var mapping = PowersetDecoder.Mapping(3, 2);
        float[] scores = [0, 0, 9, 0, 0, 0, 0];   // class 2 is speaker 1 alone

        var active = PowersetDecoder.Decode(scores, frames: 1, mapping, speakers: 3);

        Assert.False(active[0]);
        Assert.True(active[1]);
        Assert.False(active[2]);
    }

    /// <summary>
    /// The whole reason for a powerset: one argmax can say two people are talking, where a
    /// per-speaker score would have to pick a winner.
    /// </summary>
    [Fact]
    public void AnOverlapClassActivatesBothSpeakers()
    {
        var mapping = PowersetDecoder.Mapping(3, 2);
        float[] scores = [0, 0, 0, 0, 0, 9, 0];   // class 5 is speakers 0 and 2

        var active = PowersetDecoder.Decode(scores, frames: 1, mapping, speakers: 3);

        Assert.True(active[0]);
        Assert.False(active[1]);
        Assert.True(active[2]);
    }

    [Fact]
    public void EachFrameIsDecodedIndependently()
    {
        var mapping = PowersetDecoder.Mapping(3, 2);
        float[] scores =
        [
            9, 0, 0, 0, 0, 0, 0,    // silence
            0, 9, 0, 0, 0, 0, 0,    // speaker 0
        ];

        var active = PowersetDecoder.Decode(scores, frames: 2, mapping, speakers: 3);

        Assert.False(active[0]);
        Assert.True(active[3]);   // frame 1, speaker 0
    }
}

public class SpeakerClusteringTests
{
    private static float[] Vector(params float[] values) => values;

    [Fact]
    public void IdenticalVectorsAreTheSameSpeaker()
    {
        var labels = SpeakerClustering.Cluster(
            [Vector(1, 0, 0), Vector(1, 0, 0), Vector(1, 0, 0)]);

        Assert.Equal([0, 0, 0], labels);
    }

    [Fact]
    public void OrthogonalVectorsAreDifferentSpeakers()
    {
        var labels = SpeakerClustering.Cluster(
            [Vector(1, 0, 0), Vector(0, 1, 0)], threshold: 0.5);

        Assert.NotEqual(labels[0], labels[1]);
    }

    /// <summary>Speaker 1 should be whoever spoke first, not whichever cluster formed first.</summary>
    [Fact]
    public void SpeakersAreNumberedByFirstAppearance()
    {
        var labels = SpeakerClustering.Cluster(
            [Vector(0, 1, 0), Vector(1, 0, 0), Vector(0, 1, 0)], threshold: 0.5);

        Assert.Equal(0, labels[0]);
        Assert.Equal(1, labels[1]);
        Assert.Equal(0, labels[2]);
    }

    /// <summary>
    /// A cap has to win over the threshold, because "there are two people in this recording" is
    /// something the user often knows and the threshold never does.
    /// </summary>
    [Fact]
    public void AMaximumForcesMergingPastTheThreshold()
    {
        var labels = SpeakerClustering.Cluster(
            [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)],
            threshold: 0.1,
            maxSpeakers: 2);

        Assert.Equal(2, labels.Distinct().Count());
    }

    [Fact]
    public void NothingInNothingOut() => Assert.Empty(SpeakerClustering.Cluster([]));

    [Fact]
    public void OneEmbeddingIsOneSpeaker() =>
        Assert.Equal([0], SpeakerClustering.Cluster([Vector(1, 2, 3)]));

    [Fact]
    public void MagnitudeDoesNotAffectDistance()
    {
        var labels = SpeakerClustering.Cluster(
            [Vector(1, 0, 0), Vector(100, 0, 0)], threshold: 0.01);

        Assert.Equal(labels[0], labels[1]);
    }
}

public class SpeakerTurnTests
{
    [Fact]
    public void OverlapIsTheSharedSeconds()
    {
        var turn = new SpeakerTurn(0, 10, 20);

        Assert.Equal(5, turn.OverlapWith(15, 25));
        Assert.Equal(10, turn.OverlapWith(5, 25));
        Assert.Equal(0, turn.OverlapWith(25, 30));
    }

    [Fact]
    public void LabelsAreOneBased() =>
        Assert.Equal("Speaker 1", new SpeakerTurn(0, 0, 1).Label);
}

public class KaldiFbankTests
{
    private static float[] Tone(double hertz, double seconds, double amplitude = 0.3)
    {
        var samples = new float[(int)(seconds * PcmAudio.WhisperSampleRate)];

        for (var i = 0; i < samples.Length; i++)
        {
            samples[i] = (float)(amplitude * Math.Sin(2 * Math.PI * hertz * i / PcmAudio.WhisperSampleRate));
        }

        return samples;
    }

    [Fact]
    public void FramesFollowKaldisTwentyFiveOverTenMilliseconds()
    {
        var features = new KaldiFbank().Compute(Tone(440, 1.0));

        // 16000 samples: (16000 - 400) / 160 + 1 = 98 frames.
        Assert.Equal(98 * KaldiFbank.DefaultMelBins, features.Length);
    }

    [Fact]
    public void AudioShorterThanOneWindowGivesNoFrames() =>
        Assert.Empty(new KaldiFbank().Compute(Tone(440, 0.01)));

    /// <summary>
    /// Different pitches must land in different bands. A filterbank that collapsed would return
    /// the same features for everything, which is exactly what a broken one looks like.
    /// </summary>
    [Fact]
    public void DifferentTonesProduceDifferentFeatures()
    {
        var fbank = new KaldiFbank();

        var low = fbank.Compute(Tone(300, 0.5));
        var high = fbank.Compute(Tone(3000, 0.5));

        var difference = low.Zip(high, (a, b) => Math.Abs(a - b)).Max();

        Assert.True(difference > 1.0, $"Features barely differ between 300 Hz and 3 kHz: {difference}");
    }

    /// <summary>
    /// Mean normalisation is a property of the model, not of the features. The WeSpeaker
    /// embedding models carry no feature_normalize_type, meaning none — and applying it anyway
    /// flattened every embedding until two obviously different voices sat 0.03 apart.
    /// </summary>
    [Fact]
    public void MeanNormalisationIsOffUnlessAskedFor()
    {
        var fbank = new KaldiFbank();
        var audio = Tone(440, 0.5);

        var raw = fbank.Compute(audio);
        var normalised = fbank.Compute(audio, subtractMean: true);

        Assert.NotEqual(raw[0], normalised[0]);

        // Normalised features average to zero per band; raw ones do not.
        var bandMean = 0.0;
        var frames = normalised.Length / fbank.MelBins;
        for (var f = 0; f < frames; f++)
        {
            bandMean += normalised[f * fbank.MelBins];
        }

        Assert.True(Math.Abs(bandMean / frames) < 1e-3);
    }
}
