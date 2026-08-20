using LocalScribe.Core.Transcription;
using Xunit;

namespace LocalScribe.Core.Tests;

public class WhisperModelSignatureTests
{
    /// <summary>
    /// The shapes below are not invented. They were read out of the published
    /// whisper-large-v3-turbo precompiled QNN build for Snapdragon X Elite, which is the export
    /// this contract exists to support.
    /// </summary>
    private static class AiHub
    {
        public const int Layers = 4;
        public const int Heads = 20;
        public const int HeadDim = 64;
        public const int Window = 200;
        public const int Vocabulary = 51866;

        public static List<TensorSpec> EncoderInputs() =>
            [new("input_features", [1, 128, 3000])];

        public static List<TensorSpec> EncoderOutputs()
        {
            var specs = new List<TensorSpec>();
            for (var layer = 0; layer < Layers; layer++)
            {
                specs.Add(new TensorSpec($"k_cache_cross_{layer}", [Heads, 1, HeadDim, 1500]));
                specs.Add(new TensorSpec($"v_cache_cross_{layer}", [Heads, 1, 1500, HeadDim]));
            }

            return specs;
        }

        public static List<TensorSpec> DecoderInputs()
        {
            var specs = new List<TensorSpec>
            {
                new("input_ids", [1, 1]),
                new("attention_mask", [1, 1, 1, Window]),
            };

            for (var layer = 0; layer < Layers; layer++)
            {
                specs.Add(new TensorSpec($"k_cache_self_{layer}_in", [Heads, 1, HeadDim, Window - 1]));
                specs.Add(new TensorSpec($"v_cache_self_{layer}_in", [Heads, 1, Window - 1, HeadDim]));
            }

            specs.AddRange(EncoderOutputs());
            specs.Add(new TensorSpec("position_ids", [1]));

            return specs;
        }

        public static List<TensorSpec> DecoderOutputs()
        {
            var specs = new List<TensorSpec> { new("logits", [1, Vocabulary, 1, 1]) };

            for (var layer = 0; layer < Layers; layer++)
            {
                specs.Add(new TensorSpec($"k_cache_self_{layer}_out", [Heads, 1, HeadDim, Window - 1]));
                specs.Add(new TensorSpec($"v_cache_self_{layer}_out", [Heads, 1, Window - 1, HeadDim]));
            }

            return specs;
        }
    }

    private static class Optimum
    {
        // Verbatim from onnx-community/whisper-*.en: Optimum leaves every dimension dynamic,
        // including the band count. The first version of this fixture claimed [-1, 80, 3000] and
        // the real export failed to load against it.
        public static List<TensorSpec> EncoderInputs() =>
            [new("input_features", [-1, -1, -1])];

        public static List<TensorSpec> EncoderOutputs() =>
            [new("last_hidden_state", [-1, 1500, 512])];

        public static List<TensorSpec> DecoderInputs() =>
        [
            new("input_ids", [-1, -1]),
            new("encoder_hidden_states", [-1, 1500, 512]),
        ];

        public static List<TensorSpec> DecoderOutputs() =>
            [new("logits", [-1, -1, 51864])];
    }

    [Fact]
    public void TheAiHubExportIsRecognisedAsCached()
    {
        var signature = WhisperModelSignature.Detect(
            AiHub.EncoderInputs(), AiHub.EncoderOutputs(),
            AiHub.DecoderInputs(), AiHub.DecoderOutputs());

        Assert.Equal(WhisperDecoderContract.QnnCached, signature.Contract);
        Assert.Equal(AiHub.Layers, signature.DecoderLayers);
        Assert.Equal(AiHub.Window, signature.MaxDecodeLength);
        Assert.Equal(AiHub.Vocabulary, signature.VocabularySize);
    }

    [Fact]
    public void TheOptimumExportIsRecognisedAsPortable()
    {
        var signature = WhisperModelSignature.Detect(
            Optimum.EncoderInputs(), Optimum.EncoderOutputs(),
            Optimum.DecoderInputs(), Optimum.DecoderOutputs());

        Assert.Equal(WhisperDecoderContract.Portable, signature.Contract);
        Assert.Equal(0, signature.DecoderLayers);
        Assert.Equal(51864, signature.VocabularySize);
    }

    /// <summary>
    /// The band count is the difference between a transcript and fluent nonsense, and Whisper
    /// itself is inconsistent about it. It has to come from the model, never from a default.
    /// </summary>
    [Theory]
    [InlineData(80)]
    [InlineData(128)]
    public void ACachedEncoderStatesItsBandCountAndItIsUsed(int bands)
    {
        var signature = WhisperModelSignature.Detect(
            [new TensorSpec("input_features", [1, bands, 3000])],
            AiHub.EncoderOutputs(), AiHub.DecoderInputs(), AiHub.DecoderOutputs());

        Assert.Equal(bands, signature.MelBands);
    }

    /// <summary>
    /// Cached logits are (1, vocabulary, 1, 1), so taking the last dimension — which is right
    /// for the portable contract — would yield 1 and make every argmax return token zero.
    /// </summary>
    [Fact]
    public void CachedLogitsAreReadOnTheRightAxis()
    {
        var signature = WhisperModelSignature.Detect(
            AiHub.EncoderInputs(), AiHub.EncoderOutputs(),
            AiHub.DecoderInputs(), AiHub.DecoderOutputs());

        Assert.Equal(AiHub.Vocabulary, signature.VocabularySize);
        Assert.NotEqual(1, signature.VocabularySize);
    }

    /// <summary>
    /// Two exports from different runs would otherwise fail deep inside the first decode step,
    /// with nothing to point at.
    /// </summary>
    [Fact]
    public void MismatchedHalvesAreRejectedUpFront()
    {
        var encoderOutputs = AiHub.EncoderOutputs().Take(4).ToList(); // two layers, not four

        var exception = Assert.Throws<InvalidOperationException>(() =>
            WhisperModelSignature.Detect(
                AiHub.EncoderInputs(), encoderOutputs,
                AiHub.DecoderInputs(), AiHub.DecoderOutputs()));

        Assert.Contains("different models", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACachedDecoderWithoutAnAttentionMaskIsRejected()
    {
        var inputs = AiHub.DecoderInputs().Where(s => s.Name != "attention_mask").ToList();

        Assert.Throws<InvalidOperationException>(() =>
            WhisperModelSignature.Detect(
                AiHub.EncoderInputs(), AiHub.EncoderOutputs(), inputs, AiHub.DecoderOutputs()));
    }

    /// <summary>
    /// A QNN graph is compiled for one fixed input shape, so it always states the band count.
    /// A dynamic one means the export is not what it claims, and guessing would be worse than
    /// refusing.
    /// </summary>
    [Fact]
    public void ACachedEncoderWithADynamicBandCountIsRejected()
    {
        Assert.Throws<InvalidOperationException>(() =>
            WhisperModelSignature.Detect(
                [new TensorSpec("input_features", [-1, -1, -1])],
                AiHub.EncoderOutputs(), AiHub.DecoderInputs(), AiHub.DecoderOutputs()));
    }

    /// <summary>
    /// Portable exports state nothing, so the band count falls back to the vocabulary. The 128
    /// band filterbank and the 51866th token arrived together in large-v3.
    /// </summary>
    [Theory]
    [InlineData(51864, 80)]
    [InlineData(51865, 80)]
    [InlineData(51866, 128)]
    public void ADynamicPortableEncoderInfersBandsFromTheVocabulary(int vocabulary, int expectedBands)
    {
        var signature = WhisperModelSignature.Detect(
            Optimum.EncoderInputs(), Optimum.EncoderOutputs(), Optimum.DecoderInputs(),
            [new TensorSpec("logits", [-1, -1, vocabulary])]);

        Assert.Equal(expectedBands, signature.MelBands);
    }

    [Fact]
    public void DescribeNamesTheContractSoAMismatchIsVisible()
    {
        var cached = WhisperModelSignature.Detect(
            AiHub.EncoderInputs(), AiHub.EncoderOutputs(),
            AiHub.DecoderInputs(), AiHub.DecoderOutputs());

        var portable = WhisperModelSignature.Detect(
            Optimum.EncoderInputs(), Optimum.EncoderOutputs(),
            Optimum.DecoderInputs(), Optimum.DecoderOutputs());

        Assert.Contains("cached QNN", cached.Describe(), StringComparison.Ordinal);
        Assert.Contains("128 mel", cached.Describe(), StringComparison.Ordinal);
        Assert.Contains("portable", portable.Describe(), StringComparison.Ordinal);
    }
}
