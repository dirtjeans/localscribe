using System.Runtime.InteropServices;

namespace LocalScribe.Core.Diarization;

/// <summary>What works out who spoke when.</summary>
public enum SpeakerEngine
{
    /// <summary>
    /// NVIDIA's Nemotron 3 Diarization: one end-to-end model, eight speakers at most.
    /// </summary>
    Sortformer,

    /// <summary>
    /// pyannote segmentation, WeSpeaker voices and clustering, chained. The tuned and frozen
    /// pipeline the app shipped with.
    /// </summary>
    Pyannote,
}

/// <summary>Which engine a machine uses.</summary>
public static class SpeakerEngines
{
    /// <summary>
    /// Nemotron on ARM64, pyannote elsewhere.
    /// <para>
    /// Measured on the Snapdragon against the pipeline it replaces, over five recordings from two
    /// minutes to an hour: the same speakers wherever the pipeline was already right — two in
    /// the debate, five in each podcast, agreeing on 97 to 100 percent of single-speaker time —
    /// and three or six where it found sixteen and twenty-five. Four to eleven times faster at
    /// every thread budget tried. ARM64 is where that measurement was made; other machines keep
    /// the pipeline until it is made there.
    /// </para>
    /// </summary>
    public static SpeakerEngine For(Architecture architecture) =>
        architecture == Architecture.Arm64 ? SpeakerEngine.Sortformer : SpeakerEngine.Pyannote;

    /// <summary>The engine for the process this is running in.</summary>
    public static SpeakerEngine Current => For(RuntimeInformation.ProcessArchitecture);

    /// <summary>
    /// The most people an engine can tell apart. Nemotron has eight output channels and no way
    /// to add a ninth; the pipeline clusters, so its limit is what the app offers.
    /// </summary>
    public static int MostSpeakers(SpeakerEngine engine) =>
        engine == SpeakerEngine.Sortformer ? 8 : 10;
}
