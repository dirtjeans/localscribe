namespace LocalScribe.Core.Transcription;

/// <summary>What to do with speech that is not in English.</summary>
public enum SpeechTask
{
    /// <summary>
    /// Write down what was said, in the language it was said in.
    /// <para>
    /// The default, and the only one anybody should get without asking. A transcript that
    /// silently changes language is not a transcript of the recording.
    /// </para>
    /// </summary>
    Transcribe,

    /// <summary>
    /// Write it down in English, whatever was spoken.
    /// <para>
    /// Whisper does this natively rather than by translating afterwards: the same decode that
    /// recognises the speech emits English. It is one-way and English-only — the model has no
    /// task for any other target — so this is a convenience, not a translation feature.
    /// </para>
    /// </summary>
    TranslateToEnglish,
}
