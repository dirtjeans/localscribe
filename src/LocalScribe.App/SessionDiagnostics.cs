using System.Text;
using LocalScribe.Core.Audio;
using LocalScribe.Core.Transcription;

namespace LocalScribe.App;

/// <summary>
/// Records what a live session actually did, when asked to.
/// <para>
/// Off unless <c>LOCALSCRIBE_DIAGNOSTICS</c> is set, and deliberately so. This writes the
/// captured audio to disk, and an app whose whole promise is that nothing leaves the machine
/// should not quietly start keeping recordings. Nothing here is uploaded; it is written beside
/// the app for the user to read or delete.
/// </para>
/// <para>
/// It exists because the interesting live bugs are the ones where the text on screen while
/// listening differs from the text after stopping. That difference is invisible afterwards —
/// the provisional text is gone by then — so it has to be captured as it happens.
/// </para>
/// </summary>
internal sealed class SessionDiagnostics
{
    private readonly string _directory;
    private readonly StringBuilder _log = new();
    private readonly List<float> _captured = [];
    private readonly int _sampleRate;

    private SessionDiagnostics(string directory, int sampleRate)
    {
        _directory = directory;
        _sampleRate = sampleRate;
    }

    /// <summary>Where the files went, for showing the user.</summary>
    public string Directory => _directory;

    /// <summary>Returns a recorder when diagnostics are switched on, and null otherwise.</summary>
    public static SessionDiagnostics? StartIfEnabled(
        string plan,
        string transcriber,
        int sampleRate = PcmAudio.WhisperSampleRate)
    {
        var enabled = Environment.GetEnvironmentVariable("LOCALSCRIBE_DIAGNOSTICS");
        if (string.IsNullOrEmpty(enabled) || enabled == "0")
        {
            return null;
        }

        var root = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "LocalScribe",
            "diagnostics",
            DateTime.Now.ToString("yyyy-MM-dd-HHmmss"));

        System.IO.Directory.CreateDirectory(root);

        var diagnostics = new SessionDiagnostics(root, sampleRate);
        diagnostics._log.AppendLine($"plan:        {plan}");
        diagnostics._log.AppendLine($"transcriber: {transcriber}");
        diagnostics._log.AppendLine();

        return diagnostics;
    }

    public void Captured(ReadOnlySpan<float> samples) => _captured.AddRange(samples);

    /// <summary>
    /// One pass's result. Both halves are recorded: what is settled and what is still moving.
    /// The bug being chased is a disagreement between them.
    /// </summary>
    public void Pass(string provisional, IReadOnlyList<TranscriptSegment> committed)
    {
        _log.AppendLine($"[pass at {Seconds():F1}s]");
        _log.AppendLine($"  provisional: {provisional}");
        _log.AppendLine($"  committed  : {Join(committed)}");

        foreach (var segment in committed)
        {
            _log.AppendLine($"    {segment.StartSeconds,6:F2}-{segment.EndSeconds,6:F2}  {segment.Text}");
        }
    }

    public void Finished(IReadOnlyList<TranscriptSegment> committed)
    {
        _log.AppendLine();
        _log.AppendLine($"[stopped at {Seconds():F1}s]");
        _log.AppendLine($"  final: {Join(committed)}");

        foreach (var segment in committed)
        {
            _log.AppendLine($"    {segment.StartSeconds,6:F2}-{segment.EndSeconds,6:F2}  {segment.Text}");
        }

        File.WriteAllText(Path.Combine(_directory, "session.log"), _log.ToString());
        WriteWav(Path.Combine(_directory, "capture.wav"));
    }

    private double Seconds() => _captured.Count / (double)_sampleRate;

    private static string Join(IReadOnlyList<TranscriptSegment> segments) =>
        string.Join(" ", segments.Select(s => s.Text.Trim()).Where(t => t.Length > 0));

    /// <summary>
    /// The exact audio the session was fed, so the same recording can be replayed through
    /// <c>localscribe-doctor --transcribe-live</c> and the result compared.
    /// </summary>
    private void WriteWav(string path)
    {
        var samples = _captured;
        var dataBytes = samples.Count * 2;

        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        writer.Write("RIFF"u8);
        writer.Write(36 + dataBytes);
        writer.Write("WAVE"u8);
        writer.Write("fmt "u8);
        writer.Write(16);                        // PCM header length
        writer.Write((short)1);                  // PCM
        writer.Write((short)1);                  // mono
        writer.Write(_sampleRate);
        writer.Write(_sampleRate * 2);           // byte rate
        writer.Write((short)2);                  // block align
        writer.Write((short)16);                 // bits per sample
        writer.Write("data"u8);
        writer.Write(dataBytes);

        foreach (var sample in samples)
        {
            writer.Write((short)Math.Clamp(sample * 32767f, short.MinValue, short.MaxValue));
        }
    }
}
