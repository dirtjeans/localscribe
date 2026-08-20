using System.Diagnostics;
using LocalScribe.Core.Audio;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Transcription;
using LocalScribe.Onnx;

namespace LocalScribe.Doctor;

/// <summary>
/// Implements <c>--transcribe</c>: runs one real file through the real pipeline and reports what
/// happened.
/// <para>
/// This exists because every other check in the doctor is a check on prerequisites, and
/// prerequisites being present is not the same as the thing working. The setup notes already ask
/// the reader to watch the NPU graph in Task Manager during a transcription; without this they
/// have no way to start one.
/// </para>
/// <para>
/// It prints the discovered model signature before doing any work, so an export whose contract
/// does not match what was expected is visible before the first tensor is bound rather than
/// after a wall of nonsense.
/// </para>
/// </summary>
internal static class TranscribeCommand
{
    public static async Task<int> RunAsync(
        string audioPath,
        string modelDirectory,
        DeviceCapabilities capabilities,
        ExecutionPlan plan,
        string? explicitModelDirectory)
    {
        if (!File.Exists(audioPath))
        {
            Console.Error.WriteLine($"No such file: {audioPath}");
            return 1;
        }

        var directory = explicitModelDirectory
            ?? Core.Models.ModelLayout.Resolve(
                modelDirectory, capabilities.Family, plan.Encoder.Device, plan.WhisperModel);

        Heading("Transcribe");
        Console.WriteLine($"  Audio      {audioPath}");
        Console.WriteLine($"  Models     {directory}");

        PcmAudio audio;
        try
        {
            audio = WavReader.Read(audioPath);
        }
        catch (Exception exception) when (exception is InvalidDataException or NotSupportedException)
        {
            Console.Error.WriteLine($"Could not read the audio: {exception.Message}");
            return 1;
        }

        Console.WriteLine($"  Duration   {audio.DurationSeconds:F1}s at {audio.SampleRate} Hz");

        try
        {
            audio.EnsureWhisperFormat();
        }
        catch (InvalidOperationException exception)
        {
            Console.Error.WriteLine($"  {exception.Message}");
            return 1;
        }

        var loading = Stopwatch.StartNew();
        WhisperOnnxTranscriber transcriber;

        try
        {
            transcriber = WhisperOnnxTranscriber.Load(directory, plan);
        }
        catch (Exception exception)
        {
            loading.Stop();
            Console.Error.WriteLine();
            Console.Error.WriteLine($"  Loading failed after {loading.Elapsed.TotalSeconds:F1}s:");
            Console.Error.WriteLine($"  {exception.GetType().Name}: {exception.Message}");
            Explain(exception);
            return 1;
        }

        loading.Stop();
        using var owned = transcriber;

        Console.WriteLine($"  Signature  {transcriber.Signature.Describe()}");
        Console.WriteLine($"  Loaded in  {loading.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine();

        var chunker = new AudioChunker();
        var chunks = chunker.Chunk(audio);
        var results = new List<IReadOnlyList<TranscriptSegment>>();

        var decoding = Stopwatch.StartNew();

        try
        {
            foreach (var chunk in chunks)
            {
                var segments = await transcriber.TranscribeChunkAsync(chunk).ConfigureAwait(false);
                results.Add(segments);
                Console.WriteLine($"  chunk at {chunk.StartSeconds,5:F1}s -> {segments.Count} segment(s)");
            }
        }
        catch (Exception exception)
        {
            decoding.Stop();
            Console.Error.WriteLine();
            Console.Error.WriteLine($"  Decoding failed: {exception.GetType().Name}: {exception.Message}");
            Explain(exception);
            return 1;
        }

        decoding.Stop();

        var transcript = new Transcript(new TranscriptStitcher().Stitch(results));

        Heading("Transcript");
        Console.WriteLine(transcript.FullText.Length == 0 ? "  (empty)" : transcript.FullText);

        Heading("Timing");
        var speed = audio.DurationSeconds / Math.Max(0.001, decoding.Elapsed.TotalSeconds);
        Console.WriteLine($"  Decoded {audio.DurationSeconds:F1}s of audio in "
            + $"{decoding.Elapsed.TotalSeconds:F1}s ({speed:F1}x real time)");
        Console.WriteLine($"  Encoder ran on {plan.Encoder.Device}, decoder on {plan.Decoder.Device}");

        return transcript.FullText.Length == 0 ? 1 : 0;
    }

    /// <summary>
    /// Turns the failures that actually happen into the thing to go and check. ONNX Runtime's own
    /// messages are accurate and unhelpful in equal measure.
    /// </summary>
    private static void Explain(Exception exception)
    {
        var message = exception.Message;

        if (message.Contains("context binary", StringComparison.OrdinalIgnoreCase)
            || message.Contains("context from binary", StringComparison.OrdinalIgnoreCase)
            || message.Contains("EpContext", StringComparison.OrdinalIgnoreCase)
            || message.Contains("ep_cache_context", StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine();
            Console.Error.WriteLine(
                "  A context-binary error almost always means a version mismatch, and the "
                + "message names neither version. Precompiled QNN binaries are tied to the ONNX "
                + "Runtime and QAIRT they were built against; the model card lists what it "
                + "wants. Check it against the Microsoft.ML.OnnxRuntime.QNN version pinned in "
                + "LocalScribe.Onnx.csproj.");
            return;
        }

        if (message.Contains("model.bin", StringComparison.OrdinalIgnoreCase)
            || message.Contains("No such file", StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine();
            Console.Error.WriteLine(
                "  The wrapper references its context binary as './model.bin', so the two files "
                + "must sit in the same directory with those exact names.");
            return;
        }

        if (message.Contains("Invalid rank", StringComparison.OrdinalIgnoreCase)
            || message.Contains("Invalid Feed", StringComparison.OrdinalIgnoreCase)
            || message.Contains("dimensions", StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine();
            Console.Error.WriteLine(
                "  A shape mismatch here means the export's contract differs from the one "
                + "detected. The signature printed above is what the binding used.");
        }
    }

    private static void Heading(string title)
    {
        Console.WriteLine();
        Console.WriteLine(title);
        Console.WriteLine(new string('-', title.Length));
    }
}
