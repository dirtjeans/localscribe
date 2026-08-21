using System.Diagnostics;
using LocalScribe.Core.Audio;
using LocalScribe.Core.Hardware;
using LocalScribe.Core.Pipeline;
using LocalScribe.Core.Refinement;
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
            ?? Core.Models.ModelLayout.Locate(
                modelDirectory, capabilities.Family, plan.Encoder.Device, plan.WhisperModel);

        if (directory is null)
        {
            Console.Error.WriteLine(
                $"No Whisper weights under {modelDirectory}. Run --fetch-models for a portable "
                + "set, or pass --model-dir to point at one.");
            return 1;
        }

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

        // The real pipeline, not a loop of its own. A diagnostic that takes a different path
        // through the code cannot report on the path the app uses, and the difference between
        // the two is exactly where a window-boundary bug hides.
        var pipeline = new TranscriptionPipeline(transcriber);

        var decoding = Stopwatch.StartNew();
        TranscriptionResult result;

        var progress = new Progress<TranscriptionProgress>(update =>
            Console.WriteLine($"  window {update.ChunksCompleted} of about {update.ChunksTotal}: "
                + $"{Trim(update.LatestText)}"));

        try
        {
            result = await pipeline
                .RunAsync(audio, glossary: null, RefinementOutputs.Punctuation, progress)
                .ConfigureAwait(false);
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

        var transcript = result.Transcript;


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

    /// <summary>
    /// Feeds a file through the live session the way the microphone does, then stops it the way
    /// the stop button does. Reproduces the live path exactly, which a batch run does not: the
    /// rolling window, the commit horizon, and the final pass are all specific to it.
    /// </summary>
    public static async Task<int> RunLiveAsync(
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
            ?? Core.Models.ModelLayout.Locate(
                modelDirectory, capabilities.Family, plan.Encoder.Device, plan.WhisperModel);

        if (directory is null)
        {
            Console.Error.WriteLine($"No Whisper weights under {modelDirectory}.");
            return 1;
        }

        var audio = WavReader.Read(audioPath);
        audio.EnsureWhisperFormat();

        Heading("Live");
        Console.WriteLine($"  Audio      {audioPath}");
        Console.WriteLine($"  Models     {directory}");
        Console.WriteLine($"  Duration   {audio.DurationSeconds:F1}s");

        using var transcriber = WhisperOnnxTranscriber.Load(directory, plan);
        Console.WriteLine($"  Signature  {transcriber.Signature.Describe()}");
        Console.WriteLine();

        await using var session = new LiveTranscriptionSession(transcriber);

        // The microphone hands over roughly 200 ms at a time.
        var frame = (int)(0.2 * audio.SampleRate);
        var lastShown = string.Empty;

        for (var offset = 0; offset < audio.Samples.Length; offset += frame)
        {
            var length = Math.Min(frame, audio.Samples.Length - offset);
            var update = await session
                .PushAsync(audio.Samples.AsMemory(offset, length))
                .ConfigureAwait(false);

            if (update is null)
            {
                continue;
            }

            // What the window shows while listening: settled text plus the provisional tail.
            var committed = new Transcript(session.CommittedSegments).FullText;
            var onScreen = string.Join(" ", new[] { committed, update.Text }
                .Where(part => part.Length > 0));

            if (onScreen != lastShown)
            {
                lastShown = onScreen;
                Console.WriteLine($"  [live] {onScreen}");
            }
        }

        Console.WriteLine();
        Console.WriteLine("  --- stop pressed ---");
        Console.WriteLine();

        var final = new Transcript(await session.FinishAsync()).FullText;

        Heading("On screen while listening");
        Console.WriteLine(lastShown.Length == 0 ? "  (nothing)" : lastShown);

        Heading("After stopping");
        Console.WriteLine(final.Length == 0 ? "  (nothing)" : final);

        Heading("Comparison");
        Console.WriteLine($"  While listening  {Punctuation(lastShown)} punctuation marks, {lastShown.Length} chars");
        Console.WriteLine($"  After stopping   {Punctuation(final)} punctuation marks, {final.Length} chars");
        Console.WriteLine(final == lastShown
            ? "  Identical."
            : "  DIFFERENT — the stop changed the text.");

        return 0;
    }

    /// <summary>One line of a window's text, so a long window does not flood the output.</summary>
    private static string Trim(string text)
    {
        var single = text.ReplaceLineEndings(" ").Trim();

        return single.Length <= 96 ? single : single[..96] + "…";
    }

    private static int Punctuation(string text) =>
        text.Count(c => c is '.' or ',' or '?' or '!' or ';' or ':');

    private static void Heading(string title)
    {
        Console.WriteLine();
        Console.WriteLine(title);
        Console.WriteLine(new string('-', title.Length));
    }
}
