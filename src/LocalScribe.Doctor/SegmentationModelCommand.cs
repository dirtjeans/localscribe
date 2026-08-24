namespace LocalScribe.Doctor;

/// <summary>
/// Switches which segmentation model decides where the speaker changes.
/// <para>
/// Four embedding models with a wide spread of published accuracy separated this app's test
/// recording about equally well, which says the limit is not in telling voices apart. It is in
/// where the turns are drawn — and that is this model's job. A boundary the segmenter misses is
/// one no embedding model is ever asked about.
/// </para>
/// <para>
/// Switching is possible without code because the decoder is built from what the model declares:
/// how many speakers a window may hold, and how many of them may overlap. Any model carrying
/// those two numbers and the same output shape can be dropped in.
/// </para>
/// </summary>
public static class SegmentationModelCommand
{
    private const string Release =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models";

    private const string Kept = "segmentations";

    private const string ActiveFile = "active-segmentation.txt";

    /// <param name="Inside">
    /// Which file to take out of the archive. Always the full-precision one: the quantised builds
    /// here use <c>ConvInteger</c>, which ONNX Runtime has no ARM64 kernel for, so they fail at
    /// load rather than run slowly. The aligner taught this lesson once already.
    /// </param>
    private static readonly (string Name, string Archive, string Inside, string Note)[] Known =
    [
        ("pyannote", "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2", "segmentation.onnx",
            "6 MB. The default. Trained on academic meeting corpora."),
        ("reverb-v1", "sherpa-onnx-reverb-diarization-v1.tar.bz2", "model.onnx",
            "9 MB. Rev.com's, trained on their transcription corpus."),
        ("reverb-v2", "sherpa-onnx-reverb-diarization-v2.tar.bz2", "model.onnx",
            "254 MB download, larger model. Rev's better one."),
    ];

    public static async Task<int> RunAsync(string modelRoot, string? requested)
    {
        var directory = Path.Combine(modelRoot, "diarization");

        if (!Directory.Exists(directory))
        {
            Console.Error.WriteLine($"No speaker models under {directory}. Run --fetch-models first.");
            return 1;
        }

        if (string.IsNullOrWhiteSpace(requested))
        {
            Report(directory);
            return 0;
        }

        var choice = Known.FirstOrDefault(k => k.Name.Equals(requested, StringComparison.OrdinalIgnoreCase));

        if (choice.Name is null)
        {
            Console.Error.WriteLine(
                $"Unknown model '{requested}'. Known: {string.Join(", ", Known.Select(k => k.Name))}.");
            return 1;
        }

        var kept = Path.Combine(directory, Kept);
        Directory.CreateDirectory(kept);

        var stored = Path.Combine(kept, $"{choice.Name}.onnx");

        if (!File.Exists(stored) || new FileInfo(stored).Length == 0)
        {
            Console.WriteLine($"Fetching {choice.Name} — {choice.Note}");

            if (await FetchAsync(choice.Archive, choice.Inside, stored).ConfigureAwait(false) is { } failure)
            {
                Console.Error.WriteLine($"  {failure}");
                return 1;
            }
        }

        var live = Path.Combine(directory, "segmentation.onnx");

        if (File.Exists(live))
        {
            var backup = Path.Combine(kept, $"{Active(directory) ?? "pyannote"}.onnx");

            if (!File.Exists(backup))
            {
                File.Copy(live, backup);
            }
        }

        File.Copy(stored, live, overwrite: true);
        File.WriteAllText(Path.Combine(directory, ActiveFile), choice.Name);

        Console.WriteLine();
        Console.WriteLine($"Now using {choice.Name} to decide where the speakers change.");
        Console.WriteLine();
        Console.WriteLine("  Transcribe a recording again to see the difference. Watch for shifts that were");
        Console.WriteLine("  missed before rather than for labels being swapped: this model decides where a");
        Console.WriteLine("  turn begins, not whose it is.");

        return 0;
    }

    /// <summary>Downloads and unpacks one model, returning null on success or a reason on failure.</summary>
    private static async Task<string?> FetchAsync(string archive, string inside, string destination)
    {
        var scratch = Path.Combine(Path.GetTempPath(), "localscribe-segmentation-" + Guid.NewGuid().ToString("N"));

        try
        {
            Directory.CreateDirectory(scratch);

            var download = Path.Combine(scratch, archive);

            using (var client = new HttpClient { Timeout = TimeSpan.FromMinutes(30) })
            await using (var source = await client.GetStreamAsync($"{Release}/{archive}").ConfigureAwait(false))
            await using (var target = File.Create(download))
            {
                await source.CopyToAsync(target).ConfigureAwait(false);
            }

            Console.WriteLine("  Unpacking…");
            await new TarBz2Extractor().ExtractAsync(download, scratch).ConfigureAwait(false);

            // By exact name, never by "the first .onnx in there". These archives carry a quantised
            // build beside the real one, and picking that would fail at load on this machine.
            var found = Directory
                .EnumerateFiles(scratch, inside, SearchOption.AllDirectories)
                .FirstOrDefault();

            if (found is null)
            {
                return $"{archive} did not contain {inside}.";
            }

            File.Copy(found, destination, overwrite: true);
            return null;
        }
        catch (Exception exception) when (exception is HttpRequestException or IOException)
        {
            return exception.Message;
        }
        finally
        {
            try
            {
                Directory.Delete(scratch, recursive: true);
            }
            catch (IOException)
            {
                // A leftover temporary directory is not worth failing a working install over.
            }
        }
    }

    private static string? Active(string directory)
    {
        var path = Path.Combine(directory, ActiveFile);

        return File.Exists(path) ? File.ReadAllText(path).Trim() : null;
    }

    private static void Report(string directory)
    {
        Console.WriteLine();
        Console.WriteLine("Segmentation models");
        Console.WriteLine("-------------------");

        var active = Active(directory) ?? "pyannote";
        var kept = Path.Combine(directory, Kept);

        foreach (var (name, _, _, note) in Known)
        {
            var state = name.Equals(active, StringComparison.OrdinalIgnoreCase)
                ? "in use"
                : File.Exists(Path.Combine(kept, $"{name}.onnx")) ? "downloaded" : "not here";

            Console.WriteLine($"  {name,-12} {state,-12} {note}");
        }

        Console.WriteLine();
        Console.WriteLine("  This model decides where a speaker changes; the voice model decides whose");
        Console.WriteLine("  turn it is. Missed shifts are this one. Switch with --segmentation-model <name>.");
    }
}
