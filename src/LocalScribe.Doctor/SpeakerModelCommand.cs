using LocalScribe.Core.Models;
using LocalScribe.Core.Provisioning;

namespace LocalScribe.Doctor;

/// <summary>
/// Switches which voice embedding model the app uses, so two can be compared by listening.
/// <para>
/// Measuring one model against another needs labels, and the labels on a transcript this app
/// produced came from one of the models being compared. That circularity cannot be reasoned
/// away, only avoided: a person listening to the recording is not downstream of the model, and
/// is the only judge here who is not.
/// </para>
/// <para>
/// Downloaded models are kept, so switching back and forth costs nothing after the first time
/// and a comparison can be made on several recordings rather than on the first impression of
/// one.
/// </para>
/// </summary>
public static class SpeakerModelCommand
{
    private const string Release =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models";

    /// <summary>Where downloaded alternatives are kept, beside the one in use.</summary>
    private const string Kept = "embeddings";

    /// <summary>Records which one is in use, since the file in use is a copy and looks like any other.</summary>
    private const string ActiveFile = "active-embedding.txt";

    /// <summary>
    /// What can be switched to.
    /// <para>
    /// WeSpeaker only. These take filterbank features on an input named <c>feats</c>, which is
    /// what this app computes; NeMo and 3D-Speaker models take raw audio on a differently named
    /// input and would fail rather than score worse.
    /// </para>
    /// </summary>
    private static readonly (string Name, string Asset, string Note)[] Known =
    [
        ("resnet34", "wespeaker_en_voxceleb_resnet34_LM.onnx",
            "26 MB. The default, and what every threshold here was calibrated against."),
        ("resnet221", "wespeaker_en_voxceleb_resnet221_LM.onnx",
            "95 MB. Roughly a third fewer errors than resnet34 on VoxCeleb."),
        ("resnet293", "wespeaker_en_voxceleb_resnet293_LM.onnx",
            "114 MB. The best of the family on VoxCeleb."),
        ("campp", "wespeaker_en_voxceleb_CAM++_LM.onnx",
            "29 MB. Level with resnet34 on VoxCeleb; separated nothing on the debate recording."),
    ];

    public static async Task<int> RunAsync(string modelRoot, string? requested)
    {
        var directory = Path.Combine(modelRoot, "diarization");

        if (!Directory.Exists(directory))
        {
            Console.Error.WriteLine(
                $"No speaker models under {directory}. Run --fetch-models first.");
            return 1;
        }

        if (string.IsNullOrWhiteSpace(requested))
        {
            Report(directory);
            return 0;
        }

        var choice = Known.FirstOrDefault(k =>
            k.Name.Equals(requested, StringComparison.OrdinalIgnoreCase));

        if (choice.Name is null)
        {
            Console.Error.WriteLine($"Unknown model '{requested}'. Known: {string.Join(", ", Known.Select(k => k.Name))}.");
            return 1;
        }

        var kept = Path.Combine(directory, Kept);
        Directory.CreateDirectory(kept);

        var stored = Path.Combine(kept, $"{choice.Name}.onnx");

        if (!File.Exists(stored) || new FileInfo(stored).Length == 0)
        {
            Console.WriteLine($"Fetching {choice.Name} — {choice.Note}");

            try
            {
                using var downloader = new HttpFileDownloader();

                var last = string.Empty;

                await downloader.DownloadAsync(
                    new RemoteFile($"{Release}/{choice.Asset}", $"{choice.Name}.onnx"),
                    kept,
                    new Progress<InstallProgress>(update =>
                    {
                        if (update.Message == last)
                        {
                            return;
                        }

                        last = update.Message;
                        Console.WriteLine($"  {update.Message}");
                    })).ConfigureAwait(false);
            }
            catch (Exception exception) when (exception is HttpRequestException or IOException)
            {
                Console.Error.WriteLine($"Could not fetch {choice.Name}: {exception.Message}");

                if (File.Exists(stored))
                {
                    File.Delete(stored);
                }

                return 1;
            }
        }

        // Keep whatever is in use, so the first switch away from a hand-placed model does not
        // lose it. Named for what it is only when we can tell; otherwise kept as "previous".
        var live = Path.Combine(directory, "embedding.onnx");

        if (File.Exists(live))
        {
            var name = Active(directory) ?? "previous";
            var backup = Path.Combine(kept, $"{name}.onnx");

            if (!File.Exists(backup))
            {
                File.Copy(live, backup);
            }
        }

        File.Copy(stored, live, overwrite: true);
        File.WriteAllText(Path.Combine(directory, ActiveFile), choice.Name);

        Console.WriteLine();
        Console.WriteLine($"Now using {choice.Name}.");
        Console.WriteLine();
        Console.WriteLine("  Transcribe a recording again to hear the difference — speaker labels are");
        Console.WriteLine("  worked out from the audio, so an already-open transcript will not change.");

        if (!choice.Name.Equals("resnet34", StringComparison.Ordinal))
        {
            Console.WriteLine();
            Console.WriteLine("  The separation threshold was calibrated for resnet34. If this one splits or");
            Console.WriteLine("  merges voices that resnet34 got right, that may be the threshold rather than");
            Console.WriteLine("  the model: --diarize <wav> --sweep shows where the answer changes.");
        }

        return 0;
    }

    /// <summary>
    /// Notes that the fetched default is what is in use.
    /// <para>
    /// Called after --fetch-models, which pins the WeSpeaker ResNet34-LM weights. Without it the
    /// first switch away has no name to keep the old file under.
    /// </para>
    /// </summary>
    public static void RecordInstalled(string directory) =>
        File.WriteAllText(Path.Combine(directory, ActiveFile), "resnet34");

    /// <summary>The model in use, or null when nothing has recorded one.</summary>
    private static string? Active(string directory)
    {
        var path = Path.Combine(directory, ActiveFile);

        return File.Exists(path) ? File.ReadAllText(path).Trim() : null;
    }

    private static void Report(string directory)
    {
        Console.WriteLine();
        Console.WriteLine("Voice embedding models");
        Console.WriteLine("----------------------");

        var active = Active(directory);
        var kept = Path.Combine(directory, Kept);

        foreach (var (name, _, note) in Known)
        {
            var stored = Path.Combine(kept, $"{name}.onnx");

            var state = name.Equals(active, StringComparison.OrdinalIgnoreCase)
                ? "in use"
                : File.Exists(stored) ? "downloaded" : "not here";

            Console.WriteLine($"  {name,-12} {state,-12} {note}");
        }

        if (active is null)
        {
            Console.WriteLine();
            Console.WriteLine("  Nothing recorded as in use, so this is whatever was installed first.");
        }

        Console.WriteLine();
        Console.WriteLine("  Switch with --speaker-model <name>. Downloads are kept, so switching back is free.");
    }
}
