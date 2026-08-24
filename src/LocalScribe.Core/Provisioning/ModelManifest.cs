using System.Text.Json;
using System.Text.Json.Serialization;

namespace LocalScribe.Core.Provisioning;

/// <summary>
/// Records which file in a model directory plays which role.
/// <para>
/// Renaming downloaded files to fixed names would be simpler, and would also be a trap: ONNX
/// models above 2 GB store their weights in sidecar files referenced <em>by name from inside
/// the .onnx</em>. Rename the sidecar and the model fails to load; rename the .onnx and it stops
/// finding its weights. So files keep the names their publisher gave them, and this small
/// manifest records what each one is.
/// </para>
/// </summary>
public sealed record ModelManifest
{
    /// <summary>The name this manifest is stored under inside a model directory.</summary>
    public const string FileName = "localscribe-model.json";

    /// <summary>File name of the encoder graph, relative to the model directory.</summary>
    [JsonPropertyName("encoder")]
    public required string Encoder { get; init; }

    /// <summary>File name of the decoder graph, relative to the model directory.</summary>
    [JsonPropertyName("decoder")]
    public required string Decoder { get; init; }

    /// <summary>File name of the token vocabulary, relative to the model directory.</summary>
    [JsonPropertyName("vocab")]
    public required string Vocab { get; init; }

    /// <summary>Where these files came from, so a stale or wrong download can be traced.</summary>
    [JsonPropertyName("source")]
    public string? Source { get; init; }

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
    };

    /// <summary>
    /// The names assumed when a directory has no manifest. Lets a hand-assembled directory work
    /// without ceremony, which is what most people will try first.
    /// </summary>
    public static ModelManifest Conventional { get; } = new()
    {
        Encoder = "encoder.onnx",
        Decoder = "decoder.onnx",
        Vocab = "vocab.json",
    };

    /// <summary>
    /// What a directory holds when it carries no manifest, or null when it holds no model.
    /// <para>
    /// Asks <see cref="Core.Models.ModelLayout"/> rather than testing for
    /// <c>encoder.onnx</c> here, because that is where the convention is defined and it knows
    /// about more than one. AI Hub ships <c>encoder/model.onnx</c> beside the context binary; a
    /// second opinion that only recognised the flat names called such a directory empty while
    /// the app was loading models out of it perfectly well.
    /// </para>
    /// </summary>
    private static ModelManifest? Assumed(string directory)
    {
        var encoder = Models.ModelLayout.GraphPath(directory, "encoder");
        var decoder = Models.ModelLayout.GraphPath(directory, "decoder");

        if (encoder is null || decoder is null)
        {
            return null;
        }

        return new ModelManifest
        {
            Encoder = Path.GetRelativePath(directory, encoder),
            Decoder = Path.GetRelativePath(directory, decoder),
            Vocab = Conventional.Vocab,
        };
    }

    /// <summary>
    /// Reads the manifest from a model directory, falling back to conventional names.
    /// Returns <c>null</c> when the directory does not hold a usable model either way.
    /// </summary>
    public static ModelManifest? Discover(string directory)
    {
        if (!Directory.Exists(directory))
        {
            return null;
        }

        var manifestPath = Path.Combine(directory, FileName);
        if (File.Exists(manifestPath))
        {
            try
            {
                var layout = JsonSerializer.Deserialize<ModelManifest>(
                    File.ReadAllText(manifestPath),
                    JsonOptions);

                if (layout is not null && layout.IsComplete(directory))
                {
                    return layout;
                }
            }
            catch (JsonException)
            {
                // A corrupt manifest should not be fatal; the conventional names may still work.
            }
        }

        return Assumed(directory) is { } assumed && assumed.IsComplete(directory) ? assumed : null;
    }

    /// <summary>True when every file this layout names is present in the directory.</summary>
    public bool IsComplete(string directory) =>
        File.Exists(Path.Combine(directory, Encoder))
        && File.Exists(Path.Combine(directory, Decoder))
        && File.Exists(Path.Combine(directory, Vocab));

    /// <summary>Writes the manifest into a model directory.</summary>
    public void Save(string directory) =>
        File.WriteAllText(
            Path.Combine(directory, FileName),
            JsonSerializer.Serialize(this, JsonOptions));

    /// <summary>Absolute path of the encoder graph.</summary>
    public string EncoderPath(string directory) => Path.Combine(directory, Encoder);

    /// <summary>Absolute path of the decoder graph.</summary>
    public string DecoderPath(string directory) => Path.Combine(directory, Decoder);

    /// <summary>Absolute path of the vocabulary.</summary>
    public string VocabPath(string directory) => Path.Combine(directory, Vocab);

    /// <summary>
    /// Works out which downloaded files play which role by looking at their names.
    /// <para>
    /// Publishers disagree about naming — <c>encoder.onnx</c>, <c>model_encoder.onnx</c>,
    /// <c>WhisperEncoder.onnx</c>, and <c>HfWhisperEncoder.onnx</c> all appear in the wild — so
    /// matching on role words is more durable than expecting any one convention.
    /// </para>
    /// </summary>
    /// <returns>The inferred layout, or <c>null</c> when a required role could not be filled.</returns>
    public static ModelManifest? Infer(IEnumerable<string> fileNames, string? source = null)
    {
        ArgumentNullException.ThrowIfNull(fileNames);

        var names = fileNames.Select(Path.GetFileName).OfType<string>().ToList();

        var encoder = PickOnnx(names, "encoder");
        var decoder = PickOnnx(names, "decoder");
        var vocab = names.FirstOrDefault(n => n.Equals("vocab.json", StringComparison.OrdinalIgnoreCase));

        return encoder is null || decoder is null || vocab is null
            ? null
            : new ModelManifest { Encoder = encoder, Decoder = decoder, Vocab = vocab, Source = source };
    }

    /// <summary>
    /// Finds the ONNX graph for a role. Prefers the shortest match, because exports commonly
    /// ship both <c>decoder.onnx</c> and <c>decoder_with_past.onnx</c>, and the plain one is the
    /// graph this pipeline's decode loop is written against.
    /// </summary>
    private static string? PickOnnx(IEnumerable<string> names, string role) =>
        names
            .Where(n => n.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            .Where(n => n.Contains(role, StringComparison.OrdinalIgnoreCase))
            .OrderBy(n => n.Length)
            .ThenBy(n => n, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();
}
