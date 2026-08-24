using System.Formats.Tar;
using ICSharpCode.SharpZipLib.BZip2;
using LocalScribe.Core.Provisioning;

namespace LocalScribe.Doctor;

/// <summary>
/// Unpacks the <c>.tar.bz2</c> archives sherpa-onnx publishes its models in.
/// </summary>
public sealed class TarBz2Extractor : IArchiveExtractor
{
    public bool CanExtract(string archivePath) =>
        archivePath.EndsWith(".tar.bz2", StringComparison.OrdinalIgnoreCase)
        || archivePath.EndsWith(".tbz2", StringComparison.OrdinalIgnoreCase);

    public async Task ExtractAsync(
        string archivePath,
        string targetDirectory,
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(targetDirectory);

        // Decompress to a temporary tar first. Streaming bzip2 straight into the tar reader
        // works, but a corrupt download then surfaces as a confusing tar error rather than a
        // decompression one, which sends you looking in the wrong place.
        var tarPath = Path.Combine(targetDirectory, Path.GetRandomFileName() + ".tar");

        try
        {
            await using (var compressed = File.OpenRead(archivePath))
            await using (var tar = File.Create(tarPath))
            {
                BZip2.Decompress(compressed, tar, isStreamOwner: false);
            }

            await TarFile
                .ExtractToDirectoryAsync(tarPath, targetDirectory, overwriteFiles: true, cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            if (File.Exists(tarPath))
            {
                File.Delete(tarPath);
            }
        }
    }
}
