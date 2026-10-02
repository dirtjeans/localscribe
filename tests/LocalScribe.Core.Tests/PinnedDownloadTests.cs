using System.Net;
using System.Security.Cryptography;
using LocalScribe.Core.Models;
using Xunit;

namespace LocalScribe.Core.Tests;

/// <summary>
/// A pinned download either is the file it was pinned to or never reaches the name the app
/// looks for. The second half is the point: a file that is present gets loaded.
/// </summary>
public class PinnedDownloadTests
{
    private sealed class Serves(byte[] body) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(body) });
    }

    private static readonly byte[] Model = [1, 2, 3, 4, 5];

    private static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private static async Task<(string Directory, Exception? Failure)> Fetch(byte[] served, string? pin)
    {
        var directory = Path.Combine(Path.GetTempPath(), $"localscribe-pinned-{Guid.NewGuid():N}");
        var fetcher = new ModelFetcher(new HttpClient(new Serves(served)));
        var download = new ModelDownload(new Uri("https://example.invalid/model.onnx"), "model.onnx", Sha256: pin);

        try
        {
            await fetcher.FetchAsync(directory, [download]);
            return (directory, null);
        }
        catch (Exception exception)
        {
            return (directory, exception);
        }
    }

    [Fact]
    public async Task TheFileItWasPinnedToArrives()
    {
        var (directory, failure) = await Fetch(Model, Hash(Model));

        try
        {
            Assert.Null(failure);
            Assert.Equal(Model, File.ReadAllBytes(Path.Combine(directory, "model.onnx")));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task AChangedUploadNeverReachesTheModelsName()
    {
        var (directory, failure) = await Fetch([9, 9, 9], Hash(Model));

        try
        {
            Assert.IsType<InvalidDataException>(failure);
            Assert.False(File.Exists(Path.Combine(directory, "model.onnx")));
            Assert.Empty(Directory.EnumerateFiles(directory));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task AnUnpinnedFileIsTakenAsServed()
    {
        var (directory, failure) = await Fetch([9, 9, 9], pin: null);

        try
        {
            Assert.Null(failure);
            Assert.True(File.Exists(Path.Combine(directory, "model.onnx")));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
