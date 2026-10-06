using System.Text.RegularExpressions;
using LocalScribe.Core.Models;
using LocalScribe.Onnx;

namespace LocalScribe.App;

/// <summary>One model the window can name, light while it works, and describe on request.</summary>
/// <param name="Role">Which of the view model's working roles lights it.</param>
/// <param name="Name">A word or two, for the chip.</param>
/// <param name="Title">What it does, for the panel.</param>
/// <param name="Detail">The full description, for the panel.</param>
public sealed record ModelEntry(MainViewModel.ModelRoles Role, string Name, string Title, string Detail);

/// <summary>
/// Reads the view model's one-line roster — "transcriber · speaker model · cleanup" — into the
/// models a window shows, with the word aligner added.
/// <para>
/// Written first in the Mac window, which took the line apart by its separator rather than
/// change what Windows read; moved here when the Windows window wanted the same chip, so the
/// two cannot drift. The Avalonia window can link this file and drop its own copy.
/// </para>
/// </summary>
public static class ModelRoster
{
    /// <summary>The models named in a roster line, the aligner second, in pipeline order.</summary>
    /// <param name="summary">The view model's <c>HardwareSummary</c>.</param>
    /// <param name="modelRoot">Where the models live, to say which aligner build is installed.</param>
    public static IReadOnlyList<ModelEntry> From(string summary, string modelRoot)
    {
        var parts = summary
            .Split(" · ", StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToList();

        // The aligner is not in the shared roster line, but it is the stage that runs longest,
        // so a chip that lights up what is working cannot leave it out.
        if (parts.Count > 0 && AlignerName(modelRoot) is { } aligner)
        {
            parts.Insert(1, aligner);
        }

        return [.. parts.Select(Entry)];
    }

    /// <summary>The installed aligner build, as the roster would name it, or null with none.</summary>
    public static string? AlignerName(string modelRoot)
    {
        if (ForcedAligner.Find(modelRoot) is not { } directory)
        {
            return null;
        }

        // Compiled for the NPU only where the planner put Whisper there, so this is the build
        // that scans every recording longer than one window.
        if (NpuAligner.IsReady(directory))
        {
            return "MMS word aligner, fp16 on the NPU";
        }

        var quantised = AlignmentModelSource.PreferQuantised
            && File.Exists(Path.Combine(directory, AlignmentModelSource.QuantisedModelFileName));

        return quantised ? "MMS word aligner, 4-bit" : "MMS word aligner, fp16";
    }

    private static ModelEntry Entry(string part)
    {
        var p = part.ToLowerInvariant();

        if (p.Contains("aligner") && p.Contains("on the npu"))
        {
            return new(MainViewModel.ModelRoles.Words, "MMS aligner", "Word timing",
                part.Replace(" on the NPU", string.Empty, StringComparison.Ordinal)
                + " — on the Hexagon NPU, beside Whisper; places each word against the audio. "
                + $"Recordings under {NpuAligner.ShortestSeconds:F0} seconds are timed on the CPU");
        }

        if (p.Contains("aligner"))
        {
            return new(MainViewModel.ModelRoles.Words, "MMS aligner", "Word timing",
                part + " — on the CPU; places each word against the audio");
        }

        if (p.Contains("whisper") || p.StartsWith("encoder", StringComparison.Ordinal))
        {
            // The plan names devices by their enum ("on Npu"); the panel says them as words.
            var detail = part
                .Replace("on Npu", "on the Hexagon NPU", StringComparison.Ordinal)
                .Replace("on Gpu", "on the GPU", StringComparison.Ordinal)
                .Replace("on Cpu", "on the CPU", StringComparison.Ordinal);

            if (p.Contains("coreml"))
            {
                detail += " — encoder on the Neural Engine";
            }

            return new(MainViewModel.ModelRoles.Transcription,
                p.Contains("turbo") ? "Whisper turbo" : "Whisper",
                "Transcription",
                detail);
        }

        if (p.Contains("speakers"))
        {
            return new(MainViewModel.ModelRoles.Speakers,
                p.Contains("nemotron") ? "Nemotron" : p.Contains("pyannote") ? "pyannote" : "speakers",
                "Who spoke",
                part.Replace(" speakers", string.Empty, StringComparison.Ordinal) + " — on the CPU");
        }

        if (p.Contains("cleanup off"))
        {
            return new(MainViewModel.ModelRoles.None, "cleanup off", "Punctuation cleanup",
                "Off — Whisper's own punctuation is used. It can be turned on with the processing "
                + "speed settings, and adds time to each transcription.");
        }

        if (p.Contains("no cleanup"))
        {
            return new(MainViewModel.ModelRoles.None, "no cleanup", "Punctuation cleanup",
                "No local language model is running, so transcripts keep Whisper's punctuation.");
        }

        if (p.Contains("cleanup") || p.Contains("qwen"))
        {
            var model = part.Replace("cleanup: ", string.Empty, StringComparison.Ordinal);
            return new(MainViewModel.ModelRoles.Cleanup, CleanupName(model), "Punctuation cleanup", model);
        }

        // An unknown shape — a new model — keeps its first few words rather than vanishing.
        return new(MainViewModel.ModelRoles.None, string.Join(' ', part.Split(' ').Take(3)), "Model", part);
    }

    /// <summary>"qwen2.5-1.5b-instruct via Foundry Local" reads as "Qwen 1.5B" in a chip.</summary>
    private static string CleanupName(string model)
    {
        var p = model.ToLowerInvariant();

        if (p.Contains("qwen"))
        {
            var size = Regex.Match(p, @"(\d+(\.\d+)?)b");
            return size.Success ? $"Qwen {size.Groups[1].Value}B" : "Qwen";
        }

        if (p.Contains("phi"))
        {
            var version = Regex.Match(p, @"phi-(\d+(\.\d+)?)(-(mini|small|medium))?");
            return version.Success
                ? $"Phi-{version.Groups[1].Value}{(version.Groups[4].Success ? " " + version.Groups[4].Value : string.Empty)}"
                : "Phi";
        }

        var name = model.Split(" via ", StringSplitOptions.TrimEntries)[0];
        return name.Length <= 16 ? name : name[..16] + "…";
    }
}
