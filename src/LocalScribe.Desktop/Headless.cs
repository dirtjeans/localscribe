using LocalScribe.App;

namespace LocalScribe.Desktop;

/// <summary>
/// Runs the window's whole pipeline with no window: the same view model, the same engine,
/// every status change printed with a timestamp.
/// <para>
/// The window needs a display — Avalonia cannot even start its render timer while the screen
/// is asleep or locked, and dies on launch with error -6661 — so a run started from a script
/// while nobody is at the Mac could never say anything about the pipeline. The view model
/// was built to hold no UI types; this is that design paying out. Usage:
/// <c>LocalScribe.Desktop --headless &lt;audio file&gt; [--save &lt;out.scrb&gt;]</c>.
/// </para>
/// </summary>
internal static class Headless
{
    public static int Run(string path, string? saveTo = null)
    {
        if (!File.Exists(path))
        {
            Console.Error.WriteLine($"No such file: {path}");
            return 1;
        }

        using var viewModel = new MainViewModel(MainWindow.FindModelRoot(), MainWindow.OpenEngine);

        var last = string.Empty;
        var lastActive = MainViewModel.ModelRoles.None;
        var lastClickable = string.Empty;
        var renamed = false;
        viewModel.PropertyChanged += (_, e) =>
        {
            // Status, the model roster, and which models are working: progress fires many
            // times a second and would bury the lines that say what changed.
            if (e.PropertyName == nameof(MainViewModel.Status) && viewModel.Status != last)
            {
                last = viewModel.Status;
                Console.WriteLine($"{DateTime.Now:HH:mm:ss} {last}");
            }
            else if (e.PropertyName == nameof(MainViewModel.ActiveModels) && viewModel.ActiveModels != lastActive)
            {
                lastActive = viewModel.ActiveModels;
                Console.WriteLine($"{DateTime.Now:HH:mm:ss} [working] {lastActive}");
            }
            else if (e.PropertyName == nameof(MainViewModel.Paragraphs) && viewModel.IsBusy)
            {
                // What the window would draw clickable mid-run: the claim that streamed text is
                // usable at once is settled here, not by looking at the window.
                var segments = viewModel.Paragraphs.SelectMany(p => p.Segments).ToList();
                var clickable = $"{segments.Count(viewModel.IsTimed)} of {segments.Count}";

                if (clickable != lastClickable)
                {
                    lastClickable = clickable;
                    Console.WriteLine($"{DateTime.Now:HH:mm:ss} [clickable] {clickable} segments");
                }
            }
            else if (e.PropertyName == nameof(MainViewModel.SpeakerCount))
            {
                Console.WriteLine($"{DateTime.Now:HH:mm:ss} [speakers] {viewModel.SpeakerCount}");

                // LOCALSCRIBE_HEADLESS_RENAME=<name> renames Speaker 1 the moment labels first
                // appear, the way a reader would, so the claim that a mid-run name survives the
                // rest of the run is checked by the run itself. Off the event's thread: the
                // event fires from inside the publish that the rename would re-enter.
                if (!renamed && viewModel.IsBusy && viewModel.SpeakerCount > 0
                    && Environment.GetEnvironmentVariable("LOCALSCRIBE_HEADLESS_RENAME") is { Length: > 0 } name)
                {
                    renamed = true;
                    _ = Task.Run(() =>
                    {
                        viewModel.RenameSpeaker(0, 0, "Speaker 1", name, everywhere: true);
                        Console.WriteLine($"{DateTime.Now:HH:mm:ss} [renamed] Speaker 1 -> {name}: "
                            + string.Join(", ", viewModel.Paragraphs.Select(p => p.Speaker).Distinct()));
                    });
                }
            }
            else if (e.PropertyName == nameof(MainViewModel.HardwareSummary))
            {
                Console.WriteLine($"{DateTime.Now:HH:mm:ss} [models] {viewModel.HardwareSummary}");
            }
        };

        // Held for the whole run: a long command-line job deserves the same protection from
        // sleep and App Nap as the window gives itself — and running it here is what proves
        // the native call works without needing a display to open the window.
        SleepAssertion.While(true);

        try
        {
            Task.Run(async () =>
            {
                await viewModel.InitialiseAsync();
                await viewModel.TranscribeFileAsync(path);
            }).GetAwaiter().GetResult();
        }
        finally
        {
            SleepAssertion.While(false);
        }

        Console.WriteLine($"{DateTime.Now:HH:mm:ss} [done] {viewModel.Paragraphs.Count} paragraph(s), "
            + $"measured words: {viewModel.HasMeasuredWords}, speakers: "
            + string.Join(", ", viewModel.Paragraphs.Select(p => p.Speaker).Distinct()));

        // Saved so the doctor's --check-words can grade the result against its own audio:
        // "measured" says every word was placed, only a re-listen says each landed right.
        if (saveTo is not null && viewModel.CanSaveArchive)
        {
            viewModel.SaveArchive(saveTo);
            Console.WriteLine($"{DateTime.Now:HH:mm:ss} [saved] {saveTo}");
        }

        return 0;
    }
}
