using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Headless;
using Avalonia.Input;
using Avalonia.Input.Raw;
using Avalonia.Styling;
using Avalonia.Threading;
using Avalonia.VisualTree;
using LocalScribe.Desktop;

// window-snapshot <out-dir> [file.scrb|audio] [--size WxH] [--wait seconds] [--mid-run seconds] [--flyouts] [--correct]
//
// Opens the real MainWindow offscreen, lets it initialise (and open the file, if given), and
// writes window-light.png and window-dark.png. With --flyouts it also opens each toolbar
// flyout and the models panel and writes them, since those are half of the window's surface.
// With --correct it right-clicks a word and drags over two, prints the menu each brings up, then
// corrects a word through the view model and prints what the row and its times became.

var outDir = args.Length > 0 ? args[0] : "snapshots";
var width = 980.0;
var height = 720.0;
var wait = 20.0;
var flyouts = args.Contains("--flyouts");
var midRun = 30.0;

for (var i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--size" && args[i + 1].Split('x') is [var w, var h])
    {
        (width, height) = (double.Parse(w), double.Parse(h));
    }
    else if (args[i] == "--wait")
    {
        wait = double.Parse(args[i + 1]);
    }
    else if (args[i] == "--mid-run")
    {
        midRun = double.Parse(args[i + 1]);
    }
}

var file = args.Skip(1).FirstOrDefault(File.Exists);
Directory.CreateDirectory(outDir);

AppBuilder.Configure<App>()
    .UseSkia()
    .UseHeadless(new AvaloniaHeadlessPlatformOptions { UseHeadlessDrawing = false })
    .SetupWithoutStarting();

void Pump(double seconds, Func<bool>? until = null)
{
    var end = DateTime.UtcNow.AddSeconds(seconds);

    while (DateTime.UtcNow < end && until?.Invoke() != true)
    {
        Dispatcher.UIThread.RunJobs();
        AvaloniaHeadlessPlatform.ForceRenderTimerTick();
        Thread.Sleep(50);
    }

    Dispatcher.UIThread.RunJobs();
    AvaloniaHeadlessPlatform.ForceRenderTimerTick();
}

void Save(TopLevel top, string name)
{
    Pump(0.3);
    var frame = top.CaptureRenderedFrame();

    if (frame is null)
    {
        Console.Error.WriteLine($"no frame for {name}");
        return;
    }

    var path = Path.Combine(outDir, name);
    frame.Save(path);
    Console.WriteLine(path);
}

foreach (var theme in new[] { ThemeVariant.Light, ThemeVariant.Dark })
{
    Application.Current!.RequestedThemeVariant = theme;
    var label = theme == ThemeVariant.Light ? "light" : "dark";

    var window = new MainWindow(null) { Width = width, Height = height };
    window.Show();
    Pump(3);

    // Opened through the view model rather than the window's launch path, which waits for
    // Whisper to preload first — a wait the offscreen platform never finishes.
    var viewModel = typeof(MainWindow)
        .GetField("_viewModel", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
        .GetValue(window)!;
    var panel = window.FindControl<StackPanel>("ParagraphsPanel");

    if (file is not null && Path.GetExtension(file) is ".scrb")
    {
        viewModel.GetType().GetMethod("OpenArchive")!.Invoke(viewModel, [file]);
        Pump(wait, () => panel?.Children.Count > 0);
    }
    else if (file is not null)
    {
        // A recording is transcribed for real, and caught partway: the working state — glow,
        // progress, rows streaming in — is the one most worth seeing and the hardest to stage.
        // The model needs its hardware check first, or the run waits on it.
        Pump(wait, () => viewModel.GetType().GetProperty("IsModelReady")?.GetValue(viewModel) is true);
        viewModel.GetType().GetMethod("TranscribeFileAsync")!.Invoke(viewModel, [file]);
        Pump(midRun);
    }

    Pump(1.5);
    Console.WriteLine($"rows on screen: {panel?.Children.Count}, status: {window.FindControl<TextBlock>("StatusText")?.Text}");

    // Partway through, so the waveform shows its played and unplayed halves.
    if (file is not null && Path.GetExtension(file) is ".scrb")
    {
        var waveform = typeof(MainWindow)
            .GetField("_waveform", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
            .GetValue(window)!;
        waveform.GetType().GetMethod("SetPosition")!.Invoke(waveform, [40.0]);
    }

    if (args.Contains("--correct") && label == "light" && panel is not null)
    {
        var block = panel.GetVisualDescendants().OfType<SelectableTextBlock>().First();
        var text = block.Inlines?.Text ?? block.Text ?? string.Empty;
        Console.WriteLine($"first row: {text[..Math.Min(90, text.Length)]}");

        Point Middle(int index) =>
            block.TranslatePoint(block.TextLayout.HitTestTextPosition(index).Center, window)!.Value;

        void PrintMenu(string what)
        {
            var items = block.ContextMenu?.Items.Select(item => (item as MenuItem)?.Header?.ToString() ?? "---");
            Console.WriteLine($"{what}: {(items is null ? "no menu" : string.Join(" | ", items))}");
            block.ContextMenu?.Close();
            Pump(0.2);
        }

        var second = text.IndexOf(' ') + 2;
        var third = text.IndexOf(' ', second) + 2;

        window.MouseDown(Middle(second), MouseButton.Right);
        window.MouseUp(Middle(second), MouseButton.Right);
        Pump(0.4);
        PrintMenu("right-click on a word");

        window.MouseMove(Middle(second));
        window.MouseDown(Middle(second), MouseButton.Left);
        window.MouseMove(Middle(third), RawInputModifiers.LeftMouseButton);
        window.MouseUp(Middle(third), MouseButton.Left);
        Pump(0.3);
        Console.WriteLine($"selected: {block.SelectedText}");
        window.MouseDown(Middle(second), MouseButton.Right);
        window.MouseUp(Middle(second), MouseButton.Right);
        Pump(0.4);
        PrintMenu("right-click on that selection");

        var wrong = text[(text.IndexOf(' ') + 1)..].Split(' ')[0].Trim(',', '.', '?', '!');
        var words = viewModel.GetType().GetMethod("PlacesOf")!.Invoke(viewModel, [wrong]);
        Console.WriteLine($"{wrong}: {words} place(s)");
        viewModel.GetType().GetMethod("CorrectWords")!.Invoke(viewModel, [wrong, "CORRECTED Words", true, 0.0]);
        Pump(1.5);
        block = panel.GetVisualDescendants().OfType<SelectableTextBlock>().First();
        text = block.Inlines?.Text ?? block.Text ?? string.Empty;
        Console.WriteLine($"after: {text[..Math.Min(90, text.Length)]}");
        Console.WriteLine($"status: {window.FindControl<TextBlock>("StatusText")?.Text}");
    }

    if (args.Contains("--follow") && label == "light" && file is not null)
    {
        // Plays by painting the marker forward, as the player's ticks do, then scrolls the way a
        // trackpad does and keeps going: the view must follow first and stay put after.
        var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var paint = typeof(MainWindow).GetMethod("PaintMarker", flags)!;
        var follow = typeof(MainWindow).GetField("_followPlayback", flags)!;
        var scroll = window.FindControl<ScrollViewer>("TranscriptScroll")!;
        follow.SetValue(window, true);

        for (var t = 0.0; t <= 240; t += 20)
        {
            paint.Invoke(window, [t]);
            Pump(0.15);
            Console.WriteLine($"follow t={t,5:F0} s: offset {scroll.Offset.Y,7:F0}, following {follow.GetValue(window)}");
        }

        var middle = scroll.TranslatePoint(new Point(scroll.Bounds.Width / 2, scroll.Bounds.Height / 2), window)!.Value;
        window.MouseWheel(middle, new Vector(0, 3));
        Pump(0.3);
        var after = scroll.Offset.Y;
        Console.WriteLine($"wheel up: offset {after:F0}, following {follow.GetValue(window)}");

        for (var t = 260.0; t <= 400; t += 20)
        {
            paint.Invoke(window, [t]);
            Pump(0.15);
        }

        Console.WriteLine($"after more playback: offset {scroll.Offset.Y:F0} (was {after:F0}), following {follow.GetValue(window)}");
    }

    Save(window, $"window-{label}.png");

    if (flyouts)
    {
        foreach (var name in new[] { "PaceButton", "SpeakersButton", "ModelsChip" })
        {
            if (window.FindControl<Button>(name) is not { Flyout: PopupFlyoutBase flyout } button)
            {
                continue;
            }

            // The headless platform gives a flyout's popup no host of its own, so its content is
            // lent to a window with the same presenter the popup would have used, then returned.
            if (flyout is Flyout { Content: Control content } owned)
            {
                owned.Content = null;
                var stand = new Window
                {
                    SizeToContent = SizeToContent.WidthAndHeight,
                    Content = new FlyoutPresenter { Content = content },
                };
                stand.Show();
                Pump(0.6);
                Save(stand, $"flyout-{label}-{name}.png");
                ((FlyoutPresenter)stand.Content).Content = null;
                stand.Close();
                owned.Content = content;
            }

            Pump(0.3);
        }
    }

    window.Close();
    Pump(0.5);
}
