using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Markup.Xaml.MarkupExtensions;
using Avalonia.Media;
using LocalScribe.Core.Audio;
using LocalScribe.Core.Pipeline;

namespace LocalScribe.Desktop;

/// <summary>
/// The order several files will be joined in, shown before anything is transcribed: what decided
/// it, anything that argues otherwise, and a way to move a file. The order is proposed from what
/// the files say about themselves (<see cref="RecordingOrder"/>), but the person who made the
/// recordings knows, and a wrong order costs a whole run to discover. Null means cancel.
/// </summary>
internal sealed class OrderDialog : Window
{
    private readonly List<RecordingFacts> _files;
    private readonly RecordingOrder.Proposal _proposal;
    private readonly StackPanel _rows = new() { Spacing = 2 };
    private readonly TextBlock _reason = new() { TextWrapping = TextWrapping.Wrap };
    private bool _moved;

    private OrderDialog(RecordingOrder.Proposal proposal)
    {
        _proposal = proposal;
        _files = [.. proposal.Ordered];

        Title = "Transcribe as one recording";
        Width = 520;
        SizeToContent = SizeToContent.Height;
        CanResize = false;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;

        var go = new Button { Content = "Transcribe as One", IsDefault = true };
        var cancel = new Button { Content = "Cancel", IsCancel = true };
        go.Click += (_, _) => Close((IReadOnlyList<string>)[.. _files.Select(f => f.Path)]);
        cancel.Click += (_, _) => Close(null);

        var layout = new StackPanel { Margin = new Thickness(20), Spacing = 12 };

        layout.Children.Add(new TextBlock
        {
            Text = $"These {_files.Count} files will be joined into one recording and transcribed in this order.",
            TextWrapping = TextWrapping.Wrap,
        });

        layout.Children.Add(Quiet(_reason));

        foreach (var doubt in proposal.Doubts)
        {
            layout.Children.Add(new TextBlock
            {
                Text = "⚠ " + doubt,
                TextWrapping = TextWrapping.Wrap,
                Foreground = new SolidColorBrush(Color.Parse("#B45309")),
            });
        }

        layout.Children.Add(new ScrollViewer { Content = _rows, MaxHeight = 360 });

        layout.Children.Add(new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
            Spacing = 8,
            Children = { cancel, go },
        });

        Content = layout;
        Fill();
    }

    internal static async Task<IReadOnlyList<string>?> AskAsync(Window owner, RecordingOrder.Proposal proposal) =>
        await new OrderDialog(proposal).ShowDialog<IReadOnlyList<string>?>(owner);

    private void Fill()
    {
        _reason.Text = _moved ? "In the order you chose." : _proposal.Reason + TotalLength();
        _rows.Children.Clear();

        // Gaps are the proposal's, so only while its order stands: moved, they would measure
        // between files that are no longer neighbours.
        var gaps = _moved ? null : _proposal.GapsBefore;

        for (var i = 0; i < _files.Count; i++)
        {
            if (gaps?[i] is { } gap && Math.Abs(gap) >= 5)
            {
                _rows.Children.Add(Quiet(new TextBlock
                {
                    Text = gap > 0 ? $"    {Length(gap)} gap" : $"    overlaps the one before by {Length(-gap)}",
                    FontSize = 11.5,
                    FontStyle = FontStyle.Italic,
                }));
            }

            _rows.Children.Add(Row(i));
        }
    }

    private Control Row(int index)
    {
        var file = _files[index];

        var up = new Button { Content = "▲", IsEnabled = index > 0, Padding = new Thickness(8, 2) };
        var down = new Button { Content = "▼", IsEnabled = index < _files.Count - 1, Padding = new Thickness(8, 2) };
        ToolTip.SetTip(up, "Move earlier");
        ToolTip.SetTip(down, "Move later");
        up.Click += (_, _) => Move(index, index - 1);
        down.Click += (_, _) => Move(index, index + 1);

        var details = new List<string>();

        // When it began, which is not always what the file says: some writers stamp the end.
        if (_proposal.StartOf(file) is { } recorded)
        {
            details.Add($"recorded {recorded.ToLocalTime():MMM d, h:mm:ss tt}");
        }
        else if (file.LastSaved is { } saved)
        {
            details.Add($"saved {saved.ToLocalTime():MMM d, h:mm tt}");
        }

        if (file.Track is { } track)
        {
            details.Add($"part {track}");
        }

        if (file.DurationSeconds is { } seconds)
        {
            details.Add(Length(seconds));
        }

        var text = new StackPanel
        {
            Spacing = 1,
            VerticalAlignment = VerticalAlignment.Center,
            Children =
            {
                new TextBlock
                {
                    Text = $"{index + 1}.  {Path.GetFileName(file.Path)}",
                    TextTrimming = TextTrimming.CharacterEllipsis,
                },
                Quiet(new TextBlock { Text = "      " + string.Join(" · ", details), FontSize = 11.5 }),
            },
        };

        var row = new Grid { ColumnDefinitions = new ColumnDefinitions("*,Auto,Auto"), ColumnSpacing = 4, Margin = new Thickness(0, 3) };
        row.Children.Add(text);
        Grid.SetColumn(up, 1);
        Grid.SetColumn(down, 2);
        row.Children.Add(up);
        row.Children.Add(down);
        return row;
    }

    private void Move(int from, int to)
    {
        (_files[from], _files[to]) = (_files[to], _files[from]);
        _moved = true;
        Fill();
    }

    private string TotalLength() =>
        _files.All(f => f.DurationSeconds is not null)
            ? $" {Length(_files.Sum(f => f.DurationSeconds!.Value))} in all."
            : string.Empty;

    private static string Length(double seconds) =>
        RunTimes.Clock(seconds);

    private static TextBlock Quiet(TextBlock text)
    {
        text[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("SecondaryText");
        return text;
    }
}
