using LocalScribe.Core.Audio;
using LocalScribe.Core.Pipeline;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;

namespace LocalScribe.App;

/// <summary>
/// The order several files will be joined in, shown before anything is transcribed: what decided
/// it, anything that argues otherwise, and a way to move a file. The order is proposed from what
/// the files say about themselves (<see cref="RecordingOrder"/>), but the person who made the
/// recordings knows, and a wrong order costs a whole run to discover. The Mac's sheet, as a
/// ContentDialog. Null means cancel.
/// </summary>
internal sealed class OrderDialog
{
    private readonly List<RecordingFacts> _files;
    private readonly RecordingOrder.Proposal _proposal;
    private readonly StackPanel _rows = new() { Spacing = 2 };
    private readonly TextBlock _reason = Quiet(new TextBlock { TextWrapping = TextWrapping.Wrap });
    private bool _moved;

    private OrderDialog(RecordingOrder.Proposal proposal)
    {
        _proposal = proposal;
        _files = [.. proposal.Ordered];
    }

    internal static async Task<IReadOnlyList<string>?> AskAsync(XamlRoot root, RecordingOrder.Proposal proposal)
    {
        var sheet = new OrderDialog(proposal);
        var dialog = new ContentDialog
        {
            XamlRoot = root,
            Title = "Transcribe as one recording",
            Content = sheet.Layout(),
            PrimaryButtonText = "Transcribe as One",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Primary,
        };

        return await dialog.ShowAsync() == ContentDialogResult.Primary
            ? [.. sheet._files.Select(f => f.Path)]
            : null;
    }

    private StackPanel Layout()
    {
        var layout = new StackPanel { Spacing = 12 };

        layout.Children.Add(new TextBlock
        {
            Text = $"These {_files.Count} files will be joined into one recording and transcribed in this order.",
            TextWrapping = TextWrapping.Wrap,
        });

        layout.Children.Add(_reason);

        foreach (var doubt in _proposal.Doubts)
        {
            layout.Children.Add(new TextBlock
            {
                Text = "⚠ " + doubt,
                TextWrapping = TextWrapping.Wrap,
                Foreground = (Brush)Application.Current.Resources["SystemFillColorCautionBrush"],
            });
        }

        layout.Children.Add(new ScrollViewer { Content = _rows, MaxHeight = 360 });
        Fill();
        return layout;
    }

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
                    FontSize = 12,
                    FontStyle = Windows.UI.Text.FontStyle.Italic,
                }));
            }

            _rows.Children.Add(Row(i));
        }
    }

    private Grid Row(int index)
    {
        var file = _files[index];

        var up = new Button { Content = "▲", IsEnabled = index > 0, Padding = new Thickness(8, 2, 8, 2) };
        var down = new Button { Content = "▼", IsEnabled = index < _files.Count - 1, Padding = new Thickness(8, 2, 8, 2) };
        ToolTipService.SetToolTip(up, "Move earlier");
        ToolTipService.SetToolTip(down, "Move later");
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

        var text = new StackPanel { Spacing = 1, VerticalAlignment = VerticalAlignment.Center };
        text.Children.Add(new TextBlock
        {
            Text = $"{index + 1}.  {Path.GetFileName(file.Path)}",
            TextTrimming = TextTrimming.CharacterEllipsis,
        });
        text.Children.Add(Quiet(new TextBlock { Text = "      " + string.Join(" · ", details), FontSize = 12 }));

        var row = new Grid { ColumnSpacing = 4, Margin = new Thickness(0, 3, 0, 3) };
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
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
        text.Foreground = (Brush)Application.Current.Resources["TextFillColorSecondaryBrush"];
        return text;
    }
}
