using Avalonia.Controls;
using Avalonia.Layout;

namespace LocalScribe.Desktop;

/// <summary>
/// Save / Discard / Cancel over unsaved work. Built in code because it is three buttons and a
/// sentence; the shape it must keep is the contract: cancel means the work stays, and nothing
/// about closing proceeds until the user has chosen.
/// </summary>
internal sealed class CloseGateDialog : Window
{
    internal enum Choice
    {
        Cancel,
        Save,
        Discard,
    }

    private CloseGateDialog()
    {
        Title = "Unsaved transcript";
        Width = 420;
        SizeToContent = SizeToContent.Height;
        CanResize = false;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;

        var save = new Button { Content = "Save…", IsDefault = true };
        var discard = new Button { Content = "Discard" };
        var cancel = new Button { Content = "Cancel", IsCancel = true };

        save.Click += (_, _) => Close(Choice.Save);
        discard.Click += (_, _) => Close(Choice.Discard);
        cancel.Click += (_, _) => Close(Choice.Cancel);

        Content = new StackPanel
        {
            Margin = new Avalonia.Thickness(20),
            Spacing = 16,
            Children =
            {
                new TextBlock
                {
                    Text = "This transcript exists nowhere else yet. Closing without saving "
                        + "loses it — and a recording made here loses its audio too.",
                    TextWrapping = Avalonia.Media.TextWrapping.Wrap,
                },
                new StackPanel
                {
                    Orientation = Orientation.Horizontal,
                    HorizontalAlignment = HorizontalAlignment.Right,
                    Spacing = 8,
                    Children = { cancel, discard, save },
                },
            },
        };
    }

    internal static async Task<Choice> AskAsync(Window owner) =>
        await new CloseGateDialog().ShowDialog<Choice>(owner);
}
