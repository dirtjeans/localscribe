using Avalonia.Controls;
using Avalonia.Layout;

namespace LocalScribe.Desktop;

/// <summary>One question, one text box, OK or cancel. Null means the user changed their mind.</summary>
internal sealed class InputDialog : Window
{
    private InputDialog(string title, string prompt, string initial)
    {
        Title = title;
        Width = 380;
        SizeToContent = SizeToContent.Height;
        CanResize = false;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;

        var box = new TextBox { Text = initial, Watermark = prompt };
        var ok = new Button { Content = "OK", IsDefault = true };
        var cancel = new Button { Content = "Cancel", IsCancel = true };

        ok.Click += (_, _) => Close(box.Text);
        cancel.Click += (_, _) => Close(null);

        Content = new StackPanel
        {
            Margin = new Avalonia.Thickness(20),
            Spacing = 12,
            Children =
            {
                new TextBlock { Text = prompt, TextWrapping = Avalonia.Media.TextWrapping.Wrap },
                box,
                new StackPanel
                {
                    Orientation = Orientation.Horizontal,
                    HorizontalAlignment = HorizontalAlignment.Right,
                    Spacing = 8,
                    Children = { cancel, ok },
                },
            },
        };

        Opened += (_, _) =>
        {
            box.Focus();
            box.SelectAll();
        };
    }

    internal static async Task<string?> AskAsync(Window owner, string title, string prompt, string initial = "") =>
        await new InputDialog(title, prompt, initial).ShowDialog<string?>(owner);
}
