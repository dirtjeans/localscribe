using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;

namespace LocalScribe.Desktop;

/// <summary>
/// The recording as peaks, with the playback position drawn over it. Click or drag to seek:
/// the drag is live — each move restarts playback at the pointer — because that is what makes
/// "drag the waveform and watch the words follow" true rather than approximately true.
/// </summary>
internal sealed class WaveformView : Control
{
    private float[] _peaks = [];
    private double _durationSeconds;
    private double _positionSeconds = -1;
    private bool _dragging;

    private static readonly IBrush PeakBrush = new SolidColorBrush(Color.FromArgb(120, 120, 144, 180));
    private static readonly IPen CursorPen = new Pen(new SolidColorBrush(Color.FromArgb(220, 64, 128, 255)), 2);

    /// <summary>Raised when the user asks to hear from a time, by click or drag.</summary>
    public event Action<double>? SeekRequested;

    /// <summary>Hands the view its audio. Peaks are recomputed on resize by asking again.</summary>
    public Func<int, float[]>? PeakSource { get; set; }

    public void SetAudio(double durationSeconds)
    {
        _durationSeconds = durationSeconds;
        _peaks = [];
        InvalidateVisual();
    }

    public void SetPosition(double seconds)
    {
        _positionSeconds = seconds;
        InvalidateVisual();
    }

    public override void Render(DrawingContext context)
    {
        var width = (int)Bounds.Width;
        var height = Bounds.Height;

        if (width <= 0 || height <= 0 || _durationSeconds <= 0 || PeakSource is null)
        {
            return;
        }

        // One bucket per two pixels reads as a waveform rather than a comb.
        var buckets = Math.Max(16, width / 2);

        if (_peaks.Length != buckets)
        {
            _peaks = PeakSource(buckets);
        }

        var middle = height / 2;

        for (var i = 0; i < _peaks.Length; i++)
        {
            var x = i * Bounds.Width / _peaks.Length;
            var half = Math.Max(0.5, _peaks[i] * middle);

            context.FillRectangle(
                PeakBrush,
                new Rect(x, middle - half, Math.Max(1, Bounds.Width / _peaks.Length - 1), half * 2));
        }

        if (_positionSeconds >= 0)
        {
            var x = Math.Clamp(_positionSeconds / _durationSeconds, 0, 1) * Bounds.Width;
            context.DrawLine(CursorPen, new Point(x, 0), new Point(x, height));
        }
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        _dragging = true;
        Seek(e);
        e.Pointer.Capture(this);
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        if (_dragging)
        {
            Seek(e);
        }
    }

    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        _dragging = false;
        e.Pointer.Capture(null);
    }

    protected override void OnSizeChanged(SizeChangedEventArgs e)
    {
        base.OnSizeChanged(e);
        _peaks = [];
        InvalidateVisual();
    }

    private void Seek(PointerEventArgs e)
    {
        if (_durationSeconds <= 0 || Bounds.Width <= 0)
        {
            return;
        }

        var fraction = Math.Clamp(e.GetPosition(this).X / Bounds.Width, 0, 1);
        SeekRequested?.Invoke(fraction * _durationSeconds);
    }
}
