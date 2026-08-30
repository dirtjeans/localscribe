using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;

namespace LocalScribe.Desktop;

/// <summary>
/// The recording as peaks, with the playback position drawn over it. A tap plays from that
/// spot; a drag scrubs — the marker and the words follow the pointer live, with the sound
/// held until release, which is what makes "drag the waveform and watch the words follow"
/// feel navigable rather than like a stuttering restart per pixel.
/// </summary>
internal sealed class WaveformView : Control
{
    private float[] _peaks = [];
    private double _durationSeconds;
    private double _positionSeconds = -1;
    private bool _dragging;

    private static readonly IBrush PeakBrush = new SolidColorBrush(Color.FromArgb(120, 120, 144, 180));
    private static readonly IPen CursorPen = new Pen(new SolidColorBrush(Color.FromArgb(220, 64, 128, 255)), 2);

    /// <summary>A press-and-release without movement: play from here.</summary>
    public event Action<double>? TapSeek;

    /// <summary>A drag has begun; the owner decides what pauses while it lasts.</summary>
    public event Action? ScrubStarted;

    /// <summary>The pointer, mid-drag. Raised per move — the transcript follows live.</summary>
    public event Action<double>? Scrubbed;

    /// <summary>The drag ended here; the owner decides whether sound resumes.</summary>
    public event Action<double>? ScrubEnded;

    private Point _pressedAt;
    private bool _scrubbing;

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

    /// <summary>
    /// A press is ambiguous until the pointer moves: held still it is a tap (play from
    /// here), moved it is a scrub. Four pixels tells them apart — small enough that a scrub
    /// feels immediate, large enough that a click with an unsteady hand stays a click.
    /// </summary>
    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        _dragging = true;
        _scrubbing = false;
        _pressedAt = e.GetPosition(this);
        e.Pointer.Capture(this);
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        if (!_dragging)
        {
            return;
        }

        if (!_scrubbing && Math.Abs(e.GetPosition(this).X - _pressedAt.X) > 4)
        {
            _scrubbing = true;
            ScrubStarted?.Invoke();
        }

        if (_scrubbing)
        {
            Scrubbed?.Invoke(TimeAt(e));
        }
    }

    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        e.Pointer.Capture(null);

        if (!_dragging)
        {
            return;
        }

        _dragging = false;

        if (_scrubbing)
        {
            _scrubbing = false;
            ScrubEnded?.Invoke(TimeAt(e));
        }
        else
        {
            TapSeek?.Invoke(TimeAt(e));
        }
    }

    protected override void OnSizeChanged(SizeChangedEventArgs e)
    {
        base.OnSizeChanged(e);
        _peaks = [];
        InvalidateVisual();
    }

    private double TimeAt(PointerEventArgs e)
    {
        if (_durationSeconds <= 0 || Bounds.Width <= 0)
        {
            return 0;
        }

        return Math.Clamp(e.GetPosition(this).X / Bounds.Width, 0, 1) * _durationSeconds;
    }
}
