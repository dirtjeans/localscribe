using LocalScribe.Core.Transcription;

namespace LocalScribe.Core.Diarization;

/// <summary>
/// Names the user gave speakers, held where they survive the transcript being rebuilt.
/// <para>
/// Speakers can be renamed the moment their labels first appear, and that is mid-run: every
/// window that lands, every stretch the scan times, and the final assembly all divide the
/// transcript between speakers afresh. A name written onto the segments on screen would be
/// gone at the next of those. So a name is kept against the diarizer's own label for the
/// voice — which every rebuild reproduces — and put back after renumbering each time; a name
/// for one part is kept against its stretch of time, which the rebuilds keep too.
/// </para>
/// </summary>
public sealed class SpeakerNames
{
    private readonly Dictionary<string, string> _byVoice = new(StringComparer.Ordinal);
    private readonly List<(double Start, double End, string Name)> _byPart = [];
    private Dictionary<string, string> _voiceOfShown = new(StringComparer.Ordinal);

    /// <summary>True when nothing has been named.</summary>
    public bool IsEmpty => _byVoice.Count == 0 && _byPart.Count == 0;

    /// <summary>
    /// Names every segment of the voice shown under <paramref name="shownLabel"/>, in this
    /// labelling and every later one. False when that label came from no voice — nothing
    /// shown has been labelled through <see cref="Apply"/> yet.
    /// </summary>
    public bool NameVoice(string shownLabel, string name)
    {
        if (!_voiceOfShown.TryGetValue(shownLabel, out var voice))
        {
            return false;
        }

        _byVoice[voice] = name;
        return true;
    }

    /// <summary>Names whatever is spoken between two moments, in this labelling and every later one.</summary>
    public void NamePart(double startSeconds, double endSeconds, string name) =>
        _byPart.Add((startSeconds, endSeconds, name));

    /// <summary>Forgets every name: a new recording, or a new speaker search whose voices are not these.</summary>
    public void Clear()
    {
        _byVoice.Clear();
        _byPart.Clear();
        _voiceOfShown = new(StringComparer.Ordinal);
    }

    /// <summary>
    /// Puts the names back onto a fresh labelling.
    /// </summary>
    /// <param name="byVoice">The segments as the diarizer labelled them, before renumbering.</param>
    /// <param name="renumbered">The same segments, one for one, after renumbering by appearance.</param>
    public IReadOnlyList<TranscriptSegment> Apply(
        IReadOnlyList<TranscriptSegment> byVoice,
        IReadOnlyList<TranscriptSegment> renumbered)
    {
        ArgumentNullException.ThrowIfNull(byVoice);
        ArgumentNullException.ThrowIfNull(renumbered);

        if (byVoice.Count != renumbered.Count)
        {
            // Not one for one, so no segment's voice can be read off its neighbour. Unnamed
            // is honest; a name on the wrong voice is not.
            _voiceOfShown = new(StringComparer.Ordinal);
            return renumbered;
        }

        var shownVoices = new Dictionary<string, string>(StringComparer.Ordinal);
        var named = new List<TranscriptSegment>(renumbered.Count);

        for (var i = 0; i < renumbered.Count; i++)
        {
            var segment = renumbered[i];
            var voice = byVoice[i].Speaker;

            if (voice is not null && _byVoice.TryGetValue(voice, out var voiceName))
            {
                segment = segment with { Speaker = voiceName };
            }

            var middle = (segment.StartSeconds + segment.EndSeconds) / 2;

            // Latest first: renaming the same part twice means the second name.
            for (var p = _byPart.Count - 1; p >= 0; p--)
            {
                if (middle >= _byPart[p].Start && middle <= _byPart[p].End)
                {
                    segment = segment with { Speaker = _byPart[p].Name };
                    break;
                }
            }

            if (voice is not null && segment.Speaker is { } shown)
            {
                shownVoices.TryAdd(shown, voice);
            }

            named.Add(segment);
        }

        _voiceOfShown = shownVoices;
        return named;
    }
}
