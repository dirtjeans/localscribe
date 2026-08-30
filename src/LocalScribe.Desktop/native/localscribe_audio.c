/*
 * The audio layer the handoff prescribes: miniaudio, one C file, compiled once for arm64 and
 * P/Invoked. It replaces NAudio playback (TranscriptPlayer) and WASAPI capture
 * (MicrophoneCapture) with the smallest surface those two actually need.
 *
 * Position is counted at the device callback, then reported one device period behind. That is
 * the closest thing miniaudio exposes to "what the speaker has actually said": frames handed
 * to the callback are consumed within a period (about ten milliseconds here), where the
 * Windows lesson was about a WASAPI buffer holding several hundred milliseconds of unheard
 * audio. The clock diagnostic (localscribe-clock.txt) stays the judge: its gap column should
 * hold steady at about one period.
 */

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_ENGINE
#define MA_NO_NODE_GRAPH
#include "miniaudio.h"

#include <stdatomic.h>
#include <string.h>

void ls_play_stop(void);
void ls_capture_stop(void);

/* ---- playback ------------------------------------------------------------------------- */

static ma_device g_playback;
static int g_playbackInitialised = 0;

static const float* g_samples = NULL;
static ma_uint64 g_sampleCount = 0;
static _Atomic ma_uint64 g_cursor = 0;
static _Atomic int g_finished = 0;

static void playback_callback(ma_device* device, void* out, const void* in, ma_uint32 frames)
{
    (void)in;

    ma_uint64 cursor = atomic_load(&g_cursor);
    ma_uint64 available = cursor < g_sampleCount ? g_sampleCount - cursor : 0;
    ma_uint64 taking = frames < available ? frames : available;

    if (taking > 0)
    {
        memcpy(out, g_samples + cursor, (size_t)taking * sizeof(float));
        atomic_store(&g_cursor, cursor + taking);
    }

    if (taking < frames)
    {
        memset((float*)out + taking, 0, (size_t)(frames - taking) * sizeof(float));

        if (available == 0)
        {
            atomic_store(&g_finished, 1);
        }
    }

    (void)device;
}

/* Starts playing the caller's buffer from a frame offset. The buffer must stay pinned until
 * ls_play_stop returns: the callback reads it from the audio thread. */
int ls_play_start(const float* samples, ma_uint64 count, ma_uint32 sampleRate, ma_uint64 fromFrame)
{
    ls_play_stop();

    g_samples = samples;
    g_sampleCount = count;
    atomic_store(&g_cursor, fromFrame < count ? fromFrame : count);
    atomic_store(&g_finished, 0);

    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    config.sampleRate = sampleRate;
    config.dataCallback = playback_callback;

    if (ma_device_init(NULL, &config, &g_playback) != MA_SUCCESS)
    {
        return -1;
    }

    g_playbackInitialised = 1;

    if (ma_device_start(&g_playback) != MA_SUCCESS)
    {
        ma_device_uninit(&g_playback);
        g_playbackInitialised = 0;
        return -2;
    }

    return 0;
}

void ls_play_stop(void)
{
    if (g_playbackInitialised)
    {
        ma_device_uninit(&g_playback);
        g_playbackInitialised = 0;
    }

    g_samples = NULL;
    g_sampleCount = 0;
}

/* The frame the device has reached, one period behind the callback cursor so the count stays
 * on the heard side of the buffer rather than the read side. */
ma_uint64 ls_play_position(void)
{
    ma_uint64 cursor = atomic_load(&g_cursor);

    if (!g_playbackInitialised)
    {
        return cursor;
    }

    ma_uint32 period = g_playback.playback.internalPeriodSizeInFrames;

    return cursor > period ? cursor - period : 0;
}

/* 1 once the buffer has been fully handed to the device. */
int ls_play_finished(void)
{
    return atomic_load(&g_finished);
}

/* ---- capture -------------------------------------------------------------------------- */

typedef void (*ls_capture_handler)(const float* samples, ma_uint32 count);

static ma_device g_capture;
static int g_captureInitialised = 0;
static ls_capture_handler g_onCaptured = NULL;

static void capture_callback(ma_device* device, void* out, const void* in, ma_uint32 frames)
{
    (void)out;
    (void)device;

    ls_capture_handler handler = g_onCaptured;

    if (handler != NULL && frames > 0)
    {
        handler((const float*)in, frames);
    }
}

/* Starts the default microphone at the model's own rate, mono float, so the hot path has no
 * resampling stage — the same decision the WASAPI capture made. */
int ls_capture_start(ls_capture_handler handler, ma_uint32 sampleRate, ma_uint32 bufferMilliseconds)
{
    ls_capture_stop();

    g_onCaptured = handler;

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate = sampleRate;
    config.periodSizeInMilliseconds = bufferMilliseconds;
    config.dataCallback = capture_callback;

    if (ma_device_init(NULL, &config, &g_capture) != MA_SUCCESS)
    {
        g_onCaptured = NULL;
        return -1;
    }

    g_captureInitialised = 1;

    if (ma_device_start(&g_capture) != MA_SUCCESS)
    {
        ma_device_uninit(&g_capture);
        g_captureInitialised = 0;
        g_onCaptured = NULL;
        return -2;
    }

    return 0;
}

void ls_capture_stop(void)
{
    if (g_captureInitialised)
    {
        /* Uninit blocks until the audio thread is out of the callback, so after this returns
         * no further buffers arrive — the .NET side relies on that to unpin its delegate. */
        ma_device_uninit(&g_capture);
        g_captureInitialised = 0;
    }

    g_onCaptured = NULL;
}
