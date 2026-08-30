"""Generate the LocalScribe application icon.

Drawn at 1024 and downsampled, because an icon that is legible at 16 pixels is a different
drawing from one that merely looks right at 256. The mark is a speech waveform: bars that rise
and fall, which is what the app does and what nothing else on the taskbar looks like. A
microphone would collide with every recorder ever shipped, and a document with every editor.
"""

from PIL import Image, ImageDraw

# Deep slate ground with a warm accent, so the mark carries on both light and dark taskbars.
GROUND_TOP = (28, 36, 54)
GROUND_BOTTOM = (17, 22, 34)
ACCENT = (86, 204, 242)
ACCENT_DIM = (58, 145, 200)

S = 1024
PAD = S * 0.14


def rounded_ground(size: int) -> Image.Image:
    """The tile: a vertical gradient inside a squircle-ish rounded rectangle."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    gradient = Image.new("RGBA", (size, size))
    painter = ImageDraw.Draw(gradient)
    for y in range(size):
        t = y / max(1, size - 1)
        painter.line(
            [(0, y), (size, y)],
            fill=tuple(
                round(a + (b - a) * t)
                for a, b in zip(GROUND_TOP, GROUND_BOTTOM)
            )
            + (255,),
        )

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=round(size * 0.22), fill=255
    )

    image.paste(gradient, (0, 0), mask)
    return image


def waveform(image: Image.Image) -> None:
    """Seven bars, tallest in the middle, with rounded caps."""
    painter = ImageDraw.Draw(image)

    # Relative heights. Asymmetric on purpose: a symmetric set reads as a graphic equaliser
    # rather than as speech.
    heights = [0.30, 0.58, 0.86, 1.00, 0.72, 0.44, 0.24]

    span = S - (2 * PAD)
    slot = span / len(heights)
    bar = slot * 0.46
    radius = bar / 2
    centre = S / 2
    tallest = (S - (2 * PAD)) * 0.5

    for index, height in enumerate(heights):
        x = PAD + (slot * index) + ((slot - bar) / 2)
        half = tallest * height

        # The trailing bars fade, which suggests speech continuing rather than a fixed pattern.
        colour = ACCENT if height > 0.4 else ACCENT_DIM

        painter.rounded_rectangle(
            [x, centre - half, x + bar, centre + half],
            radius=radius,
            fill=colour + (255,),
        )


def build() -> Image.Image:
    image = rounded_ground(S)
    waveform(image)
    return image


def iconset(master: Image.Image, name: str) -> None:
    """The sizes iconutil wants, as a .iconset directory ready for `iconutil -c icns`.

    The drawing is the Windows tile, unchanged — kinship across platforms is the point — but
    inset to 82% of the canvas, because macOS icons live inside a transparent margin and a
    full-bleed tile reads as oversized next to every other icon in the Dock.
    """
    import os

    directory = f"{name}.iconset"
    os.makedirs(directory, exist_ok=True)

    inset = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    content = round(S * 0.82)
    offset = (S - content) // 2
    inset.paste(master.resize((content, content), Image.LANCZOS), (offset, offset))

    # Each frame from the master, never from each other — the same discipline as the .ico.
    for points in [16, 32, 128, 256, 512]:
        for scale in [1, 2]:
            pixels = points * scale
            suffix = "@2x" if scale == 2 else ""
            inset.resize((pixels, pixels), Image.LANCZOS).save(
                f"{directory}/icon_{points}x{points}{suffix}.png"
            )

    print(f"wrote {directory}; run: iconutil -c icns {directory}")


def main() -> None:
    import sys

    master = build()

    if "--iconset" in sys.argv:
        iconset(master, "localscribe")
        return

    out_png = "localscribe.png"
    master.resize((512, 512), Image.LANCZOS).save(out_png)

    # Every size Windows actually asks for. Explicit rather than letting Pillow pick, so the
    # 16 and 20 pixel variants are downsampled from the master rather than from each other.
    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    frames = [master.resize((s, s), Image.LANCZOS) for s in sizes]

    frames[-1].save(
        "localscribe.ico",
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=frames[:-1],
    )

    print("wrote localscribe.ico and localscribe.png")


if __name__ == "__main__":
    main()
