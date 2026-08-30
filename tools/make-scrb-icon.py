"""Generate the icon for saved .scrb transcripts.

A file icon is a different drawing from an app icon: it has to read as "a document belonging to
that app" in a folder full of other documents, at 16 pixels, next to the app's own tile. So the
mark is the app's waveform carried on a page — the universal document silhouette with a folded
corner, in the app's slate, with the wave in the app's accent. No text lines: they turn to mud
below 32 pixels and say nothing the wave does not.

Same discipline as make-icon.py: drawn once at 1024 and downsampled to every size Windows asks
for, each frame from the master rather than from each other.
"""

from PIL import Image, ImageDraw

# The app icon's palette, so the file reads as kin to the tile beside it.
PAGE_TOP = (28, 36, 54)
PAGE_BOTTOM = (17, 22, 34)
FOLD = (46, 58, 82)
ACCENT = (86, 204, 242)
ACCENT_DIM = (58, 145, 200)

S = 1024


def page(size: int) -> tuple[Image.Image, float, float, float, float]:
    """A portrait page with a folded corner; returns the image and the page bounds."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # Portrait proportions inside the square canvas.
    width = size * 0.66
    height = size * 0.86
    left = (size - width) / 2
    top = (size - height) / 2
    right = left + width
    bottom = top + height
    fold = width * 0.30
    radius = round(size * 0.045)

    gradient = Image.new("RGBA", (size, size))
    painter = ImageDraw.Draw(gradient)
    for y in range(size):
        t = y / max(1, size - 1)
        painter.line(
            [(0, y), (size, y)],
            fill=tuple(
                round(a + (b - a) * t)
                for a, b in zip(PAGE_TOP, PAGE_BOTTOM)
            )
            + (255,),
        )

    mask = Image.new("L", (size, size), 0)
    shape = ImageDraw.Draw(mask)
    shape.rounded_rectangle([left, top, right, bottom], radius=radius, fill=255)

    # The folded corner: cut the top-right triangle off the page...
    shape.polygon(
        [(right - fold, top - 1), (right + 1, top - 1), (right + 1, top + fold)],
        fill=0,
    )

    image.paste(gradient, (0, 0), mask)

    # ...and draw the fold itself as a lighter triangle, the way every OS draws one.
    ImageDraw.Draw(image).polygon(
        [(right - fold, top), (right - fold, top + fold), (right, top + fold)],
        fill=FOLD + (255,),
    )

    return image, left, top, right, bottom


def waveform(image: Image.Image, left: float, top: float, right: float, bottom: float) -> None:
    """The app's seven bars, sized to sit in the lower two thirds of the page."""
    painter = ImageDraw.Draw(image)

    # The same asymmetric heights as the application icon; kinship is the point.
    heights = [0.30, 0.58, 0.86, 1.00, 0.72, 0.44, 0.24]

    pad = (right - left) * 0.16
    span = (right - left) - (2 * pad)
    slot = span / len(heights)
    bar = slot * 0.46
    radius = bar / 2

    # Below the fold, so the two features never fight at small sizes.
    centre = top + (bottom - top) * 0.58
    tallest = (bottom - top) * 0.30

    for index, height in enumerate(heights):
        x = left + pad + (slot * index) + ((slot - bar) / 2)
        half = tallest * height

        colour = ACCENT if height > 0.4 else ACCENT_DIM

        painter.rounded_rectangle(
            [x, centre - half, x + bar, centre + half],
            radius=radius,
            fill=colour + (255,),
        )


def build() -> Image.Image:
    image, left, top, right, bottom = page(S)
    waveform(image, left, top, right, bottom)
    return image


def iconset(master: Image.Image, name: str) -> None:
    """The sizes iconutil wants; see make-icon.py for the reasoning. The page keeps its own
    proportions — documents already sit inside whitespace, so no extra inset is added."""
    import os

    directory = f"{name}.iconset"
    os.makedirs(directory, exist_ok=True)

    for points in [16, 32, 128, 256, 512]:
        for scale in [1, 2]:
            pixels = points * scale
            suffix = "@2x" if scale == 2 else ""
            master.resize((pixels, pixels), Image.LANCZOS).save(
                f"{directory}/icon_{points}x{points}{suffix}.png"
            )

    print(f"wrote {directory}; run: iconutil -c icns {directory}")


def main() -> None:
    import sys

    master = build()

    if "--iconset" in sys.argv:
        iconset(master, "scrb")
        return

    out_png = "scrb.png"
    master.resize((512, 512), Image.LANCZOS).save(out_png)

    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    frames = [master.resize((s, s), Image.LANCZOS) for s in sizes]

    frames[-1].save(
        "scrb.ico",
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=frames[:-1],
    )

    print("wrote scrb.ico and scrb.png")


if __name__ == "__main__":
    main()
