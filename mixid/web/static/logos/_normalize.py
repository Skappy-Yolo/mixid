"""One-off: normalize raw platform logos to 28x28 transparent PNGs.

Run once: `python _normalize.py` from this folder. Deletes the _src_* files
afterward, leaving only the final youtube.png / soundcloud.png / mixcloud.png
/ audiomack.png / spotify.png assets that index.html references.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
TARGET = 56  # 2x retina; html renders at 14x14 or 28x28


def trim_white_background(im: Image.Image, threshold: int = 240) -> Image.Image:
    """JPGs come on a white canvas. Make near-white pixels transparent."""
    im = im.convert("RGBA")
    px = im.load()
    w, h = im.size
    for y in range(h):
        for x in range(w):
            r, g, b, _ = px[x, y]
            if r >= threshold and g >= threshold and b >= threshold:
                px[x, y] = (255, 255, 255, 0)
    return im


def crop_to_content(im: Image.Image) -> Image.Image:
    bbox = im.getbbox()
    return im.crop(bbox) if bbox else im


def fit_square(im: Image.Image, size: int) -> Image.Image:
    im.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    x = (size - im.width) // 2
    y = (size - im.height) // 2
    canvas.paste(im, (x, y), im)
    return canvas


JOBS = [
    ("_src_youtube.png", "youtube.png", False),
    ("_src_soundcloud.webp", "soundcloud.png", False),
    ("_src_mixcloud.webp", "mixcloud.png", False),
    ("_src_audiomack.jpg", "audiomack.png", True),
    ("_src_spotify.png", "spotify.png", False),
]


def main() -> None:
    for src_name, dst_name, needs_white_trim in JOBS:
        src = HERE / src_name
        dst = HERE / dst_name
        if not src.exists():
            print(f"SKIP missing source: {src_name}")
            continue
        im = Image.open(src)
        if needs_white_trim:
            im = trim_white_background(im)
        else:
            im = im.convert("RGBA")
        im = crop_to_content(im)
        im = fit_square(im, TARGET)
        im.save(dst, "PNG", optimize=True)
        print(f"OK   {src_name} -> {dst_name} ({dst.stat().st_size} bytes)")

    # Clean up source files so the folder is shippable
    for src_name, _, _ in JOBS:
        p = HERE / src_name
        if p.exists():
            p.unlink()
            print(f"RM   {src_name}")


if __name__ == "__main__":
    main()
