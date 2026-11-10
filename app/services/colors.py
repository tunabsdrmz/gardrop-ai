"""Main colors of a garment, and how well two colors go together."""

import colorsys

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from app.schemas import ColorSwatch

# A small fashion palette; every detected color gets the nearest name.
NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 22),
    "white": (245, 245, 242),
    "gray": (128, 128, 128),
    "navy": (31, 42, 68),
    "blue": (52, 101, 164),
    "light blue": (150, 190, 230),
    "red": (190, 30, 45),
    "burgundy": (107, 31, 51),
    "pink": (230, 160, 180),
    "orange": (230, 120, 40),
    "yellow": (240, 200, 60),
    "beige": (225, 210, 185),
    "camel": (196, 162, 122),
    "brown": (100, 70, 45),
    "olive": (110, 115, 70),
    "green": (50, 130, 70),
    "purple": (110, 70, 140),
}

NEUTRAL_NAMES = {"black", "white", "gray", "navy", "beige", "camel", "brown"}


def to_hex(rgb) -> str:
    r, g, b = (int(round(max(0, min(255, c)))) for c in rgb)
    return f"{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def nearest_name(rgb) -> str:
    def distance(name: str) -> float:
        return sum((a - b) ** 2 for a, b in zip(NAMED_COLORS[name], rgb, strict=True))

    return min(NAMED_COLORS, key=distance)


def _foreground_pixels(image: Image.Image) -> np.ndarray:
    """Pixels that belong to the garment, not the background."""
    small = image.convert("RGBA")
    small.thumbnail((128, 128))
    grid = np.asarray(small).astype(float)
    pixels = grid.reshape(-1, 4)

    if (pixels[:, 3] < 250).any():
        # A cut-out PNG (like the ones the iOS app makes): keep the opaque part.
        mask = pixels[:, 3] > 128
    else:
        # A normal photo: guess the background from the border and drop pixels close to it.
        border = np.concatenate([grid[0, :, :3], grid[-1, :, :3], grid[:, 0, :3], grid[:, -1, :3]])
        background = np.median(border, axis=0)
        mask = np.linalg.norm(pixels[:, :3] - background, axis=1) > 40
        if mask.sum() < 50:  # the garment fills the whole photo
            mask = np.ones(len(pixels), dtype=bool)
    return pixels[mask, :3]


def dominant_colors(image: Image.Image, max_colors: int = 4) -> list[ColorSwatch]:
    """Groups similar pixels with k-means and returns the biggest groups."""
    pixels = _foreground_pixels(image)
    if len(pixels) == 0:
        return []

    k = min(max_colors, len(np.unique(pixels, axis=0)))
    kmeans = KMeans(n_clusters=k, n_init=4, random_state=0).fit(pixels)
    counts = np.bincount(kmeans.labels_, minlength=k)

    swatches = []
    for index in np.argsort(counts)[::-1]:
        ratio = counts[index] / counts.sum()
        if ratio < 0.05:  # ignore tiny details like buttons
            continue
        rgb = kmeans.cluster_centers_[index]
        swatches.append(
            ColorSwatch(hex=to_hex(rgb), name=nearest_name(rgb), ratio=round(float(ratio), 3))
        )
    return swatches


def is_neutral(hex_color: str) -> bool:
    rgb = hex_to_rgb(hex_color)
    _, saturation, value = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    return saturation < 0.18 or value < 0.18 or nearest_name(rgb) in NEUTRAL_NAMES


def hue_degrees(hex_color: str) -> float:
    hue, _, _ = colorsys.rgb_to_hsv(*(c / 255 for c in hex_to_rgb(hex_color)))
    return hue * 360


def color_harmony(a: str, b: str) -> float:
    """0...1. Neutrals go with everything; similar or opposite hues work; the rest clash a bit."""
    if is_neutral(a) or is_neutral(b):
        return 1.0
    difference = abs(hue_degrees(a) - hue_degrees(b)) % 360
    difference = min(difference, 360 - difference)
    if difference <= 30:
        return 0.9  # analogous: blue + light blue
    if difference >= 150:
        return 0.75  # complementary: blue + orange
    return 0.45
