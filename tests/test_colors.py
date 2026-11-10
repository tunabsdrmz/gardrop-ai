from PIL import Image, ImageDraw

from app.services.colors import color_harmony, dominant_colors, is_neutral


def test_single_color_image():
    swatches = dominant_colors(Image.new("RGB", (64, 64), (190, 30, 45)))

    assert swatches[0].name == "red"
    assert swatches[0].ratio == 1.0


def test_cutout_ignores_transparent_background():
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((16, 16, 48, 48), fill=(52, 101, 164, 255))

    assert dominant_colors(image)[0].name == "blue"


def test_photo_ignores_plain_background():
    image = Image.new("RGB", (64, 64), (245, 245, 242))
    ImageDraw.Draw(image).rectangle((20, 10, 44, 54), fill=(110, 115, 70))

    assert dominant_colors(image)[0].name == "olive"


def test_neutrals_go_with_everything():
    assert is_neutral("1F2A44")  # navy
    assert color_harmony("1F2A44", "E67828") == 1.0


def test_similar_hues_beat_clashing_hues():
    analogous = color_harmony("3465A4", "96BEE6")  # blue + light blue
    clashing = color_harmony("BE1E2D", "32824A")  # red + green, 120° apart
    assert analogous > clashing
