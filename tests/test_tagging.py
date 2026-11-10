from PIL import Image

from app.services.tagging import tag
from tests.conftest import png_bytes


def test_tag_picks_closest_labels(fake_embedder):
    result = tag(Image.new("RGB", (32, 32), (63, 86, 114)), fake_embedder)

    assert result.category.label == "bottom"
    assert result.pattern.label == "solid"
    assert result.season.label == "winter"
    assert result.formality.label == "casual"
    assert result.category.confidence > 0.9
    assert len(result.embedding) == 4


def test_tag_endpoint(client):
    response = client.post("/tag", files={"file": ("jeans.png", png_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["category"]["label"] == "bottom"
    assert body["colors"][0]["name"] == "blue"


def test_tag_rejects_non_images(client):
    response = client.post("/tag", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 415


def test_tag_rejects_broken_images(client):
    response = client.post("/tag", files={"file": ("broken.png", b"not a png", "image/png")})

    assert response.status_code == 400
