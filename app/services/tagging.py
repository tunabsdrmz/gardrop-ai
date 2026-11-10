"""Zero-shot tagging: compare the photo with short descriptions and pick the closest.

No training needed. To support a new label, add a line of text.
"""

import numpy as np
from PIL import Image

from app.schemas import Prediction, TagResponse
from app.services.colors import dominant_colors
from app.services.embedder import Embedder

# Several phrasings per label make the result more stable.
TAXONOMY: dict[str, dict[str, list[str]]] = {
    "category": {
        "top": ["a photo of a t-shirt", "a photo of a shirt", "a photo of a sweater",
                "a photo of a blouse"],
        "bottom": ["a photo of jeans", "a photo of trousers", "a photo of a skirt",
                   "a photo of shorts"],
        "outerwear": ["a photo of a coat", "a photo of a jacket", "a photo of a blazer"],
        "shoes": ["a photo of shoes", "a photo of sneakers", "a photo of boots"],
        "dress": ["a photo of a dress", "a photo of a gown"],
    },
    "pattern": {
        "solid": ["a plain solid-colored garment"],
        "striped": ["a striped garment"],
        "plaid": ["a plaid or checked garment"],
        "floral": ["a garment with a floral print"],
        "dotted": ["a polka dot garment"],
        "graphic": ["a garment with a graphic print or a logo"],
    },
    "season": {
        "summer": ["light summer clothing", "thin breathable clothing for hot weather"],
        "winter": ["warm winter clothing", "thick wool or padded clothing for cold weather"],
        "all-season": ["everyday clothing for mild weather"],
    },
    "formality": {
        "sporty": ["sportswear", "athletic clothing"],
        "casual": ["casual everyday clothing"],
        "smart-casual": ["smart casual clothing"],
        "formal": ["formal business or evening wear"],
    },
}


def classify(
    image_vector: np.ndarray, labels: dict[str, list[str]], embedder: Embedder
) -> Prediction:
    names = list(labels)
    label_vectors = []
    for name in names:
        mean = embedder.embed_texts(labels[name]).mean(axis=0)
        label_vectors.append(mean / (np.linalg.norm(mean) + 1e-12))

    # Cosine similarity, scaled like CLIP does, then softmax into probabilities.
    logits = 100 * np.stack(label_vectors) @ image_vector
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()

    best = int(np.argmax(probabilities))
    return Prediction(label=names[best], confidence=round(float(probabilities[best]), 3))


def tag(image: Image.Image, embedder: Embedder) -> TagResponse:
    vector = embedder.embed_image(image)
    return TagResponse(
        category=classify(vector, TAXONOMY["category"], embedder),
        pattern=classify(vector, TAXONOMY["pattern"], embedder),
        season=classify(vector, TAXONOMY["season"], embedder),
        formality=classify(vector, TAXONOMY["formality"], embedder),
        colors=dominant_colors(image),
        embedding=[round(float(v), 5) for v in vector],
    )
