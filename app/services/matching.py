"""'Will this go with my closet?' Rule-based scoring, explainable to the user."""

from itertools import combinations

import numpy as np

from app.schemas import ClosetItem, MatchRequest, MatchResponse, Pairing
from app.services.colors import color_harmony, hex_to_rgb, is_neutral, nearest_name

FORMALITY_ORDER = ["sporty", "casual", "smart-casual", "formal"]

# Which categories are worn together.
PARTNERS: dict[str, set[str]] = {
    "top": {"bottom", "shoes", "outerwear"},
    "bottom": {"top", "shoes", "outerwear"},
    "shoes": {"top", "bottom", "dress", "outerwear"},
    "dress": {"shoes", "outerwear"},
    "outerwear": {"top", "bottom", "dress", "shoes"},
}

COMPATIBLE = 0.6
MAX_PER_CATEGORY = 60  # keeps outfit counting fast for big closets


def pair_score(a: ClosetItem, b: ClosetItem) -> float:
    """0...1: color harmony 50%, formality 30%, pattern 20%. Big formality gaps always fail."""
    color = color_harmony(a.colors[0], b.colors[0]) if a.colors and b.colors else 0.8
    pattern = 0.35 if a.pattern != "solid" and b.pattern != "solid" else 1.0
    gap = abs(FORMALITY_ORDER.index(a.formality) - FORMALITY_ORDER.index(b.formality))
    formality = {0: 1.0, 1: 0.75}.get(gap, 0.35)
    score = 0.5 * color + 0.2 * pattern + 0.3 * formality
    if gap >= 2:  # e.g. a hoodie with oxford shoes: never a match, whatever the colors
        score = min(score, 0.5)
    return round(score, 3)


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


def find_similar(item: ClosetItem, closet: list[ClosetItem]) -> list[Pairing]:
    """Same category and looks almost the same: probably a duplicate purchase."""
    similar = []
    for other in closet:
        if other.category != item.category or other.id == item.id:
            continue
        if item.embedding and other.embedding:
            similarity = _cosine(item.embedding, other.embedding)
            is_similar = similarity >= 0.9
        else:
            same_color = bool(item.colors and other.colors) and nearest_name(
                hex_to_rgb(item.colors[0])
            ) == nearest_name(hex_to_rgb(other.colors[0]))
            similarity = 0.9 if same_color else 0.0
            is_similar = same_color and item.pattern == other.pattern
        if is_similar:
            similar.append(Pairing(item_id=other.id, name=other.name, score=round(similarity, 3)))
    return similar


def _works_together(pieces: list[ClosetItem]) -> bool:
    return all(pair_score(a, b) >= COMPATIBLE for a, b in combinations(pieces, 2))


def count_outfits(item: ClosetItem, closet: list[ClosetItem]) -> int:
    """How many complete outfits the new item creates with what you already own."""
    by_category: dict[str, list[ClosetItem]] = {}
    for other in closet:
        if other.id != item.id:
            by_category.setdefault(other.category, []).append(other)
    tops = by_category.get("top", [])[:MAX_PER_CATEGORY]
    bottoms = by_category.get("bottom", [])[:MAX_PER_CATEGORY]
    shoes = by_category.get("shoes", [])[:MAX_PER_CATEGORY] or [None]
    dresses = by_category.get("dress", [])[:MAX_PER_CATEGORY]

    def outfits(candidates):
        return sum(1 for pieces in candidates if _works_together([p for p in pieces if p]))

    match item.category:
        case "top":
            return outfits([item, b, s] for b in bottoms for s in shoes)
        case "bottom":
            return outfits([t, item, s] for t in tops for s in shoes)
        case "dress":
            return outfits([item, s] for s in shoes)
        case "shoes":
            return outfits([t, b, item] for t in tops for b in bottoms) + outfits(
                [d, item] for d in dresses
            )
        case "outerwear":
            return outfits([t, b, item] for t in tops for b in bottoms) + outfits(
                [d, item] for d in dresses
            )
    return 0


def match(request: MatchRequest) -> MatchResponse:
    item, closet = request.item, request.closet

    partners = [c for c in closet if c.category in PARTNERS[item.category] and c.id != item.id]
    scored = sorted(((pair_score(item, c), c) for c in partners), key=lambda p: p[0], reverse=True)
    best = [Pairing(item_id=c.id, name=c.name, score=s) for s, c in scored[:5]]
    good_partners = sum(1 for s, _ in scored if s >= COMPATIBLE)

    outfit_count = count_outfits(item, closet)
    similar = find_similar(item, closet)

    average = sum(p.score for p in best) / len(best) if best else 0.0
    score = round(100 * (0.6 * average + 0.4 * min(outfit_count, 10) / 10))
    if similar:
        score = max(0, score - 15)
    verdict = "great" if score >= 70 else "good" if score >= 45 else "hard"

    reasons = []
    if not partners:
        reasons.append("Nothing in your closet to pair it with yet.")
    else:
        reasons.append(f"Goes well with {good_partners} of {len(partners)} matching pieces.")
        reasons.append(f"Creates {outfit_count} complete outfits with what you own.")
    if similar:
        names = ", ".join(p.name for p in similar[:3])
        reasons.append(f"You already own something very similar: {names}.")
    if item.colors and is_neutral(item.colors[0]):
        reasons.append("A neutral color, easy to combine.")
    if item.pattern != "solid":
        reasons.append("A patterned piece: it pairs best with solid colors.")

    return MatchResponse(
        score=score,
        verdict=verdict,
        outfit_count=outfit_count,
        best_pairings=best,
        similar_items=similar,
        reasons=reasons,
    )
