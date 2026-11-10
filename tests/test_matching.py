from app.schemas import ClosetItem, MatchRequest
from app.services.matching import count_outfits, match, pair_score


def item(id, category, color, pattern="solid", formality="casual", name=None):
    return ClosetItem(id=id, name=name or id, category=category, colors=[color],
                      pattern=pattern, formality=formality)


CLOSET = [
    item("jeans", "bottom", "1F2A44"),
    item("chinos", "bottom", "E1D2B9"),
    item("boots", "shoes", "64462D"),
    item("sneakers", "shoes", "F5F5F2"),
    item("white-tee", "top", "F5F5F2"),
]


def test_neutral_top_is_a_great_buy():
    result = match(MatchRequest(item=item("camel-knit", "top", "C4A27A"), closet=CLOSET))

    assert result.verdict == "great"
    assert result.outfit_count == 4  # 2 bottoms x 2 shoes
    assert {p.item_id for p in result.best_pairings} >= {"jeans", "chinos"}


def test_duplicate_is_flagged_and_penalized():
    duplicate = item("another-white-tee", "top", "F2F2F0")
    result = match(MatchRequest(item=duplicate, closet=CLOSET))

    assert [p.item_id for p in result.similar_items] == ["white-tee"]
    assert any("similar" in r for r in result.reasons)


def test_empty_closet_is_hard():
    result = match(MatchRequest(item=item("tee", "top", "BE1E2D"), closet=[]))

    assert result.verdict == "hard"
    assert result.outfit_count == 0
    assert result.reasons[0] == "Nothing in your closet to pair it with yet."


def test_two_patterns_score_lower_than_pattern_and_solid():
    floral = item("floral", "top", "BE1E2D", pattern="floral")
    plaid = item("plaid", "bottom", "1F2A44", pattern="plaid")
    plain = item("plain", "bottom", "1F2A44")

    assert pair_score(floral, plain) > pair_score(floral, plaid)


def test_formal_shoes_dont_pair_with_sportswear():
    sporty_top = item("hoodie", "top", "808080", formality="sporty")
    formal_shoes = item("oxfords", "shoes", "141416", formality="formal")
    bottom = item("trousers", "bottom", "141416", formality="formal")

    assert count_outfits(sporty_top, [formal_shoes, bottom]) == 0


def test_match_endpoint(client):
    body = {"item": {"id": "x", "name": "Knit", "category": "top", "colors": ["C4A27A"]},
            "closet": [c.model_dump() for c in CLOSET]}
    response = client.post("/match", json=body)

    assert response.status_code == 200
    assert response.json()["verdict"] == "great"


def test_match_validates_categories(client):
    body = {"item": {"id": "x", "name": "Hat", "category": "hat"}, "closet": []}

    assert client.post("/match", json=body).status_code == 422
