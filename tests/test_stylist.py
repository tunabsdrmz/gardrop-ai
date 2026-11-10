from app.main import app
from app.schemas import ClosetItem, StylistRequest, Weather
from app.services.stylist import get_anthropic_client, run_stylist
from tests.conftest import FakeAnthropic, reply, text_block, tool_block

REQUEST = StylistRequest(
    message="What should I wear to a wedding on Saturday?",
    closet=[
        ClosetItem(id="dress-1", name="Burgundy dress", category="dress", colors=["6B1F33"],
                   formality="formal"),
        ClosetItem(id="heels-1", name="Black heels", category="shoes", colors=["141416"],
                   formality="formal"),
    ],
    weather=Weather(city="Ankara", temperature=12, low=6, condition="cloudy"),
)


def test_stylist_uses_tools_then_answers():
    fake = FakeAnthropic([
        reply(tool_block("t1", "get_closet", {}), tool_block("t2", "get_weather", {}),
              stop_reason="tool_use"),
        reply(tool_block("t3", "suggest_outfit", {"title": "Wedding guest",
                                                  "item_ids": ["dress-1", "heels-1"],
                                                  "reason": "Elegant and warm enough."}),
              stop_reason="tool_use"),
        reply(text_block("Your burgundy dress is perfect for it.")),
    ])

    result = run_stylist(REQUEST, fake, "test-model")

    assert result.reply == "Your burgundy dress is perfect for it."
    assert result.outfits[0].item_ids == ["dress-1", "heels-1"]
    # The closet was sent back to Claude as a tool result.
    first_results = fake.calls[1]["messages"][-1]["content"]
    assert "Burgundy dress" in first_results[0]["content"]
    assert "Ankara" in first_results[1]["content"]


def test_unknown_item_ids_are_reported_back_as_errors():
    fake = FakeAnthropic([
        reply(tool_block("t1", "suggest_outfit", {"title": "x", "item_ids": ["made-up"],
                                                  "reason": "x"}), stop_reason="tool_use"),
        reply(text_block("Sorry, let me check your closet.")),
    ])

    result = run_stylist(REQUEST, fake, "test-model")

    assert result.outfits == []
    tool_result = fake.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True


def test_stylist_returns_503_without_api_key(client):
    body = {"message": "Hi", "closet": []}

    assert client.post("/stylist", json=body).status_code == 503


def test_stylist_endpoint(client):
    app.dependency_overrides[get_anthropic_client] = lambda: FakeAnthropic(
        [reply(text_block("Try your burgundy dress."))]
    )
    body = {"message": "Wedding?", "closet": [c.model_dump() for c in REQUEST.closet]}
    response = client.post("/stylist", json=body)

    assert response.status_code == 200
    assert response.json()["reply"] == "Try your burgundy dress."
