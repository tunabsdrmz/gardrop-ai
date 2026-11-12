import json

from app.config import Settings
from app.main import app
from app.schemas import ClosetItem, StylistRequest, Weather
from app.services.stylist import (
    AnthropicStylist,
    OpenAICompatibleStylist,
    get_stylist,
)
from tests.conftest import (
    FakeAnthropic,
    FakeOpenAI,
    chat_reply,
    reply,
    text_block,
    tool_block,
    tool_call,
)

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

OUTFIT = {"title": "Wedding guest", "item_ids": ["dress-1", "heels-1"],
          "reason": "Elegant and warm enough."}


# MARK: - OpenAI-compatible (Ollama, Groq, Gemini)


def test_openai_stylist_uses_tools_then_answers():
    fake = FakeOpenAI([
        chat_reply(tool_calls=[tool_call("c1", "get_closet", "{}"),
                               tool_call("c2", "get_weather", "{}")]),
        chat_reply(tool_calls=[tool_call("c3", "suggest_outfit", json.dumps(OUTFIT))]),
        chat_reply(content="Your burgundy dress is perfect for it."),
    ])

    result = OpenAICompatibleStylist(fake, "test-model").run(REQUEST)

    assert result.reply == "Your burgundy dress is perfect for it."
    assert result.outfits[0].item_ids == ["dress-1", "heels-1"]
    sent = fake.calls[1]["messages"]
    assert sent[0]["role"] == "system"
    tool_messages = [m for m in sent if m["role"] == "tool"]
    assert "Burgundy dress" in tool_messages[0]["content"]
    assert "Ankara" in tool_messages[1]["content"]


def test_openai_stylist_survives_broken_tool_arguments():
    fake = FakeOpenAI([
        chat_reply(tool_calls=[tool_call("c1", "suggest_outfit", "{not json")]),
        chat_reply(content="Let me try again later."),
    ])

    result = OpenAICompatibleStylist(fake, "test-model").run(REQUEST)

    assert result.outfits == []
    tool_message = fake.calls[1]["messages"][-1]
    assert "not valid JSON" in tool_message["content"]


# MARK: - Anthropic


def test_anthropic_stylist_uses_tools_then_answers():
    fake = FakeAnthropic([
        reply(tool_block("t1", "get_closet", {}), tool_block("t2", "get_weather", {}),
              stop_reason="tool_use"),
        reply(tool_block("t3", "suggest_outfit", OUTFIT), stop_reason="tool_use"),
        reply(text_block("Your burgundy dress is perfect for it.")),
    ])

    result = AnthropicStylist(fake, "test-model").run(REQUEST)

    assert result.reply == "Your burgundy dress is perfect for it."
    assert result.outfits[0].item_ids == ["dress-1", "heels-1"]
    first_results = fake.calls[1]["messages"][-1]["content"]
    assert "Burgundy dress" in first_results[0]["content"]


def test_unknown_item_ids_are_reported_back_as_errors():
    fake = FakeAnthropic([
        reply(tool_block("t1", "suggest_outfit", {"title": "x", "item_ids": ["made-up"],
                                                  "reason": "x"}), stop_reason="tool_use"),
        reply(text_block("Sorry, let me check your closet.")),
    ])

    result = AnthropicStylist(fake, "test-model").run(REQUEST)

    assert result.outfits == []
    assert fake.calls[1]["messages"][-1]["content"][0]["is_error"] is True


# MARK: - Configuration and endpoint


def test_get_stylist_picks_the_backend():
    assert isinstance(get_stylist(Settings(stylist_provider="ollama")), OpenAICompatibleStylist)
    assert get_stylist(Settings(stylist_provider="gemini", gemini_api_key=None)) is None
    claude = get_stylist(Settings(stylist_provider="anthropic", anthropic_api_key="k"))
    assert isinstance(claude, AnthropicStylist)
    assert claude.model == "claude-haiku-5-5"


def test_stylist_returns_503_when_not_configured(client):
    assert client.post("/stylist", json={"message": "Hi", "closet": []}).status_code == 503


def test_stylist_endpoint(client):
    fake = FakeOpenAI([chat_reply(content="Try your burgundy dress.")])
    app.dependency_overrides[get_stylist] = lambda: OpenAICompatibleStylist(fake, "m")
    body = {"message": "Wedding?", "closet": [c.model_dump() for c in REQUEST.closet]}
    response = client.post("/stylist", json=body)

    assert response.status_code == 200
    assert response.json()["reply"] == "Try your burgundy dress."


def test_stylist_returns_502_when_the_model_is_down(client):
    class Down:
        def run(self, request):
            raise ConnectionError("Ollama is not running")

    app.dependency_overrides[get_stylist] = lambda: Down()
    response = client.post("/stylist", json={"message": "Hi", "closet": []})

    assert response.status_code == 502
