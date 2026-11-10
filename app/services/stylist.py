"""AI stylist: Claude with tool use.

Claude can't see the closet directly. It asks for it by calling our tools
(get_closet, get_weather), and shows outfits by calling suggest_outfit.
We run the tools, send the results back, and repeat until Claude answers.
"""

import json
from functools import lru_cache
from typing import Any

from app.config import get_settings
from app.schemas import StylistRequest, StylistResponse, SuggestedOutfit

SYSTEM_PROMPT = """You are the personal stylist inside Gardrop, a wardrobe app.
- Only suggest pieces the user owns. Look them up with get_closet and use their ids.
- Check get_weather before suggesting outfits for today or tomorrow.
- Show every outfit you recommend with suggest_outfit (at most 3).
- Keep your reply short: 2-4 warm, specific sentences. No lists of item ids in the text.
- Reply in the same language as the user's latest message."""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_closet",
        "description": "Returns the user's clothing items. Optionally only one category.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["top", "bottom", "outerwear", "shoes", "dress"],
                }
            },
        },
    },
    {
        "name": "get_weather",
        "description": "Returns today's weather where the user is.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "suggest_outfit",
        "description": "Shows an outfit to the user as a card in the app. Call once per outfit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short name, e.g. 'Wedding guest'"},
                "item_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "reason": {"type": "string", "description": "One sentence on why it works"},
            },
            "required": ["title", "item_ids", "reason"],
        },
    },
]

MAX_TURNS = 6
MAX_OUTFITS = 3


def _run_tool(
    name: str, arguments: dict[str, Any], request: StylistRequest, outfits: list[SuggestedOutfit]
) -> tuple[Any, bool]:
    """Executes one tool call. Returns (result, is_error)."""
    if name == "get_closet":
        category = arguments.get("category")
        items = [i for i in request.closet if category is None or i.category == category]
        return [i.model_dump(exclude={"embedding"}) for i in items], False

    if name == "get_weather":
        if request.weather is None:
            return {"error": "Weather is unknown. Ask the user or suggest versatile layers."}, True
        return request.weather.model_dump(), False

    if name == "suggest_outfit":
        known = {i.id for i in request.closet}
        ids = [str(i) for i in arguments.get("item_ids", [])]
        unknown = [i for i in ids if i not in known]
        if unknown:
            return {"error": f"These ids are not in the closet: {unknown}. Use get_closet."}, True
        if len(outfits) >= MAX_OUTFITS:
            return {"error": "Already showed 3 outfits. Write your reply now."}, True
        outfits.append(
            SuggestedOutfit(
                title=str(arguments.get("title", "")),
                item_ids=ids,
                reason=str(arguments.get("reason", "")),
            )
        )
        return {"shown": True}, False

    return {"error": f"Unknown tool: {name}"}, True


def run_stylist(request: StylistRequest, client: Any, model: str) -> StylistResponse:
    messages: list[dict[str, Any]] = [
        {"role": m.role, "content": m.content} for m in request.history[-10:]
    ]
    messages.append({"role": "user", "content": request.message})
    outfits: list[SuggestedOutfit] = []

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            reply = "".join(b.text for b in response.content if b.type == "text").strip()
            return StylistResponse(reply=reply, outfits=outfits)

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            output, is_error = _run_tool(block.name, block.input, request, outfits)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output, ensure_ascii=False),
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})

    # Claude kept calling tools; return what we have instead of looping forever.
    return StylistResponse(reply="Here are my picks for you.", outfits=outfits)


@lru_cache
def get_anthropic_client() -> Any | None:
    """None when no API key is configured; the endpoint then answers 503."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        return None
    import anthropic

    return anthropic.Anthropic(api_key=settings.anthropic_api_key)
