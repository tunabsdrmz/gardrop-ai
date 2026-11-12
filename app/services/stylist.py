"""AI stylist: an LLM with tool use.

The model can't see the closet directly. It asks for it by calling our tools
(get_closet, get_weather), and shows outfits by calling suggest_outfit.
We run the tools, send the results back, and repeat until the model answers.

Two backends share the same tools and rules:
- OpenAICompatibleStylist: Ollama (local, free), Groq and Gemini
- AnthropicStylist: Claude
"""

import json
from functools import lru_cache
from typing import Annotated, Any, Protocol

from fastapi import Depends

from app.config import Settings, get_settings
from app.schemas import StylistRequest, StylistResponse, SuggestedOutfit

SYSTEM_PROMPT = """You are the personal stylist inside Gardrop, a wardrobe app.
- Only suggest pieces the user owns. Look them up with get_closet and use their ids.
- Check get_weather before suggesting outfits for today or tomorrow.
- Show every outfit you recommend with suggest_outfit (at most 3).
- Keep your reply short: 2-4 warm, specific sentences. No lists of item ids in the text.
- Reply in the same language as the user's latest message."""

# Tool definitions, written once. Each backend converts them to its own format.
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
FALLBACK_REPLY = "Here are my picks for you."


class Stylist(Protocol):
    def run(self, request: StylistRequest) -> StylistResponse: ...


def run_tool(
    name: str, arguments: dict[str, Any], request: StylistRequest, outfits: list[SuggestedOutfit]
) -> tuple[Any, bool]:
    """Executes one tool call. Returns (result, is_error). Shared by every backend."""
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


def _history(request: StylistRequest) -> list[dict[str, Any]]:
    """The last 10 messages plus the new one; older ones only add cost."""
    messages = [{"role": m.role, "content": m.content} for m in request.history[-10:]]
    messages.append({"role": "user", "content": request.message})
    return messages


class OpenAICompatibleStylist:
    """Ollama, Groq and Gemini all speak the OpenAI chat format."""

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    @staticmethod
    def _tools() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in TOOLS
        ]

    def run(self, request: StylistRequest) -> StylistResponse:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += _history(request)
        outfits: list[SuggestedOutfit] = []

        for _ in range(MAX_TURNS):
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=self._tools()
            )
            message = response.choices[0].message
            calls = message.tool_calls or []

            if not calls:
                return StylistResponse(reply=(message.content or "").strip(), outfits=outfits)

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in calls
                    ],
                }
            )
            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    output = {"error": "Tool arguments were not valid JSON. Try again."}
                else:
                    output, _ = run_tool(call.function.name, arguments, request, outfits)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(output, ensure_ascii=False),
                    }
                )

        return StylistResponse(reply=FALLBACK_REPLY, outfits=outfits)


class AnthropicStylist:
    """Claude's native tool-use format."""

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def run(self, request: StylistRequest) -> StylistResponse:
        messages = _history(request)
        outfits: list[SuggestedOutfit] = []

        for _ in range(MAX_TURNS):
            response = self.client.messages.create(
                model=self.model,
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
                output, is_error = run_tool(block.name, block.input, request, outfits)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output, ensure_ascii=False),
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": results})

        return StylistResponse(reply=FALLBACK_REPLY, outfits=outfits)


# MARK: - Clients

PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


@lru_cache
def _openai_client(base_url: str, api_key: str) -> Any:
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=60)


@lru_cache
def _anthropic_client(api_key: str) -> Any:
    import anthropic

    return anthropic.Anthropic(api_key=api_key)


def get_stylist(settings: Annotated[Settings, Depends(get_settings)]) -> Stylist | None:
    """Builds the stylist for the configured provider, or None when a key is missing."""
    if not settings.stylist_configured:
        return None
    model = settings.resolved_stylist_model

    match settings.stylist_provider:
        case "ollama":
            # Ollama ignores the key, but the client needs a non-empty one.
            client = _openai_client(settings.ollama_base_url, "ollama")
            return OpenAICompatibleStylist(client, model)
        case "groq":
            client = _openai_client(PROVIDER_BASE_URLS["groq"], settings.groq_api_key or "")
            return OpenAICompatibleStylist(client, model)
        case "gemini":
            client = _openai_client(PROVIDER_BASE_URLS["gemini"], settings.gemini_api_key or "")
            return OpenAICompatibleStylist(client, model)
        case "anthropic":
            return AnthropicStylist(_anthropic_client(settings.anthropic_api_key or ""), model)
    return None
