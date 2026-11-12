"""Request and response shapes. FastAPI validates every request against these."""

from typing import Literal

from pydantic import BaseModel, Field

# Same values as the iOS app's ClothingCategory.
Category = Literal["top", "bottom", "outerwear", "shoes", "dress"]
Pattern = Literal["solid", "striped", "plaid", "floral", "dotted", "graphic"]
Season = Literal["summer", "winter", "all-season"]
Formality = Literal["sporty", "casual", "smart-casual", "formal"]


class HealthResponse(BaseModel):
    status: str
    version: str
    stylist_provider: str
    stylist_enabled: bool
    clip_loaded: bool


# MARK: - /tag


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class ColorSwatch(BaseModel):
    hex: str = Field(examples=["C4A27A"])
    name: str = Field(examples=["camel"])
    ratio: float = Field(ge=0, le=1, description="Share of the garment covered by this color")


class TagResponse(BaseModel):
    category: Prediction
    pattern: Prediction
    season: Prediction
    formality: Prediction
    colors: list[ColorSwatch]
    embedding: list[float] = Field(description="CLIP image embedding, used by /match")


# MARK: - /match


class ClosetItem(BaseModel):
    id: str
    name: str
    category: Category
    colors: list[str] = Field(default_factory=list, description="Hex colors, main color first")
    pattern: Pattern = "solid"
    formality: Formality = "casual"
    embedding: list[float] | None = None


class MatchRequest(BaseModel):
    item: ClosetItem
    closet: list[ClosetItem] = Field(max_length=1000)


class Pairing(BaseModel):
    item_id: str
    name: str
    score: float


class MatchResponse(BaseModel):
    score: int = Field(ge=0, le=100)
    verdict: Literal["great", "good", "hard"]
    outfit_count: int
    best_pairings: list[Pairing]
    similar_items: list[Pairing]
    reasons: list[str]


# MARK: - /stylist


class Weather(BaseModel):
    city: str | None = None
    temperature: float
    low: float | None = None
    condition: str = Field(examples=["sunny", "rainy"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class StylistRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)
    closet: list[ClosetItem] = Field(max_length=1000)
    weather: Weather | None = None


class SuggestedOutfit(BaseModel):
    title: str
    item_ids: list[str]
    reason: str


class StylistResponse(BaseModel):
    reply: str
    outfits: list[SuggestedOutfit]
