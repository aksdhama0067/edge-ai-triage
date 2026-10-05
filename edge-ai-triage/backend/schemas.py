"""
schemas.py
----------
Pydantic request/response models shared across endpoints. Keeping these in
one place makes the API's contract easy to audit — every field the frontend
can rely on is declared here, and nothing about model confidence or
"clinical validity" is implied unless it is genuinely true at runtime.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QualityGateResult(BaseModel):
    quality: str = Field(..., description='"good" | "warning" | "poor"')
    score: int = Field(..., ge=0, le=100)
    issues: List[str] = Field(default_factory=list)
    recommendation: str
    metrics: Dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_mode: str  # "trained" | "unavailable"
    clinically_validated: bool
    similarity_available: bool
    assistant_provider: str  # "local" | "openai"
    version: str


class LesionQuestionnaire(BaseModel):
    """Structured dermatology follow-up questions (all optional, default = no)."""

    recent_change: bool = False
    size_change: bool = False
    appearance_change: bool = False
    painful: bool = False
    itchy: bool = False
    bleeding: bool = False
    duration_days: int = 0


class RespiratorySymptoms(BaseModel):
    """Optional respiratory danger-sign inputs (WHO IMCI/IMAI pathway)."""

    cough: bool = False
    cough_days: int = 0
    fever: bool = False
    breathing_difficulty: str = "none"  # "none" | "mild" | "severe"
    chest_indrawing: bool = False


class ModelAnalysis(BaseModel):
    available: bool
    model_mode: str
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    top_class: Optional[str] = None
    top_confidence: Optional[float] = None
    disclaimer: str


class TriageResponse(BaseModel):
    triage_tier: str  # "Red" | "Yellow" | "Green"
    confidence: float
    reason: str
    factors: List[str] = Field(default_factory=list)
    quality_gate: QualityGateResult
    model_analysis: ModelAnalysis
    disclaimer: str


class ExplainResponse(BaseModel):
    available: bool
    message: str = ""
    predicted_class: Optional[str] = None
    predicted_confidence: Optional[float] = None
    heatmap_overlay_base64: Optional[str] = None
    disclaimer: str = (
        "The highlighted region shows areas that contributed to the model's "
        "prediction. This visualization does not prove that the highlighted "
        "region is medically abnormal."
    )


class SimilarMatch(BaseModel):
    label: Optional[str] = None
    similarity_score: float
    thumbnail_base64: Optional[str] = None


class SimilarityResponse(BaseModel):
    available: bool
    message: str = ""
    matches: List[SimilarMatch] = Field(default_factory=list)
    disclaimer: str = "Visual similarity does not indicate that two lesions have the same diagnosis."


class AssistantRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None
    language: str = "en"


class AssistantResponse(BaseModel):
    answer: str
    provider: str  # "local" | "openai"
    disclaimer: str = "This assistant explains the tool. It does not diagnose medical conditions."
