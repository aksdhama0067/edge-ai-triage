"""
inference.py
------------
FastAPI application. Wires together the modular services:

    frontend -> image_quality -> model (CNN) -> explainability / similarity
             -> triage (rules + questionnaire fusion) -> result

Run (from the project root, recommended):
    python -m uvicorn backend.inference:app --reload --host 127.0.0.1 --port 8000

Run (from inside backend/, also supported):
    uvicorn inference:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Support both `python -m uvicorn backend.inference:app` (package mode) and
# `cd backend && uvicorn inference:app` (script mode).
try:
    from . import config
    from .explainability import generate_gradcam
    from .image_quality import assess_image_quality
    from .model import get_embedding, load_model, model_status, predict_lesion
    from .preprocess import load_image_from_bytes, parse_lesion_questionnaire, parse_symptom_slots, preprocess_image
    from .schemas import (
        AssistantRequest,
        AssistantResponse,
        ExplainResponse,
        HealthResponse,
        ModelAnalysis,
        QualityGateResult,
        SimilarityResponse,
        TriageResponse,
    )
    from .services.assistant import answer_question
    from .similarity import SimilarityIndex
    from .triage import evaluate_danger_signs, evaluate_lesion_questionnaire, fuse_triage_decision
except ImportError:
    import config
    from explainability import generate_gradcam
    from image_quality import assess_image_quality
    from model import get_embedding, load_model, model_status, predict_lesion
    from preprocess import load_image_from_bytes, parse_lesion_questionnaire, parse_symptom_slots, preprocess_image
    from schemas import (
        AssistantRequest,
        AssistantResponse,
        ExplainResponse,
        HealthResponse,
        ModelAnalysis,
        QualityGateResult,
        SimilarityResponse,
        TriageResponse,
    )
    from services.assistant import answer_question
    from similarity import SimilarityIndex
    from triage import evaluate_danger_signs, evaluate_lesion_questionnaire, fuse_triage_decision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edge_ai_triage.inference")

app = FastAPI(
    title="Edge AI Triage — Research Prototype",
    description=(
        "An educational/research prototype for offline-first skin-lesion triage assistance. "
        "This is NOT a medical diagnostic device and has not been clinically validated."
    ),
    version=config.APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the CNN once at startup and load/attempt the similarity index once.
_model = load_model()
_similarity_index = SimilarityIndex()

STATIC_DIR = config.FRONTEND_DIR / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


DISCLAIMER = (
    "This is an educational/research prototype, not a medical diagnosis. "
    "Consider professional evaluation for any health concern."
)


def _read_and_validate_image(image: UploadFile, image_bytes: bytes) -> None:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(image_bytes) > config.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {config.MAX_IMAGE_MB:.0f} MB upload limit.",
        )
    content_type = (image.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {content_type}")
    try:
        load_image_from_bytes(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file could not be read as an image.")


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    """Serve the local frontend so the whole demo runs from one origin."""
    index_path = config.FRONTEND_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend build not found.")
    return FileResponse(str(index_path))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    status = model_status()
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_mode=status["mode"],
        clinically_validated=status["clinically_validated"],
        similarity_available=_similarity_index.available,
        assistant_provider="openai" if config.assistant_external_configured() else "local",
        version=config.APP_VERSION,
    )


# ---------------------------------------------------------------------------
# Image quality (standalone)
# ---------------------------------------------------------------------------

@app.post("/api/image-quality", response_model=QualityGateResult)
async def image_quality_endpoint(image: UploadFile = File(...)) -> QualityGateResult:
    image_bytes = await image.read()
    _read_and_validate_image(image, image_bytes)
    result = assess_image_quality(image_bytes)
    return QualityGateResult(**result)


# ---------------------------------------------------------------------------
# Triage (full pipeline)
# ---------------------------------------------------------------------------

@app.post("/api/triage", response_model=TriageResponse)
async def triage(
    image: UploadFile = File(..., description="Lesion photo (JPEG/PNG/WebP)"),
    symptoms: str = Form(default="{}", description="JSON string of respiratory symptoms (WHO IMCI pathway)"),
    questionnaire: str = Form(default="{}", description="JSON string of the lesion follow-up questionnaire"),
) -> TriageResponse:
    """
    Full pipeline: image quality gate -> CNN prediction -> WHO danger-sign
    rules -> lesion questionnaire -> late-fusion triage decision.
    """
    image_bytes = await image.read()
    _read_and_validate_image(image, image_bytes)

    quality_result = assess_image_quality(image_bytes)
    quality_gate = QualityGateResult(**quality_result)

    status = model_status()
    lesion_probabilities: dict[str, float] = {}
    model_analysis = ModelAnalysis(
        available=False,
        model_mode=status["mode"],
        disclaimer=(
            "A trained lesion model is not installed. No learned lesion prediction was made."
            if status["mode"] == "unavailable"
            else "This model has not undergone clinical validation."
        ),
    )

    try:
        raw_symptoms = json.loads(symptoms) if symptoms else {}
        raw_questionnaire = json.loads(questionnaire) if questionnaire else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`symptoms` and `questionnaire` must be valid JSON strings.")

    parsed_symptoms = parse_symptom_slots(raw_symptoms)
    parsed_questionnaire = parse_lesion_questionnaire(raw_questionnaire)
    danger_sign_result = evaluate_danger_signs(parsed_symptoms)
    questionnaire_result = evaluate_lesion_questionnaire(parsed_questionnaire)

    if quality_result["quality"] != "poor" and _model is not None:
        image_tensor = preprocess_image(image_bytes)
        lesion_probabilities = predict_lesion(_model, image_tensor)
        top_class = max(lesion_probabilities, key=lesion_probabilities.get)
        model_analysis = ModelAnalysis(
            available=True,
            model_mode=status["mode"],
            class_probabilities={k: round(v, 4) for k, v in lesion_probabilities.items()},
            top_class=top_class,
            top_confidence=round(lesion_probabilities[top_class], 4),
            disclaimer="This model has not undergone clinical validation.",
        )
    elif quality_result["quality"] == "poor":
        # Never run a learned model on an unusable image.
        lesion_probabilities = {}
    else:
        # No checkpoint means there is deliberately no learned lesion signal.
        lesion_probabilities = {}

    if lesion_probabilities:
        decision = fuse_triage_decision(lesion_probabilities, danger_sign_result, questionnaire_result, quality_result)
    else:
        factors = [f"Image quality: {quality_result['quality'].capitalize()} (score {quality_result['score']}/100)"]
        if danger_sign_result["danger_sign_present"]:
            factors.extend(danger_sign_result["triggered_rules"])
            decision = {
                "triage_tier": "Red",
                "confidence": 0.0,
                "reason": "Reported symptom(s) matched the configured danger-sign rules. A trained skin-lesion model was not used.",
                "factors": factors,
            }
        elif quality_result["quality"] == "poor":
            decision = {
                "triage_tier": "Yellow",
                "confidence": 0.0,
                "reason": f"Image quality was too low for visual assessment. {quality_result['recommendation']}",
                "factors": factors + ["Retake the image before attempting visual AI analysis."],
            }
        else:
            factors.append("Trained skin-lesion model: unavailable")
            if questionnaire_result["concern_flags"]:
                factors.extend(questionnaire_result["concern_flags"])
            decision = {
                "triage_tier": "Yellow",
                "confidence": 0.0,
                "reason": "A trained skin-lesion model is not installed, so no visual diagnosis-like prediction was made. The result is limited to image quality and reported history.",
                "factors": factors,
            }

    return TriageResponse(
        triage_tier=decision["triage_tier"],
        confidence=decision["confidence"],
        reason=decision["reason"],
        factors=decision["factors"],
        quality_gate=quality_gate,
        model_analysis=model_analysis,
        disclaimer=DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Explainability (Grad-CAM)
# ---------------------------------------------------------------------------

@app.post("/api/explain", response_model=ExplainResponse)
async def explain(image: UploadFile = File(...)) -> ExplainResponse:
    if config.DISABLE_GRADCAM:
        return ExplainResponse(available=False, message="Explainability is disabled on this deployment.")
    if _model is None:
        return ExplainResponse(available=False, message="Grad-CAM is unavailable until a trained lesion model is installed.")

    image_bytes = await image.read()
    _read_and_validate_image(image, image_bytes)

    quality_result = assess_image_quality(image_bytes)
    if quality_result["quality"] == "poor":
        return ExplainResponse(
            available=False,
            message="Image quality is too low to generate a reliable attention map. Please retake the photo.",
        )

    original_image = load_image_from_bytes(image_bytes)
    image_tensor = preprocess_image(image_bytes)

    try:
        result = generate_gradcam(_model, image_tensor, original_image)
    except Exception as exc:  # pragma: no cover
        logger.warning("Grad-CAM generation failed: %s", exc)
        return ExplainResponse(available=False, message="Could not generate an attention map for this image.")

    return ExplainResponse(
        available=True,
        predicted_class=result["predicted_class"],
        predicted_confidence=result["predicted_confidence"],
        heatmap_overlay_base64=result["heatmap_overlay_base64"],
    )


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

@app.post("/api/similarity", response_model=SimilarityResponse)
async def similarity_endpoint(image: UploadFile = File(...)) -> SimilarityResponse:
    if _model is None:
        return SimilarityResponse(
            available=False,
            message="Similarity search is unavailable until a trained lesion model is installed.",
        )
    if not _similarity_index.available:
        return SimilarityResponse(
            available=False,
            message="Similarity search is not configured yet. See models/README.md to build an index.",
        )

    image_bytes = await image.read()
    _read_and_validate_image(image, image_bytes)

    image_tensor = preprocess_image(image_bytes)
    embedding = get_embedding(_model, image_tensor)
    result = _similarity_index.search(embedding)

    return SimilarityResponse(available=result["available"], message=result.get("message", ""), matches=result["matches"])


# ---------------------------------------------------------------------------
# AI assistant
# ---------------------------------------------------------------------------

@app.post("/api/assistant", response_model=AssistantResponse)
async def assistant_endpoint(request: AssistantRequest) -> AssistantResponse:
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="`question` must not be empty.")
    result = answer_question(request.question.strip(), request.context)
    return AssistantResponse(answer=result["answer"], provider=result["provider"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference:app", host="127.0.0.1", port=8000, reload=True)
