
from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import evaluate_danger_signs, fuse_triage_decision, load_model, predict_lesion
from preprocess import check_image_quality, parse_symptom_slots, preprocess_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edge_ai_triage.inference")

app = FastAPI(
    title="Edge AI Diagnostic Assistance API",
    description="Offline-first CV + rule-based triage for low-bandwidth rural clinics.",
    version="0.1.0",
)

# Allow the local static frontend (opened via file:// or a simple dev server)
# to call this API without CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup so requests don't pay the load cost.
_model = load_model()

#response
class QualityGateResult(BaseModel):
    passed: bool
    message: str


class TriageResponse(BaseModel):
    triage_tier: str
    confidence: float
    reason: str
    quality_gate: QualityGateResult


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

#endpoint

@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Quick system status check, useful for CHWs verifying the offline device is up."""
    return HealthResponse(status="ok", model_loaded=_model is not None, version=app.version)


@app.post("/api/triage", response_model=TriageResponse)
async def triage(
    image: UploadFile = File(..., description="Lesion photo (JPEG/PNG)"),
    symptoms: str = Form(
        default="{}",
        description=(
            "JSON string of respiratory symptoms, e.g. "
            '{"cough": true, "cough_days": 5, "fever": true, '
            '"breathing_difficulty": "severe", "chest_indrawing": false}'
        ),
    ),
) -> TriageResponse:
    """
    Accepts an image upload and a JSON-encoded symptom form, runs the full
    offline triage pipeline, and returns a triage tier with an explanation.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    # 1. Quality gate
    quality_passed, quality_message = check_image_quality(image_bytes)
    quality_gate = QualityGateResult(passed=quality_passed, message=quality_message)

    if not quality_passed:
        # We still return a 200 with a clear "cannot assess" response rather
        # than a hard error, since a CHW needs actionable guidance (retake
        # the photo), not a stack trace.
        return TriageResponse(
            triage_tier="Yellow",
            confidence=0.0,
            reason=f"Image quality gate failed: {quality_message} "
                   "Defaulting to manual clinical review.",
            quality_gate=quality_gate,
        )

    # 2. Parse symptoms
    try:
        raw_symptoms = json.loads(symptoms) if symptoms else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`symptoms` must be a valid JSON string.")
    parsed_symptoms = parse_symptom_slots(raw_symptoms)

    try:
        image_tensor = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}")
    lesion_probabilities = predict_lesion(_model, image_tensor)

    danger_sign_result = evaluate_danger_signs(parsed_symptoms)

    decision = fuse_triage_decision(lesion_probabilities, danger_sign_result)

    return TriageResponse(
        triage_tier=decision["triage_tier"],
        confidence=decision["confidence"],
        reason=decision["reason"],
        quality_gate=quality_gate,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)
