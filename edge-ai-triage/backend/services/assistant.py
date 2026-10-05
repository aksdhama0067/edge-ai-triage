"""
services/assistant.py
----------------------
An "explain the tool" assistant, not a diagnostic one.

Default mode ("local"): a deterministic, keyword-matched knowledge base of
predefined, verified explanations. No network calls, no invented medical
facts, fully offline.

Optional mode ("openai"): if AI_PROVIDER=openai and OPENAI_API_KEY is set,
questions are answered by an OpenAI-compatible chat completions endpoint
under a constrained system prompt that forbids diagnosis and invented facts.
If the request fails for any reason (no network, bad key, timeout), this
module falls back to the local knowledge base rather than erroring out.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    from .. import config
except ImportError:
    import config

logger = logging.getLogger("edge_ai_triage.assistant")

try:
    import requests

    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore
    _REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Local deterministic knowledge base
# ---------------------------------------------------------------------------
# Keys are keyword sets; the first matching entry wins. Keep every answer
# short, plain-language, and free of anything resembling a diagnosis.

_KNOWLEDGE_BASE = [
    (
        {"quality", "blur", "blurry", "dark", "bright", "glare", "resolution"},
        "Image quality checks (blur, brightness, contrast, glare, resolution) use standard "
        "computer-vision measurements, not a medical judgment. A 'poor' result means the photo "
        "itself is hard to analyze — usually fixed by retaking it in better light, holding the "
        "camera steady, and filling the frame.",
    ),
    (
        {"gradcam", "grad-cam", "attention", "heatmap", "highlight"},
        "The AI attention heatmap (Grad-CAM) shows which parts of the image most influenced the "
        "model's prediction — it visualizes the model's own process. It is not a detector of "
        "abnormal tissue, and a highlighted area is not evidence of a medical finding.",
    ),
    (
        {"similar", "similarity", "match", "matches"},
        "Similar-image results are found by comparing the visual pattern of your photo to a "
        "reference set using the model's internal representation. A visual match does NOT mean "
        "two lesions share the same diagnosis — it only means they look alike to the model.",
    ),
    (
        {"category", "categories", "melanoma", "carcinoma", "keratosis", "class", "classes"},
        "The three categories the demo model can output (Melanoma, Basal Cell Carcinoma, Benign "
        "Keratosis) are visual pattern categories used for this prototype, not a full diagnostic "
        "taxonomy. A category name here means 'the model's output resembles this pattern,' not "
        "a confirmed diagnosis.",
    ),
    (
        {"confidence", "accurate", "accuracy", "reliable", "trust", "clinically", "validated"},
        "This model has not been clinically validated. Unless the health check explicitly "
        "reports 'trained', no learned lesion prediction is available and no confidence number "
        "should be treated as evidence. Confidence scores, when available, reflect the model's "
        "own certainty, not real-world correctness.",
    ),
    (
        {"triage", "red", "yellow", "green", "tier", "priority"},
        "The triage color is a review-priority suggestion built from transparent rules — image "
        "quality, the model's visual category, reported symptoms, and questionnaire answers. Red "
        "means 'consider professional evaluation soon,' not 'this is confirmed serious.' Every "
        "tier lists the specific factors that produced it.",
    ),
    (
        {"limitation", "limitations", "wrong", "mistake", "error"},
        "Key limitations: the demo model may be untrained, the training/reference dataset is "
        "small or synthetic, similarity search needs a reference index that may not be built yet, "
        "and none of this has undergone clinical validation. Treat every output as a prompt to "
        "seek a second, human opinion — not as a result to act on directly.",
    ),
    (
        {"privacy", "data", "stored", "delete", "store"},
        "See the Privacy panel in the app for the exact, technically accurate statement about "
        "what this specific deployment does with your image and answers.",
    ),
    (
        {"professional", "doctor", "evaluation", "see a doctor", "clinician"},
        "This tool cannot tell you whether you need to see a clinician. As a general rule, any "
        "lesion that is changing, bleeding, painful, or that you're personally concerned about is "
        "worth having a qualified professional look at, regardless of what any app says.",
    ),
]

_DEFAULT_ANSWER = (
    "I can explain how this tool's image quality check, AI category prediction, attention "
    "heatmap, similarity search, and triage rules work, and what their limitations are — ask me "
    "about any of those. I can't diagnose a condition or tell you what your specific result means "
    "medically."
)


def _local_answer(question: str) -> str:
    lowered = question.lower()
    for keywords, answer in _KNOWLEDGE_BASE:
        if any(keyword in lowered for keyword in keywords):
            return answer
    return _DEFAULT_ANSWER


_SYSTEM_PROMPT = (
    "You are an explanatory assistant embedded in an educational skin-lesion triage research "
    "prototype. You explain what the tool's components do (image quality checks, a CNN category "
    "prediction, Grad-CAM attention heatmaps, embedding-based similarity search, and a rule-based "
    "triage engine) and their limitations. "
    "You must NEVER diagnose a medical condition, NEVER state or imply someone has or does not "
    "have a disease, NEVER invent clinical facts, accuracy figures, or statistics that were not "
    "given to you in context, and ALWAYS recommend that medical concerns be evaluated by a "
    "qualified professional rather than by this tool. Keep answers short and in plain language."
)


def _external_answer(question: str, context: Optional[Dict[str, Any]]) -> Optional[str]:
    if not (_REQUESTS_AVAILABLE and config.assistant_external_configured()):
        return None

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "system", "content": f"Current result context (for reference only): {context}"})
    messages.append({"role": "user", "content": question})

    try:
        response = requests.post(
            f"{config.OPENAI_BASE_URL.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": config.MODEL_NAME, "messages": messages, "temperature": 0.2, "max_tokens": 300},
            timeout=config.ASSISTANT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # pragma: no cover
        logger.warning("External assistant call failed (%s); falling back to local answers.", exc)
        return None


def answer_question(question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Return {"answer": str, "provider": "local" | "openai"}."""
    if config.assistant_external_configured():
        external = _external_answer(question, context)
        if external:
            return {"answer": external, "provider": "openai"}

    return {"answer": _local_answer(question), "provider": "local"}
