"""
preprocess.py
--------------
Image tensor preparation (resize + ImageNet normalization) for the CNN, and
symptom-slot parsing/standardization for the WHO IMCI respiratory pathway.

Image *quality* assessment (blur, brightness, contrast, glare, resolution)
now lives in image_quality.py, which is a broader heuristic module than the
single blur check this file used to contain.
"""

from __future__ import annotations

import io
from typing import Any, Dict

import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_PREPROCESS_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load raw image bytes into a PIL Image, normalized to RGB."""
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Resize to 224x224 and apply standard ImageNet normalization.
    Returns a 4D tensor of shape (1, 3, 224, 224) ready for model inference.
    """
    image = load_image_from_bytes(image_bytes)
    tensor = _PREPROCESS_TRANSFORM(image)
    return tensor.unsqueeze(0)


# ---------------------------------------------------------------------------
# Symptom parsing (WHO IMCI respiratory pathway)
# ---------------------------------------------------------------------------

def _to_bool(value: Any) -> bool:
    """Coerce loosely-typed form input (str/int/bool) into a clean bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on", "y"}
    return False


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_symptom_slots(symptom_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardize raw, loosely-typed respiratory symptom input from the
    frontend form into clean boolean/numeric feature slots consumed by the
    WHO IMCI rule engine and the late-fusion triage module.
    """
    breathing_raw = symptom_dict.get("breathing_difficulty", "none")
    if isinstance(breathing_raw, str):
        breathing_level = breathing_raw.strip().lower()
        if breathing_level not in {"none", "mild", "severe"}:
            breathing_level = "severe" if _to_bool(breathing_raw) else "none"
    else:
        breathing_level = "severe" if _to_bool(breathing_raw) else "none"

    return {
        "cough": _to_bool(symptom_dict.get("cough", False)),
        "cough_days": max(0, int(_to_float(symptom_dict.get("cough_days", 0)))),
        "fever": _to_bool(symptom_dict.get("fever", False)),
        "breathing_difficulty": breathing_level,  # "none" | "mild" | "severe"
        "breathing_difficulty_severe": breathing_level == "severe",
        "chest_indrawing": _to_bool(symptom_dict.get("chest_indrawing", False)),
    }


def parse_lesion_questionnaire(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize the dermatology follow-up questionnaire into clean types."""
    return {
        "recent_change": _to_bool(raw.get("recent_change", False)),
        "size_change": _to_bool(raw.get("size_change", False)),
        "appearance_change": _to_bool(raw.get("appearance_change", False)),
        "painful": _to_bool(raw.get("painful", False)),
        "itchy": _to_bool(raw.get("itchy", False)),
        "bleeding": _to_bool(raw.get("bleeding", False)),
        "duration_days": max(0, int(_to_float(raw.get("duration_days", 0)))),
    }
