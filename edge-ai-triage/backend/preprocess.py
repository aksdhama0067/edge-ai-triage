"""
preprocess.py
--------------
Image preprocessing (resize + ImageNet normalization), a lightweight
blur/quality gate using OpenCV Laplacian variance, and symptom-slot
parsing/standardization for the respiratory triage pathway.

Designed to run on constrained edge hardware (Raspberry Pi / Jetson Nano),
so we avoid heavy dependencies where possible and keep everything CPU-friendly.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python-headless is required for the image quality gate. "
        "Install it with `pip install opencv-python-headless`."
    ) from exc

import torch
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Below this Laplacian variance, an image is considered too blurry to be
# clinically useful. This threshold was chosen empirically for phone-camera
# lesion photography and can be tuned per-deployment.
BLUR_VARIANCE_THRESHOLD = 100.0

_PREPROCESS_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load raw image bytes into a PIL Image, normalized to RGB."""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    return image


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Resize to 224x224 and apply standard ImageNet normalization.

    Returns a 4D tensor of shape (1, 3, 224, 224) ready for model inference.
    """
    image = load_image_from_bytes(image_bytes)
    tensor = _PREPROCESS_TRANSFORM(image)
    return tensor.unsqueeze(0)


def check_image_quality(image_bytes: bytes) -> Tuple[bool, str]:
    """
    Mock but functional image-quality gate using OpenCV's Laplacian variance
    as a blur estimator. Low-bandwidth clinics often submit compressed or
    motion-blurred phone photos, so this gate protects the CNN from being
    trusted on unusable input.

    Returns
    -------
    (quality_passed, message)
    """
    try:
        image = load_image_from_bytes(image_bytes)
    except Exception:
        return False, "Image could not be read. Please retake the photo."

    np_image = np.array(image)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    if laplacian_variance < BLUR_VARIANCE_THRESHOLD:
        return (
            False,
            f"Image appears too blurry (sharpness score {laplacian_variance:.1f}, "
            f"minimum {BLUR_VARIANCE_THRESHOLD:.0f}). Please retake in better light "
            "and hold the camera steady.",
        )

    height, width = gray.shape[:2]
    if height < 100 or width < 100:
        return False, "Image resolution is too low. Please retake at a higher resolution."

    return True, "Image quality acceptable"


# ---------------------------------------------------------------------------
# Symptom parsing
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
    Standardize raw, loosely-typed symptom input from the frontend form into
    clean boolean/numeric feature slots consumed by the WHO IMCI rule engine
    and the late-fusion triage module.

    Expected raw keys (all optional, missing -> safe defaults):
      - cough (bool-ish)
      - cough_days (number)
      - fever (bool-ish)
      - breathing_difficulty (one of "none", "mild", "severe", or bool-ish)
      - chest_indrawing (bool-ish)
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
