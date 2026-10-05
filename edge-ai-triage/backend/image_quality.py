"""
image_quality.py
-----------------
Image Quality Assessment — a heuristic computer-vision module, NOT a trained
neural network and NOT a medical diagnosis. It estimates whether a photo is
usable for the downstream lesion classifier and gives a CHW/patient plain-
language reason and recommendation when it isn't.

Every check here is a classical, inspectable CV technique:
  - blur          -> Laplacian variance
  - brightness    -> mean grayscale intensity
  - contrast      -> grayscale standard deviation
  - glare/exposure-> fraction of near-saturated (overexposed) pixels
  - resolution    -> raw pixel dimensions
  - visible subject-> fraction of the frame that isn't near-uniform background,
                      as a very rough proxy for "is there a subject filling
                      the frame" (not lesion detection)

The combined 0-100 score and "good"/"warning"/"poor" bucket are a weighted
heuristic, not a calibrated probability.
"""

from __future__ import annotations

import io
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python-headless is required for image quality assessment. "
        "Install it with `pip install opencv-python-headless`."
    ) from exc

# ---------------------------------------------------------------------------
# Thresholds (tunable; chosen for typical phone-camera lesion photography)
# ---------------------------------------------------------------------------

BLUR_VARIANCE_MIN = 100.0          # below this: flagged as blurry
BRIGHTNESS_LOW = 40.0               # 0-255 grayscale mean
BRIGHTNESS_HIGH = 220.0
CONTRAST_MIN = 15.0                 # grayscale standard deviation
GLARE_PIXEL_THRESHOLD = 250         # pixel value considered "blown out"
GLARE_FRACTION_MAX = 0.06           # fraction of pixels allowed to be blown out
MIN_WIDTH = 200
MIN_HEIGHT = 200
MIN_SUBJECT_FRACTION = 0.03          # rough "is anything but flat background here"


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


def _score_penalty(condition: bool, penalty: int) -> int:
    return penalty if condition else 0


def assess_image_quality(image_bytes: bytes) -> Dict:
    """
    Run the full heuristic quality assessment.

    Returns a dict matching schemas.QualityGateResult:
        {
          "quality": "good" | "warning" | "poor",
          "score": 0-100,
          "issues": [str, ...],
          "recommendation": str,
          "metrics": {raw numeric measurements, for transparency/debugging},
        }
    """
    try:
        image = load_image_from_bytes(image_bytes)
    except Exception:
        return {
            "quality": "poor",
            "score": 0,
            "issues": ["unreadable_image"],
            "recommendation": "This file could not be read as an image. Please try a different photo.",
            "metrics": {},
        }

    np_rgb = np.array(image)
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]

    issues: List[str] = []
    score = 100

    # --- Resolution ---------------------------------------------------
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        issues.append("low_resolution")
        score -= 25

    # --- Blur (Laplacian variance) -------------------------------------
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_variance < BLUR_VARIANCE_MIN:
        issues.append("blurry")
        score -= 30

    # --- Brightness ------------------------------------------------------
    brightness = float(gray.mean())
    if brightness < BRIGHTNESS_LOW:
        issues.append("too_dark")
        score -= 20
    elif brightness > BRIGHTNESS_HIGH:
        issues.append("overexposed")
        score -= 15

    # --- Contrast ----------------------------------------------------------
    contrast = float(gray.std())
    if contrast < CONTRAST_MIN:
        issues.append("low_contrast")
        score -= 15

    # --- Glare / highlight clipping -----------------------------------------
    glare_fraction = float(np.mean(gray >= GLARE_PIXEL_THRESHOLD))
    if glare_fraction > GLARE_FRACTION_MAX:
        issues.append("glare")
        score -= 15

    # --- Rough "is there a visible subject" proxy -----------------------------
    # Uses edge density as a cheap stand-in for "the frame isn't just a flat
    # wall/table" — this is NOT lesion detection, just a usability signal.
    edges = cv2.Canny(gray, 50, 150)
    edge_fraction = float(np.mean(edges > 0))
    if edge_fraction < MIN_SUBJECT_FRACTION:
        issues.append("no_clear_subject")
        score -= 15

    score = max(0, min(100, score))

    if score >= 75 and not issues:
        quality = "good"
    elif score >= 45:
        quality = "warning"
    else:
        quality = "poor"

    recommendation = _build_recommendation(quality, issues)

    return {
        "quality": quality,
        "score": score,
        "issues": issues,
        "recommendation": recommendation,
        "metrics": {
            "width": float(width),
            "height": float(height),
            "blur_variance": round(laplacian_variance, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "glare_fraction": round(glare_fraction, 4),
            "edge_fraction": round(edge_fraction, 4),
        },
    }


_ISSUE_MESSAGES = {
    "unreadable_image": "the file could not be read as an image",
    "low_resolution": "the image resolution is very low",
    "blurry": "the image appears blurry",
    "too_dark": "the image is too dark",
    "overexposed": "the image is overexposed",
    "low_contrast": "the image has very low contrast",
    "glare": "there is significant glare or a blown-out highlight",
    "no_clear_subject": "no clear subject fills the frame",
}


def _build_recommendation(quality: str, issues: List[str]) -> str:
    if quality == "good":
        return "Image quality looks good for analysis."

    reasons = "; ".join(_ISSUE_MESSAGES.get(i, i) for i in issues) or "quality looks reduced"
    if quality == "warning":
        return (
            f"Image quality is usable but not ideal ({reasons}). "
            "Consider retaking the photo in better lighting, holding the camera steady."
        )
    return (
        f"Image quality is too low for a reliable analysis ({reasons}). "
        "Please retake the photo: use natural light, hold the camera steady, "
        "and fill the frame with the area of interest."
    )
