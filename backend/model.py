"""
model.py
--------
- A lightweight MobileNetV3-based skin-lesion classifier.
- A deterministic, auditable WHO IMCI/IMAI danger-sign rule engine for
  respiratory symptoms.
- A late-fusion triage module that combines both into a single triage tier.

This module is written to run offline on edge hardware, and to fail safe:
if pretrained weights are unavailable (e.g. first run, no internet, or
hardware not yet provisioned), it falls back to randomly-initialized
weights so the API can still start and be demoed end-to-end.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger("edge_ai_triage.model")

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------

LESION_CLASSES: List[str] = [
    "Melanoma",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
]

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "lesion_classifier.pt")

# Confidence thresholds for CNN-only triage (used when no WHO danger sign fires)
RED_CONFIDENCE_THRESHOLD = 0.75  # high-confidence malignant-pattern prediction
YELLOW_CONFIDENCE_THRESHOLD = 0.40

MALIGNANT_CLASSES = {"Melanoma", "Basal Cell Carcinoma"}


# ---------------------------------------------------------------------------
# Lightweight CNN classifier
# ---------------------------------------------------------------------------

class LesionClassifier(nn.Module):
    """
    Thin wrapper around MobileNetV3-Small, chosen for its small memory
    footprint and fast CPU inference on Raspberry Pi / Jetson Nano class
    hardware. The classifier head is replaced to predict our 3 lesion classes.
    """

    def __init__(self, num_classes: int = len(LESION_CLASSES), pretrained_backbone: bool = False):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
        self.backbone = models.mobilenet_v3_small(weights=weights)

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_model() -> LesionClassifier:
    """
    Load the lesion classifier, using saved fine-tuned weights if present.
    Falls back to an ImageNet-initialized backbone with a fresh head if no
    weights file exists yet, so the server can start and demo the full
    pipeline before real training data / hardware arrives.
    """
    model = LesionClassifier(pretrained_backbone=False)

    if os.path.exists(WEIGHTS_PATH):
        try:
            state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info("Loaded fine-tuned weights from %s", WEIGHTS_PATH)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load weights at %s (%s); using mock weights.", WEIGHTS_PATH, exc)
    else:
        logger.warning(
            "No trained weights found at %s. Serving with randomly-initialized "
            "mock weights so the API stays up for demo purposes. Predictions "
            "are NOT clinically meaningful until the model is trained.",
            WEIGHTS_PATH,
        )

    model.eval()
    return model


@torch.no_grad()
def predict_lesion(model: LesionClassifier, image_tensor: torch.Tensor) -> Dict[str, float]:
    """Run inference and return a dict of {class_name: probability}."""
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    return {cls: float(probabilities[i]) for i, cls in enumerate(LESION_CLASSES)}


# ---------------------------------------------------------------------------
# Deterministic WHO IMCI / IMAI danger-sign rule engine
# ---------------------------------------------------------------------------

def evaluate_danger_signs(symptoms_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explicit, auditable safety rules loosely based on WHO IMCI/IMAI guidance
    for classifying respiratory danger signs in resource-limited settings.

    This function intentionally contains no learned parameters: every branch
    is inspectable and traceable to a specific clinical rule, so a human
    reviewer can audit exactly why a case was flagged.

    Returns
    -------
    {
        "danger_sign_present": bool,
        "triggered_rules": [str, ...],
    }
    """
    triggered_rules: List[str] = []

    if symptoms_data.get("chest_indrawing"):
        triggered_rules.append("WHO IMCI: chest indrawing present (severe pneumonia indicator)")

    if symptoms_data.get("breathing_difficulty") == "severe":
        triggered_rules.append("WHO IMCI: severe breathing difficulty reported")

    if symptoms_data.get("fever") and symptoms_data.get("cough_days", 0) >= 14:
        triggered_rules.append("WHO IMCI: fever with cough persisting >= 14 days (possible TB referral)")

    if symptoms_data.get("fever") and symptoms_data.get("breathing_difficulty") in {"mild", "severe"}:
        triggered_rules.append("WHO IMCI: fever combined with breathing difficulty")

    return {
        "danger_sign_present": len(triggered_rules) > 0,
        "triggered_rules": triggered_rules,
    }


# ---------------------------------------------------------------------------
# Late-fusion triage module
# ---------------------------------------------------------------------------

def fuse_triage_decision(
    lesion_probabilities: Dict[str, float],
    danger_sign_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combine CNN vision probabilities with WHO danger-sign rule outputs into
    a single triage tier, per the paper's decoupled late-fusion design.

    Decision logic:
      1. Any WHO danger sign present -> force tier "Red", regardless of CNN
         output. Safety rules always override model confidence.
      2. Otherwise, use the CNN's top malignant-class confidence:
         - >= RED_CONFIDENCE_THRESHOLD    -> "Red"    (Refer Urgently)
         - >= YELLOW_CONFIDENCE_THRESHOLD -> "Yellow"  (Review Soon)
         - below that                      -> "Green"  (Routine)
    """
    top_class = max(lesion_probabilities, key=lesion_probabilities.get)
    top_confidence = lesion_probabilities[top_class]
    malignant_confidence = sum(
        prob for cls, prob in lesion_probabilities.items() if cls in MALIGNANT_CLASSES
    )

    if danger_sign_result["danger_sign_present"]:
        rules_text = "; ".join(danger_sign_result["triggered_rules"])
        return {
            "triage_tier": "Red",
            "confidence": round(top_confidence, 4),
            "reason": f"WHO danger sign(s) detected: {rules_text}. Visual pattern: {top_class} "
                      f"({top_confidence:.0%} confidence).",
        }

    if malignant_confidence >= RED_CONFIDENCE_THRESHOLD:
        return {
            "triage_tier": "Red",
            "confidence": round(top_confidence, 4),
            "reason": f"High-confidence visual pattern consistent with {top_class} "
                      f"({top_confidence:.0%} confidence). No respiratory danger signs reported.",
        }

    if malignant_confidence >= YELLOW_CONFIDENCE_THRESHOLD:
        return {
            "triage_tier": "Yellow",
            "confidence": round(top_confidence, 4),
            "reason": f"Moderate-confidence visual pattern ({top_class}, {top_confidence:.0%}). "
                      "Recommend review within a few days.",
        }

    return {
        "triage_tier": "Green",
        "confidence": round(top_confidence, 4),
        "reason": f"Low-risk visual pattern ({top_class}, {top_confidence:.0%}) and no respiratory "
                  "danger signs. Routine follow-up recommended.",
    }
