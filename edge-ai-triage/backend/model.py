"""
model.py
--------
The lightweight MobileNetV3-Small skin-lesion classifier: definition,
weight loading, prediction, and honest status reporting.

Rule-based triage logic (WHO IMCI danger signs, late fusion, questionnaire
handling) lives in triage.py, not here — this module is only responsible
for the learned vision component.

This module fails safe: if trained weights are not present at
config.WEIGHTS_PATH, the API remains available but learned lesion inference
is disabled. `model_status()` reports this as "unavailable" and callers
must not generate probabilities, Grad-CAM maps, or embeddings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

try:  # package mode: `python -m uvicorn backend.inference:app`
    from . import config
except ImportError:  # script mode: `cd backend && uvicorn inference:app`
    import config

logger = logging.getLogger("edge_ai_triage.model")

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------

LESION_CLASSES: List[str] = [
    "Melanoma",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
]

MALIGNANT_CLASSES = {"Melanoma", "Basal Cell Carcinoma"}

# Confidence thresholds for CNN-only triage (used when no WHO danger sign fires)
RED_CONFIDENCE_THRESHOLD = 0.75
YELLOW_CONFIDENCE_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Lightweight CNN classifier
# ---------------------------------------------------------------------------

class LesionClassifier(nn.Module):
    """
    Thin wrapper around MobileNetV3-Small, chosen for its small memory
    footprint and fast CPU inference on Raspberry Pi / Jetson Nano class
    hardware as well as an ordinary Windows laptop. The classifier head is
    replaced to predict our 3 lesion classes.
    """

    def __init__(self, num_classes: int = len(LESION_CLASSES), pretrained_backbone: bool = False):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
        self.backbone = models.mobilenet_v3_small(weights=weights)

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled feature map (pre-classifier) as an embedding vector."""
        features = self.backbone.features(x)
        pooled = self.backbone.avgpool(features)
        return torch.flatten(pooled, 1)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """The last convolutional block — the standard Grad-CAM target for this architecture."""
        return self.backbone.features[-1]


def load_model() -> Optional[LesionClassifier]:
    """Load a trained lesion classifier, or return ``None`` if weights are absent.

    The previous demo behavior instantiated a fresh classifier head when no
    checkpoint existed. That made the API look functional while allowing
    meaningless random probabilities to reach triage and Grad-CAM. The safer
    behavior is to keep the server healthy but disable learned-lesion
    inference until a real checkpoint is present.
    """
    weights_path = config.WEIGHTS_PATH
    if not weights_path.is_file():
        logger.warning(
            "No trained weights found at %s. Lesion inference, Grad-CAM, and "
            "embedding search are disabled until a trained checkpoint is installed.",
            weights_path,
        )
        return None

    model = LesionClassifier(pretrained_backbone=False)
    try:
        state_dict = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state_dict)
    except Exception as exc:
        logger.error("Failed to load trained weights at %s: %s", weights_path, exc)
        return None

    model.eval()
    logger.info("Loaded trained lesion weights from %s", weights_path)
    return model


def model_status() -> Dict[str, Any]:
    """Return explicit, honest model readiness information for the UI/API."""
    weights_available = config.WEIGHTS_PATH.is_file()
    return {
        "weights_available": weights_available,
        "mode": "trained" if weights_available else "unavailable",
        "clinically_validated": False,
    }


@torch.no_grad()
def predict_lesion(model: LesionClassifier, image_tensor: torch.Tensor) -> Dict[str, float]:
    """Run inference and return a dict of {class_name: probability}."""
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    return {cls: float(probabilities[i]) for i, cls in enumerate(LESION_CLASSES)}


@torch.no_grad()
def get_embedding(model: LesionClassifier, image_tensor: torch.Tensor) -> "list[float]":
    """
    Return a pooled feature-vector embedding for the image, used by the
    similarity-search service. Independent of the classifier head, so it
    remains usable even before the head is trained (though embedding
    *quality* still depends on training/fine-tuning for this domain).
    """
    embedding = model.forward_features(image_tensor).squeeze(0)
    return embedding.tolist()


def top_prediction(lesion_probabilities: Dict[str, float]) -> Tuple[str, float]:
    top_class = max(lesion_probabilities, key=lesion_probabilities.get)
    return top_class, lesion_probabilities[top_class]
