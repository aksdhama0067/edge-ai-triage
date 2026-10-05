"""
explainability.py
------------------
A from-scratch Grad-CAM implementation for the MobileNetV3-based lesion
classifier — no extra heavyweight dependency (e.g. captum) needed.

Grad-CAM highlights which spatial regions of the input most influenced the
model's chosen class, by weighting the last convolutional layer's feature
maps by the gradient of that class's score with respect to those maps.

IMPORTANT — what this does and does not mean:
This shows where the *model* looked when making *its* prediction. It is not
a lesion-detection or abnormality-detection tool, and a highlighted region
is not evidence of a medical finding — especially given the model may be
untrained (see model.model_status()). The UI-facing disclaimer this module
returns must always ship with any heatmap.
"""

from __future__ import annotations

import base64
import io
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python-headless is required for Grad-CAM overlays.") from exc

try:
    from .model import LESION_CLASSES, LesionClassifier
except ImportError:
    from model import LESION_CLASSES, LesionClassifier


class GradCAM:
    """Minimal Grad-CAM: registers forward/backward hooks on one target layer."""

    def __init__(self, model: LesionClassifier):
        self.model = model
        self.target_layer = model.gradcam_target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input_, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_index: Optional[int] = None):
        self.model.zero_grad(set_to_none=True)
        image_tensor = image_tensor.clone().requires_grad_(True)

        logits = self.model(image_tensor)
        if class_index is None:
            class_index = int(torch.argmax(logits, dim=1).item())

        score = logits[0, class_index]
        score.backward()

        activations = self._activations[0]           # (C, H, W)
        gradients = self._gradients[0]                # (C, H, W)
        weights = gradients.mean(dim=(1, 2))          # (C,)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        if float(cam.max()) > 0:
            cam = cam / cam.max()

        probabilities = torch.softmax(logits, dim=1).squeeze(0).detach()
        return cam.numpy(), class_index, float(probabilities[class_index])

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def _overlay_heatmap(original_image: Image.Image, cam: np.ndarray) -> str:
    """Resize the CAM to the original image size, colorize, and blend. Returns base64 PNG."""
    original_rgb = np.array(original_image.convert("RGB"))
    height, width = original_rgb.shape[:2]

    cam_resized = cv2.resize(cam, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(0.55 * original_rgb + 0.45 * heatmap_rgb)
    overlay_image = Image.fromarray(overlay)

    buffer = io.BytesIO()
    overlay_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def generate_gradcam(model: LesionClassifier, image_tensor: torch.Tensor, original_image: Image.Image) -> Dict:
    """
    Run Grad-CAM for the model's top-predicted class and return an overlay.

    Returns
    -------
    {
        "available": True,
        "predicted_class": str,
        "predicted_confidence": float,
        "heatmap_overlay_base64": str,
    }
    """
    gradcam = GradCAM(model)
    try:
        cam, class_index, confidence = gradcam(image_tensor)
    finally:
        gradcam.remove_hooks()

    overlay_b64 = _overlay_heatmap(original_image, cam)

    return {
        "available": True,
        "predicted_class": LESION_CLASSES[class_index],
        "predicted_confidence": round(confidence, 4),
        "heatmap_overlay_base64": overlay_b64,
    }
