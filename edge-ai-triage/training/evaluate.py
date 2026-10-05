"""
training/evaluate.py
---------------------
Metrics used by train.py, plus a standalone CLI to evaluate an existing
checkpoint against a manifest (e.g. a held-out test set from a different
collection run).

Metrics are computed from scratch with numpy — no scikit-learn dependency —
so this stays consistent with the project's "don't add unnecessary
frameworks" constraint. For research use beyond this prototype, swapping in
scikit-learn's implementations is a reasonable, well-tested alternative.

Usage
-----
    python evaluate.py --manifest ../data/manifest.csv --data-root ../data/images \\
        --weights ../models/lesion_classifier.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from model import LESION_CLASSES, LesionClassifier  # noqa: E402

from dataset import LesionDataset  # noqa: E402

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def precision_recall_f1_per_class(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    num_classes = matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for c in range(num_classes):
        tp = matrix[c, c]
        fp = matrix[:, c].sum() - tp
        fn = matrix[c, :].sum() - tp
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (
            2 * precision[c] * recall[c] / (precision[c] + recall[c])
            if (precision[c] + recall[c]) > 0
            else 0.0
        )

    return {"precision": precision, "recall": recall, "f1": f1}


def roc_auc_one_vs_rest(y_true: np.ndarray, y_probs: np.ndarray, class_index: int) -> float:
    """
    Compute ROC-AUC for one class vs. the rest via the Mann-Whitney U
    statistic, which is equivalent to the area under the ROC curve and
    avoids needing to sweep thresholds by hand.
    """
    binary_labels = (y_true == class_index).astype(int)
    scores = y_probs[:, class_index]

    n_pos = binary_labels.sum()
    n_neg = len(binary_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # undefined without both classes present

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Average ranks for ties
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_ranks_by_value = np.zeros(len(counts))
    np.add.at(sum_ranks_by_value, inverse, ranks)
    avg_rank_by_value = sum_ranks_by_value / counts
    ranks = avg_rank_by_value[inverse]

    sum_ranks_pos = ranks[binary_labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, class_names: List[str]) -> Dict:
    y_pred = np.argmax(y_probs, axis=1)
    num_classes = len(class_names)

    matrix = confusion_matrix(y_true, y_pred, num_classes)
    per_class = precision_recall_f1_per_class(matrix)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    roc_auc = {
        class_names[c]: roc_auc_one_vs_rest(y_true, y_probs, c) for c in range(num_classes)
    }

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.nanmean(per_class["precision"])),
        "macro_recall": float(np.nanmean(per_class["recall"])),
        "macro_f1": float(np.nanmean(per_class["f1"])),
        "per_class": {
            class_names[c]: {
                "precision": float(per_class["precision"][c]),
                "recall": float(per_class["recall"][c]),
                "f1": float(per_class["f1"][c]),
            }
            for c in range(num_classes)
        },
        "roc_auc_one_vs_rest": roc_auc,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": class_names,
    }


def _load_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


@torch.no_grad()
def _run_inference(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint against a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--weights", required=True, help="Path to a lesion_classifier.pt checkpoint")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LesionDataset(args.manifest, args.data_root, transform=_load_eval_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = LesionClassifier(pretrained_backbone=False).to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)

    y_true, y_probs = _run_inference(model, loader, device)
    metrics = compute_metrics(y_true, y_probs, LESION_CLASSES)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
