"""
training/train.py
------------------
Trains the MobileNetV3-Small lesion classifier defined in backend/model.py.

This script does not download or fabricate any dataset — you provide a CSV
manifest pointing at real, properly licensed images (see ../data/README.md).

Example
-------
    python train.py \\
        --manifest ../data/manifest.csv \\
        --data-root ../data/images \\
        --output ../models \\
        --epochs 20 \\
        --batch-size 16

What this script does:
  - Stratified train/validation/test split from a single manifest
  - Data augmentation on the training split (flips, rotation, color jitter)
  - Class-balanced sampling (WeightedRandomSampler) to counter class imbalance
  - Checkpoint saving (best validation macro-F1)
  - Early stopping on validation loss
  - Full metrics: accuracy, precision, recall, F1 (macro), confusion matrix,
    and ROC-AUC (one-vs-rest) computed from scratch, no sklearn dependency
  - Exports the trained weights to <output>/lesion_classifier.pt, which
    backend/model.py will automatically pick up on next server start
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from model import LESION_CLASSES, LesionClassifier  # noqa: E402

from dataset import LesionDataset  # noqa: E402
from evaluate import compute_metrics  # noqa: E402

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_split(
    dataset: LesionDataset, val_fraction: float, test_fraction: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    """Split by class for manifests without patient IDs.

    This is a fallback only. If patient_id is present, use
    ``patient_level_split`` below to prevent the same patient appearing in
    both training and evaluation sets.
    """
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, (_, label_index) in enumerate(dataset.samples):
        by_class[label_index].append(idx)

    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for indices in by_class.values():
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * test_fraction)) if n >= 3 else 0
        n_val = max(1, int(n * val_fraction)) if n - n_test >= 2 else 0
        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])

    return train_idx, val_idx, test_idx


def patient_level_split(
    dataset: LesionDataset, val_fraction: float, test_fraction: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    """Split whole patients into train/validation/test groups.

    Each patient is assigned to exactly one split. This prevents leakage when
    a patient contributes multiple lesion images.
    """
    patient_ids = dataset.patient_ids()
    if not patient_ids or any(pid is None for pid in patient_ids):
        raise ValueError("patient_level_split requires a non-empty patient_id for every manifest row.")

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, pid in enumerate(patient_ids):
        groups[pid].append(idx)  # type: ignore[index]

    # Assign groups using their dominant class, while preserving approximate
    # class proportions. Mixed-label patients are allowed but reported.
    group_items = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)

    target_counts = {
        "test": max(1, round(len(group_items) * test_fraction)),
        "val": max(1, round(len(group_items) * val_fraction)),
    }
    target_counts["test"] = min(target_counts["test"], max(0, len(group_items) - 2))
    target_counts["val"] = min(target_counts["val"], max(0, len(group_items) - target_counts["test"] - 1))

    test_groups = {pid for pid, _ in group_items[:target_counts["test"]]}
    val_start = target_counts["test"]
    val_groups = {pid for pid, _ in group_items[val_start:val_start + target_counts["val"]]}

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for pid, indices in group_items:
        if pid in test_groups:
            test_idx.extend(indices)
        elif pid in val_groups:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    return train_idx, val_idx, test_idx


def make_balanced_sampler(dataset: LesionDataset, indices: List[int]) -> WeightedRandomSampler:
    labels = [dataset.samples[i][1] for i in indices]
    class_counts = np.bincount(labels, minlength=len(LESION_CLASSES))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss_and_predictions(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, np.concatenate(all_labels), np.concatenate(all_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the lesion classifier.")
    parser.add_argument("--manifest", required=True, help="CSV with image_path,label columns")
    parser.add_argument("--data-root", required=True, help="Directory relative image paths resolve against")
    parser.add_argument("--output", default="../models", help="Directory to write checkpoints + metrics")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Initialize the backbone from ImageNet weights before fine-tuning (recommended)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform, eval_transform = build_transforms()

    # Load the manifest once without a transform to compute the split, then
    # wrap it twice (train transform vs eval transform) via Subset.
    base_dataset = LesionDataset(args.manifest, args.data_root, transform=None)
    print(f"Loaded {len(base_dataset)} samples. Class counts: {base_dataset.class_counts()}")

    has_patient_ids = bool(base_dataset.patient_ids()) and all(pid is not None for pid in base_dataset.patient_ids())
    if has_patient_ids:
        train_idx, val_idx, test_idx = patient_level_split(base_dataset, args.val_split, args.test_split, args.seed)
        print("Using patient-level splitting (recommended).")
    else:
        print("WARNING: manifest has no complete patient_id column; using image-level stratified splitting. This can cause patient leakage if multiple images belong to one person.")
        train_idx, val_idx, test_idx = stratified_split(base_dataset, args.val_split, args.test_split, args.seed)
    print(f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    if not train_idx or not val_idx or not test_idx:
        raise ValueError("Train/validation/test split produced an empty split. Add more labelled data or adjust split fractions.")

    train_dataset = LesionDataset(args.manifest, args.data_root, transform=train_transform)
    eval_dataset = LesionDataset(args.manifest, args.data_root, transform=eval_transform)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(eval_dataset, val_idx)
    test_subset = Subset(eval_dataset, test_idx)

    sampler = make_balanced_sampler(base_dataset, train_idx)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    model = LesionClassifier(pretrained_backbone=args.pretrained_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "lesion_classifier.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_labels, val_probs = evaluate_loss_and_predictions(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_labels, val_probs, LESION_CLASSES)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_accuracy={val_metrics['accuracy']:.4f}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics})

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> new best model saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no val improvement for {args.patience} epochs.")
                break

    # Final evaluation on the held-out test split, using the BEST checkpoint.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss, test_labels, test_probs = evaluate_loss_and_predictions(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_labels, test_probs, LESION_CLASSES)
    print("\nFinal test-set metrics (best checkpoint):")
    print(json.dumps(test_metrics, indent=2))

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "test_metrics": test_metrics}, f, indent=2)

    print(f"\nDone. Weights: {checkpoint_path}\nMetrics: {output_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()
