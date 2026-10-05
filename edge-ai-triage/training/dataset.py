"""
training/dataset.py
--------------------
A configurable PyTorch Dataset for the lesion classifier.

Expects a CSV manifest with, at minimum, these columns:
    image_path   - path to the image, relative to --data-root (or absolute)
    label        - one of backend.model.LESION_CLASSES

Any additional columns (patient_id, age, etc.) are ignored by the model but
kept available on the Dataset object for later analysis.

This file intentionally does NOT download or bundle any dataset. Point
--manifest / --data-root at a properly licensed dataset you have obtained
yourself (see ../data/README.md).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from model import LESION_CLASSES  # noqa: E402

LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LESION_CLASSES)}


class LesionDataset(Dataset):
    """
    Loads (image, label_index) pairs from a CSV manifest.

    Parameters
    ----------
    manifest_path: path to a CSV with `image_path` and `label` columns.
    data_root: directory that relative image_path values are resolved against.
    transform: torchvision-style transform applied to the loaded PIL image.
    """

    def __init__(self, manifest_path: str, data_root: str, transform: Optional[Callable] = None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.records: List[Dict[str, str]] = []

        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. See data/README.md for how to set up a dataset."
            )

        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "image_path" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
                raise ValueError("Manifest CSV must contain at least `image_path` and `label` columns.")

            for row in reader:
                label = row["label"].strip()
                if label not in LABEL_TO_INDEX:
                    raise ValueError(
                        f"Unknown label '{label}' in manifest. Expected one of {list(LABEL_TO_INDEX)}."
                    )
                image_path = Path(row["image_path"])
                if not image_path.is_absolute():
                    image_path = self.data_root / image_path
                self.samples.append((image_path, LABEL_TO_INDEX[label]))
                self.records.append(dict(row))

        if not self.samples:
            raise ValueError(f"Manifest at {manifest_path} contained zero usable rows.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label_index = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label_index

    def class_counts(self) -> dict:
        counts = {label: 0 for label in LESION_CLASSES}
        for _, label_index in self.samples:
            counts[LESION_CLASSES[label_index]] += 1
        return counts

    def patient_ids(self) -> List[Optional[str]]:
        """Return optional patient IDs aligned with ``self.samples``.

        Patient-level splitting is preferred whenever the source dataset
        supplies stable patient identifiers. ``None`` means the manifest did
        not provide one.
        """
        values: List[Optional[str]] = []
        for record in self.records:
            value = (record.get("patient_id") or "").strip()
            values.append(value or None)
        return values
