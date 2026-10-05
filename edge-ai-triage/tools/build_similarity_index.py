"""
Build a FAISS visual-similarity index from a labelled image manifest.

This is an OFFLINE preparation script. It requires a trained lesion model and
FAISS. It never changes the model or the source dataset.

Example (from project root):
    python tools/build_similarity_index.py \
        --manifest data/manifest.csv \
        --data-root data/images
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config
from backend.model import get_embedding, load_model, model_status
from backend.preprocess import preprocess_image
from backend.similarity import build_index_from_embeddings


def thumbnail_base64(image: Image.Image, size: int = 192) -> str:
    image = image.convert("RGB")
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=78, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Edge AI Triage FAISS similarity index.")
    parser.add_argument("--manifest", required=True, help="CSV containing image_path and label columns")
    parser.add_argument("--data-root", required=True, help="Root used to resolve relative image_path values")
    parser.add_argument("--top-k", type=int, default=config.SIMILARITY_TOP_K)
    args = parser.parse_args()

    status = model_status()
    if status["mode"] != "trained":
        raise SystemExit("A trained lesion_classifier.pt is required before building a similarity index.")

    model = load_model()
    if model is None:
        raise SystemExit("The trained model could not be loaded.")

    try:
        import faiss  # noqa: F401
    except ImportError as exc:
        raise SystemExit("FAISS is required. Install it separately with: pip install faiss-cpu") from exc

    manifest = Path(args.manifest)
    root = Path(args.data_root)
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")

    embeddings = []
    metadata = []
    with manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "image_path" not in fields or "label" not in fields:
            raise SystemExit("Manifest must contain image_path and label columns.")

        for row_number, row in enumerate(reader, start=2):
            path = Path(row["image_path"])
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise SystemExit(f"Missing image on row {row_number}: {path}")
            try:
                image_bytes = path.read_bytes()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                tensor = preprocess_image(image_bytes)
                with torch.no_grad():
                    embedding = np.asarray(get_embedding(model, tensor), dtype="float32")
                embeddings.append(embedding)
                metadata.append({
                    "label": row["label"].strip(),
                    "source": str(path),
                    "thumbnail_base64": thumbnail_base64(image),
                })
            except Exception as exc:
                raise SystemExit(f"Could not process row {row_number} ({path}): {exc}") from exc

    if not embeddings:
        raise SystemExit("Manifest contained no usable images.")

    matrix = np.vstack(embeddings).astype("float32")
    # Normalize so IndexFlatIP behaves like cosine similarity.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)

    # Store a cosine-similarity index rather than inventing a nonlinear L2
    # 'similarity score'. Search code maps cosine [-1,1] into a display score.
    config.SIMILARITY_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(config.SIMILARITY_INDEX_FILE))
    config.SIMILARITY_METADATA_FILE.write_text(
        __import__("json").dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Built index with {len(metadata)} images.")
    print(f"Index: {config.SIMILARITY_INDEX_FILE}")
    print(f"Metadata: {config.SIMILARITY_METADATA_FILE}")


if __name__ == "__main__":
    main()
