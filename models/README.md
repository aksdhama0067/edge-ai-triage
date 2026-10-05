# `models/`

This directory holds artifacts the backend loads at runtime. It is **not**
checked into version control with real weights — only this README should
exist here until you add your own.

## Lesion classifier weights

Path expected: `models/lesion_classifier.pt`
(override with the `TRIAGE_WEIGHTS_PATH` environment variable)

- If this file **does not exist**, the backend still starts, but **learned
  lesion inference is disabled**. `/api/health` reports
  `"model_mode": "unavailable"` and `"model_loaded": false`. The app will
  not generate random lesion probabilities, Grad-CAM maps, or similarity
  embeddings from an untrained head.
- If this file **does exist**, it must be a PyTorch `state_dict` produced by
  `training/train.py` (or anything else that saves a compatible
  `LesionClassifier` state dict) for the 3 classes in
  `backend/model.py::LESION_CLASSES`. `/api/health` will then report
  `"model_mode": "trained"`. This still does **not** mean clinically
  validated — that flag is separate and this prototype never sets it to
  `true` automatically.

To produce this file yourself (after obtaining a properly licensed dataset):

```bash
cd training
python train.py --manifest ../data/manifest.csv --data-root ../data/images --output ../models
```

See `../data/README.md` for how to prepare `manifest.csv`.

## Similarity search index (optional)

Path expected: `models/similarity_index/index.faiss` and
`models/similarity_index/metadata.json`
(override the directory with `TRIAGE_SIMILARITY_INDEX_DIR`)

- If these files are missing, or the `faiss` package isn't installed,
  `/api/similarity` returns `{"available": false, ...}` rather than
  fabricating matches.
- Build one only after a **trained** lesion model exists. Use the reproducible
  helper from the project root:

  ```bash
  python tools/build_similarity_index.py --manifest data/manifest.csv --data-root data/images
  ```

  The builder stores normalized embeddings in an inner-product FAISS index,
  which is treated as cosine similarity by the API. It also stores small
  thumbnails and source metadata. Do not include private patient images or
  data you are not permitted to redistribute.

## What NOT to put here

Do not commit real patient images, PHI, or any dataset you don't have clear
rights to redistribute. `models/` should contain only derived artifacts
(weights, indices) — raw source images belong in `../data/` per its own
README, and even there only with a dataset you're properly licensed to use.
