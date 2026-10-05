# Edge AI Triage — Research Prototype

An educational/research prototype for offline-first skin-lesion triage
assistance, combining a lightweight CNN, a heuristic image-quality gate,
Grad-CAM explainability, optional visual similarity search, a deterministic
WHO IMCI/IMAI respiratory rule engine, a structured lesion questionnaire,
and a "explain the tool" AI assistant.

**This is not a medical diagnostic device.** No part of it has been
clinically validated, and the app is built to say so at every layer rather
than imply otherwise — see [Limitations](#limitations) and
[Medical safety](#medical-safety-design) below.

---

## Table of contents

- [Architecture](#architecture)
- [Installation (Windows)](#installation-windows)
- [Installation (macOS/Linux)](#installation-macoslinux)
- [Running the app](#running-the-app)
- [Model setup](#model-setup)
- [Dataset setup](#dataset-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Explainability (Grad-CAM)](#explainability-grad-cam)
- [Similarity search](#similarity-search)
- [AI assistant](#ai-assistant)
- [Accessibility](#accessibility)
- [Privacy](#privacy)
- [API reference](#api-reference)
- [Testing](#testing)
- [Limitations](#limitations)
- [Medical safety design](#medical-safety-design)
- [Research methodology](#research-methodology)
- [Future work](#future-work)

---

## Architecture

```text
                          frontend/index.html + static/
                          (camera, data-saver, i18n EN/HI, a11y)
                                       │
                                       │ fetch()
                                       ▼
                        ┌───────────────────────────┐
                        │     backend/inference.py    │
                        │     FastAPI + CORS           │
                        └──────────────┬───────────────┘
                                       │
        ┌──────────────┬──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼              ▼
 image_quality.py  model.py      explainability.py similarity.py services/
 (heuristic CV     (MobileNetV3  (Grad-CAM)        (FAISS, gated) assistant.py
  quality gate)     CNN)                                          (local +
        │              │                                          optional LLM)
        └──────┬───────┘
               ▼
         triage.py
   (WHO IMCI rules + lesion
    questionnaire + late fusion)
               │
               ▼
        TriageResponse
    (tier, reason, factors,
     honest disclaimers)
```

Everything under `backend/` is designed to run fully offline: no call in
the default configuration reaches the network. The one opt-in exception is
the AI assistant, which only calls out if you explicitly set
`AI_PROVIDER=openai` and an API key (see [AI assistant](#ai-assistant)).

## Installation (Windows)

Requires **Python 3.12**.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Installation (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the app

From the project root (recommended — matches the package-mode imports used
throughout `backend/`):

```powershell
python -m uvicorn backend.inference:app --reload --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000/** — the backend serves the frontend
directly, so there's nothing else to start.

(Alternative, also supported: `cd backend` then
`uvicorn inference:app --reload --host 127.0.0.1 --port 8000` — every
backend module supports both import styles.)

## Model setup

See `models/README.md` in full. Short version: no weights are bundled. The
app starts safely without a checkpoint, but learned lesion inference is
disabled until you train one and place the resulting `lesion_classifier.pt`
in `models/`. `/api/health` reports `model_mode: "unavailable"` and
`model_loaded: false` in that state.

## Dataset setup

See `data/README.md` in full. No dataset is downloaded or bundled — you
supply your own properly licensed images and a `manifest.csv`
(`image_path,label` columns; label is one of `Melanoma`,
`Basal Cell Carcinoma`, `Benign Keratosis`). `data/dataset.csv` is
**synthetic demo metadata only**, unrelated to training.

## Training

```bash
cd training
python train.py --manifest ../data/manifest.csv --data-root ../data/images --output ../models --epochs 20 --batch-size 16
```

Includes: patient-level splitting when `patient_id` is available (otherwise
a clearly warned image-level stratified split), augmentation (flips, rotation,
color jitter), class-balanced sampling, checkpoint saving on best validation
macro-F1, early stopping, and a full metrics report (accuracy,
precision/recall/F1, confusion matrix, ROC-AUC) written to
`models/training_history.json`. Run `python train.py --help` for every
option.

## Evaluation

Evaluate any checkpoint independently:

```bash
cd training
python evaluate.py --manifest ../data/test_manifest.csv --data-root ../data/images --weights ../models/lesion_classifier.pt
```

Prints the same metrics `train.py` computes, from scratch (no scikit-learn
dependency) so results are easy to audit.

## Explainability (Grad-CAM)

`POST /api/explain` returns a Grad-CAM heatmap overlay for the model's
top-predicted class **only when a trained checkpoint is installed**, implemented from scratch against MobileNetV3's last
convolutional block (`backend/explainability.py`) — no extra dependency.
The frontend's "Show AI attention" button displays the original photo next
to the heatmap with the required disclaimer: this shows where the *model*
looked, not evidence of a medical finding.

## Similarity search

`POST /api/similarity` looks up visually similar reference images via a
FAISS cosine-similarity index over CNN embeddings (`backend/similarity.py`). Out of
the box this reports:

```json
{ "available": false, "message": "Similarity search is not configured yet. ..." }
```

because no reference index ships with the project. Build one with
`python tools/build_similarity_index.py ...` after training (see
`models/README.md`). `faiss-cpu` remains an optional dependency.

## AI assistant

`POST /api/assistant` explains the tool — not medical facts. Two modes,
controlled entirely by environment variables (never hard-coded):

```bash
# Default: deterministic local knowledge base, no network calls at all
AI_PROVIDER=local

# Optional: route questions through an OpenAI-compatible chat endpoint
AI_PROVIDER=openai
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1   # or any compatible endpoint
```

If the external call fails for any reason (no key, network error, timeout),
the assistant silently falls back to the local knowledge base rather than
erroring. The system prompt used for the external mode explicitly forbids
diagnosis and invented facts (see `backend/services/assistant.py`).

## Accessibility

- English/Hindi toggle (`frontend/static/i18n.js`) — UI chrome is
  hand-translated; clinical category names and API-generated disclaimers
  are deliberately left in English rather than machine-translated.
- Semantic HTML (`<fieldset>`/`<legend>`, `<details>`/`<summary>`,
  `aria-live` regions on progress and results, a skip-to-content link).
- Visible focus states (`:focus-visible`), keyboard-operable controls only
  (no click-only interactions), and `prefers-reduced-motion` support.
- 🔊 Read-aloud buttons next to the AI analysis, triage result, and
  assistant answer, using the browser's native Web Speech API (no external
  TTS service, no account needed).
- 📷 Camera capture via `getUserMedia`, with a graceful fallback message
  and plain file upload when the camera is unavailable or permission is
  denied.
- **Data Saver Mode**: resizes/compresses the photo client-side (canvas,
  max 900px edge, JPEG quality 0.72) before upload, and shows
  `Original: X MB — Compressed: Y KB` so the trade-off is visible.

## Privacy

The in-app Privacy panel states plainly: images and questionnaire answers
are processed **in memory only** and are never written to disk by this
application, and the page uses no tracking cookies or analytics. That
statement is only shown because it's actually true of the current
implementation — check `backend/inference.py` yourself; nothing there
persists an uploaded image. The one caveat, also stated in-app: if you (the
operator) turn on the external AI assistant, assistant *questions* (not
images) may leave the machine.

## API reference

| Method | Path                | Purpose                                             |
|--------|---------------------|------------------------------------------------------|
| GET    | `/api/health`       | Model/assistant/similarity status                    |
| POST   | `/api/image-quality`| Heuristic image quality check only                   |
| POST   | `/api/triage`        | Full pipeline: quality + CNN + WHO rules + fusion     |
| POST   | `/api/explain`       | Grad-CAM heatmap for the top-predicted class          |
| POST   | `/api/similarity`    | Visual similarity search (if configured)              |
| POST   | `/api/assistant`     | "Explain the tool" Q&A                               |

All endpoints validate file type/size (`TRIAGE_MAX_IMAGE_MB`, default 12MB)
and return structured HTTP errors (400/413/422) rather than crashing on bad
input. See `backend/schemas.py` for exact response shapes and interactive
docs at `http://127.0.0.1:8000/docs` once the server is running.

## Testing

Manual smoke test sequence (all verified against this codebase):

```bash
curl http://127.0.0.1:8000/api/health
curl -F "image=@lesion.jpg" http://127.0.0.1:8000/api/image-quality
curl -F "image=@lesion.jpg" -F 'symptoms={}' -F 'questionnaire={}' http://127.0.0.1:8000/api/triage
curl -F "image=@lesion.jpg" http://127.0.0.1:8000/api/explain
curl -F "image=@lesion.jpg" http://127.0.0.1:8000/api/similarity
curl -X POST http://127.0.0.1:8000/api/assistant -H "Content-Type: application/json" -d '{"question":"what does the heatmap mean?"}'

# Error paths
curl -F "image=@not_an_image.txt" http://127.0.0.1:8000/api/image-quality   # -> 400
curl -X POST http://127.0.0.1:8000/api/triage -F 'symptoms={}'                # -> 422 (missing image)
curl -F "image=@huge_file.png" http://127.0.0.1:8000/api/image-quality       # -> 413 (if over the size limit)
```

Then open `http://127.0.0.1:8000/` in a browser, upload or capture a photo,
fill the questionnaire, run the analysis, and click through "Show AI
attention," the assistant chips, the language toggle, and the Privacy
panel. Check the browser console for errors as you go.

## Limitations

- The lesion classifier ships **without trained weights** — `/api/health` will say
  `"model_mode": "unavailable"` until you train it. The app does not run
  random/untrained lesion predictions. Even once trained,
  `"clinically_validated"` never becomes `true` in this codebase; that
  would require real clinical validation, not just a training run.
- No reference dataset or similarity index ships with the project;
  `/api/similarity` is honest about that rather than fabricating matches.
- The respiratory WHO IMCI rules and the lesion questionnaire rules are
  deliberately simple, illustrative rule sets — not a substitute for
  full IMCI/IMAI protocols or clinical judgment.
- Grad-CAM shows model attention, not ground truth about tissue.
- The optional external AI assistant depends on whatever model you point
  it at; the local mode has no such dependency but is limited to its fixed
  knowledge base.

## Medical safety design

Every layer of this app is written to avoid overstating what it knows:

- The model never reports a "diagnosis" — only a category name plus an
  explicit "not clinically validated" disclaimer.
- The triage tiers are phrased as review-priority language
  ("Refer urgently" / "Review soon" / "Routine"), never as a claim about
  disease presence or absence.
- Every triage result carries its own "Why this result?" breakdown, so a
  tier is always traceable to specific rules and the model's stated
  confidence — never an opaque score.
- The assistant's system prompt (external mode) and knowledge base (local
  mode) both explicitly refuse diagnosis and forbid inventing clinical
  facts, accuracy figures, or statistics not already in context.

## Research methodology

See `research/methodology.md` for the source paper's four core
contributions (offline-first edge deployment, decoupled multimodal fusion,
model compression pipeline, human-in-the-loop safety governance) and how
this codebase maps onto each.

## Future work

- **Offline/edge export**: the model is a standard `torch.nn.Module`, so
  exporting to **ONNX** (`torch.onnx.export`) or converting further to
  **TensorFlow Lite** for actual Raspberry Pi/Jetson Nano deployment is a
  natural next step; this repo does not currently ship that export script.
  A **browser/WebGPU/WASM** path (e.g. via ONNX Runtime Web) is a viable
  longer-term option for true offline-in-browser inference, but is not
  implemented here — claiming otherwise would be dishonest about the
  current state of the code.
- INT8 quantization and knowledge distillation (per
  `research/methodology.md`) as a deployment-time step once real training
  data exists.
- A real similarity-index build script against a licensed reference
  dataset, plus periodic index refresh tooling.
- Shadow-mode logging (compare model output against CHW/clinician judgment
  without acting on it) before any real-world pilot.
