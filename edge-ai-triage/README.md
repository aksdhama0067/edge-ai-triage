# Field Triage — Offline Diagnostic Assistance Prototype

A software prototype of the system described in *Edge AI Diagnostic
Assistance for Low-Bandwidth Rural Clinics*: an offline-first triage tool
that combines a lightweight CNN skin-lesion classifier with a deterministic
WHO IMCI/IMAI respiratory danger-sign rule engine, fused into a single
Red/Yellow/Green triage recommendation for community health workers (CHWs).

Hardware (Raspberry Pi / Jetson Nano) arrival was delayed for this build, so
this prototype runs entirely on a laptop: a FastAPI backend serving the
model and rule engine, and a static HTML/JS frontend that talks to it.

## Architecture

```text
                 ┌──────────────────────────┐
                 │   frontend/index.html     │
                 │   (CHW dashboard, runs    │
                 │   in any local browser)   │
                 └────────────┬─────────────┘
                              │ fetch() multipart/form-data
                              │ POST /api/triage
                              ▼
                 ┌──────────────────────────┐
                 │  backend/inference.py     │
                 │  FastAPI + CORS           │
                 └────────────┬─────────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
   ┌────────────────┐ ┌───────────────┐ ┌─────────────────┐
   │ preprocess.py   │ │ model.py       │ │ model.py         │
   │ quality gate +  │ │ MobileNetV3    │ │ WHO IMCI danger-  │
   │ ImageNet resize │ │ lesion CNN     │ │ sign rule engine  │
   └────────────────┘ └───────┬────────┘ └─────────┬────────┘
                               └─────────┬──────────┘
                                         ▼
                          ┌───────────────────────────┐
                          │ model.py: late-fusion      │
                          │ Red / Yellow / Green tier  │
                          └───────────────────────────┘
```

Everything above the dashed line at deployment time is designed to run
fully offline on a Raspberry Pi 4/5 or Jetson Nano — the FastAPI server and
model swap onto that hardware unchanged; only the model weights (after
INT8 quantization) and any performance tuning would differ.

## Project layout

```text
frontend/     Static CHW dashboard (HTML/CSS/JS), talks to the API via fetch()
backend/      FastAPI app, CNN model, preprocessing, and the WHO rule engine
data/         Mock patient dataset (25 synthetic records)
research/     Methodology summary of the underlying paper
```

## Setup

Requires Python 3.10+.

1. **Create and activate a virtual environment** (from the project root):

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend API:**

   ```bash
   cd backend
   uvicorn inference:app --reload --host 0.0.0.0 --port 8000
   ```

   You should see Uvicorn start on `http://127.0.0.1:8000`. On first run,
   you'll see a log warning that no trained weights were found — this is
   expected. The model serves with randomly-initialized weights so the full
   pipeline can be demoed end-to-end before real training data is available;
   predictions are **not clinically meaningful** until a trained weights
   file is placed at `backend/weights/lesion_classifier.pt`.

4. **Open the frontend:**

   Open `frontend/index.html` directly in a browser (double-click it, or
   use a simple static server). The dashboard status indicator at the top
   right will confirm it can reach the backend at `http://127.0.0.1:8000`.

5. **Try it out:** upload any lesion photo, fill in the respiratory
   symptom checkboxes, and click **Run triage analysis**. Try checking
   "Chest indrawing observed" to see how a WHO danger sign forces a Red
   tier regardless of the image.

## API reference

- `GET /api/health` — returns `{status, model_loaded, version}`.
- `POST /api/triage` — multipart form with:
  - `image`: lesion photo file
  - `symptoms`: JSON string, e.g.
    `{"cough": true, "cough_days": 5, "fever": true, "breathing_difficulty": "severe", "chest_indrawing": false}`

  Returns:

  ```json
  {
    "triage_tier": "Red",
    "confidence": 0.89,
    "reason": "Severe respiratory symptom reported (Chest indrawing) combined with high-risk visual pattern.",
    "quality_gate": { "passed": true, "message": "Image quality acceptable" }
  }
  ```

## Migrating to Raspberry Pi / Jetson Nano when hardware arrives

1. Copy `backend/` and `requirements.txt` onto the device; install
   dependencies (consider `torch` wheels built for ARM, or ONNX Runtime for
   a lighter inference path).
2. Run the model compression pipeline described in
   `research/methodology.md` (INT8 quantization + knowledge distillation)
   and drop the resulting weights at `backend/weights/lesion_classifier.pt`.
3. Point the frontend's `API_BASE` in `frontend/app.js` at the device's
   local address (or serve the frontend from the device itself so the whole
   stack is on one box with no network dependency).
4. Re-run the quality-gate blur threshold calibration
   (`BLUR_VARIANCE_THRESHOLD` in `backend/preprocess.py`) against real
   device-camera photos, since blur sensitivity varies by camera hardware.
5. Begin shadow-mode staging as described in the methodology: log model
   outputs alongside CHW judgment without acting on them, before any
   Yellow/Red output is used to change a real workflow.

## Important caveats

- This is a **hackathon software prototype**, not a validated medical
  device. The CNN ships without trained weights; only the WHO rule engine
  encodes real clinical logic.
- `data/dataset.csv` contains **synthetic, mock records** for demo
  purposes only — no real patient data.
