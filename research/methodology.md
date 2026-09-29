# Methodology Summary

**Source paper:** *Edge AI Diagnostic Assistance for Low-Bandwidth Rural Clinics*

This document summarizes the four core contributions of the paper that this
software prototype demonstrates.

## 1. Offline-first edge deployment on low-cost hardware

The system is designed to run entirely on-device — no cloud round-trip, no
dependence on clinic connectivity. The target deployment hardware is a
Raspberry Pi 4/5 or Jetson Nano, chosen for low cost and low power draw
relative to server-class inference hardware. Every component in this
prototype (image preprocessing, CNN inference, rule evaluation, and the
web UI) is built to run locally, so the only thing that changes when moving
from a laptop to a Pi/Jetson is the compute budget, not the architecture.

## 2. Decoupled multimodal fusion (CNN vision + WHO rule engine)

Rather than training a single end-to-end model to output a triage decision,
the system deliberately decouples two independent signal sources:

- A CNN (MobileNetV3-based) that classifies skin lesion photographs into
  visual categories (Melanoma, Basal Cell Carcinoma, Benign Keratosis) and
  outputs calibrated class probabilities.
- A deterministic rule engine that encodes WHO IMCI/IMAI danger-sign logic
  for respiratory symptoms (e.g. chest indrawing, severe breathing
  difficulty), with no learned parameters.

A late-fusion step combines both signals: any WHO danger sign present forces
an urgent ("Red") triage tier regardless of what the CNN predicts, since
safety rules must never be overridden by model confidence. Absent a danger
sign, the CNN's confidence determines the tier. This separation keeps the
safety-critical logic auditable and independent of model drift, while still
letting the CNN contribute nuanced visual assessment.

## 3. Model compression pipeline (INT8 quantization, knowledge distillation)

To fit inference within the memory and thermal envelope of edge hardware,
the production pipeline (beyond this prototype) applies:

- **Knowledge distillation** from a larger teacher CNN into the lightweight
  MobileNetV3 student used here.
- **INT8 post-training quantization** to shrink model size and speed up
  CPU inference, with accuracy validated against the full-precision model
  on a held-out set before deployment.
- **Structured pruning** of low-magnitude filters where latency budgets
  require it.

This prototype ships the uncompressed model definition so the architecture
is easy to inspect; the compression pipeline is a deployment-time step
applied before shipping weights to physical hardware.

## 4. Human-in-the-loop safety governance and shadow-mode staging

The paper proposes that no triage output reaches clinical use without a
staged rollout:

- **Shadow mode**: the model runs alongside existing clinical workflows
  without influencing decisions, so its outputs can be compared against
  community health worker (CHW) judgment and, where available, physician
  review.
- **Human-in-the-loop review**: Yellow and Red tier outputs are treated as
  triage prioritization signals for a human, not autonomous diagnoses —
  the interface language throughout the prototype ("Refer Urgently",
  "Review Soon", "Routine") reflects this framing deliberately.
- **Auditability**: the WHO rule engine's triggered rules are always
  returned in the response reason string, so any Red classification can be
  traced to a specific, inspectable clinical rule rather than an opaque
  model score.

This prototype implements the full request/response pipeline described
above; it does not include a trained, clinically-validated model — the
classifier ships with either randomly-initialized or placeholder weights
and is intended to demonstrate system architecture and interaction design,
not to produce clinically valid predictions.
