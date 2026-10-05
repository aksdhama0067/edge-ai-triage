"""
triage.py
---------
All rule-based decision logic lives here, deliberately separated from the
learned vision model (model.py). Every function in this module is a plain,
inspectable rule — there are no learned parameters — so any triage tier can
be traced back to a specific, auditable reason. This is the "Why this
result?" data source for the frontend.

Three inputs are combined:
  1. Image quality (image_quality.py)               -> can force a re-take
  2. CNN lesion classification (model.py)             -> visual pattern signal
  3. WHO IMCI/IMAI respiratory danger signs            -> safety override
  4. Structured lesion questionnaire                   -> transparent escalation

Rule of precedence: safety signals only ever escalate the tier, never
downgrade it. A questionnaire answer can turn Green into Yellow, but nothing
in this module turns Red into anything lower.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from .model import MALIGNANT_CLASSES, RED_CONFIDENCE_THRESHOLD, YELLOW_CONFIDENCE_THRESHOLD, top_prediction
except ImportError:
    from model import MALIGNANT_CLASSES, RED_CONFIDENCE_THRESHOLD, YELLOW_CONFIDENCE_THRESHOLD, top_prediction

TIER_ORDER = {"Green": 0, "Yellow": 1, "Red": 2}


# ---------------------------------------------------------------------------
# WHO IMCI / IMAI respiratory danger-sign rule engine
# ---------------------------------------------------------------------------

def evaluate_danger_signs(symptoms_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explicit, auditable safety rules loosely based on WHO IMCI/IMAI guidance
    for classifying respiratory danger signs in resource-limited settings.

    Returns
    -------
    {"danger_sign_present": bool, "triggered_rules": [str, ...]}
    """
    triggered_rules: List[str] = []

    if symptoms_data.get("chest_indrawing"):
        triggered_rules.append("WHO IMCI: chest indrawing present (severe pneumonia indicator)")

    if symptoms_data.get("breathing_difficulty") == "severe":
        triggered_rules.append("WHO IMCI: severe breathing difficulty reported")

    if symptoms_data.get("fever") and symptoms_data.get("cough_days", 0) >= 14:
        triggered_rules.append("WHO IMCI: fever with cough persisting >= 14 days (possible TB referral)")

    if symptoms_data.get("fever") and symptoms_data.get("breathing_difficulty") in {"mild", "severe"}:
        triggered_rules.append("WHO IMCI: fever combined with breathing difficulty")

    return {
        "danger_sign_present": len(triggered_rules) > 0,
        "triggered_rules": triggered_rules,
    }


# ---------------------------------------------------------------------------
# Structured lesion questionnaire (dermatology follow-up)
# ---------------------------------------------------------------------------

def evaluate_lesion_questionnaire(questionnaire: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic, transparent flags from patient/CHW-reported lesion
    history. These NEVER produce a fabricated probability — they are plain
    boolean flags with a plain-language reason each, used only to decide
    whether escalation to a closer look is warranted.
    """
    flags: List[str] = []

    if questionnaire.get("bleeding"):
        flags.append("Reported bleeding from the lesion")

    if questionnaire.get("recent_change") and questionnaire.get("size_change"):
        flags.append("Reported recent change and size change")
    elif questionnaire.get("recent_change"):
        flags.append("Reported recent change in appearance")
    elif questionnaire.get("size_change"):
        flags.append("Reported recent change in size")

    if questionnaire.get("painful") and questionnaire.get("itchy"):
        flags.append("Reported both pain and itching at the site")

    duration_days = questionnaire.get("duration_days", 0) or 0
    if duration_days and duration_days < 21 and (questionnaire.get("recent_change") or questionnaire.get("size_change")):
        flags.append("Change reported within a relatively short time frame (under 3 weeks)")

    return {
        "concern_flags": flags,
        "concern_count": len(flags),
    }


# ---------------------------------------------------------------------------
# Late-fusion triage module
# ---------------------------------------------------------------------------

def fuse_triage_decision(
    lesion_probabilities: Dict[str, float],
    danger_sign_result: Dict[str, Any],
    questionnaire_result: Dict[str, Any] | None = None,
    quality_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Combine every available signal into one triage tier, in order of
    precedence:

      1. Poor image quality -> "Yellow" (cannot assess reliably; retake photo).
         This never gets escalated to Red purely from bad image quality —
         "we can't tell" is not the same claim as "this looks concerning".
      2. Any WHO danger sign -> forced "Red", regardless of the CNN output.
      3. Otherwise, CNN malignant-class confidence sets the base tier
         (Red / Yellow / Green thresholds).
      4. Lesion-questionnaire concern flags can escalate Green -> Yellow,
         never downgrade, and never fabricate a numeric probability from
         questionnaire answers alone.

    Returns a dict with "triage_tier", "confidence", "reason", and the full
    ordered list of contributing "factors" for the "Why this result?" UI.
    """
    questionnaire_result = questionnaire_result or {"concern_flags": [], "concern_count": 0}
    factors: List[str] = []

    top_class, top_confidence = top_prediction(lesion_probabilities)
    malignant_confidence = sum(p for c, p in lesion_probabilities.items() if c in MALIGNANT_CLASSES)

    factors.append(
        f"Image quality: {quality_result['quality'].capitalize()} (score {quality_result['score']}/100)"
        if quality_result else "Image quality: not assessed"
    )
    factors.append(
        f"Model result: {top_class} pattern, {top_confidence:.0%} confidence "
        f"(model has NOT been clinically validated)"
    )

    if quality_result and quality_result.get("quality") == "poor":
        factors.append("Image quality gate: failed — recommendation is to retake the photo")
        return {
            "triage_tier": "Yellow",
            "confidence": 0.0,
            "reason": "Image quality was too low for a reliable visual assessment. "
                      f"{quality_result.get('recommendation', '')}",
            "factors": factors,
        }

    if danger_sign_result["danger_sign_present"]:
        factors.extend(danger_sign_result["triggered_rules"])
        rules_text = "; ".join(danger_sign_result["triggered_rules"])
        return {
            "triage_tier": "Red",
            "confidence": round(top_confidence, 4),
            "reason": f"Reported symptom(s) matched WHO danger-sign criteria: {rules_text}.",
            "factors": factors,
        }

    if malignant_confidence >= RED_CONFIDENCE_THRESHOLD:
        tier = "Red"
        reason = (
            f"The model classified this image as belonging to a category associated "
            f"with higher review priority ({top_class}, {top_confidence:.0%} confidence). "
            "No respiratory danger signs were reported."
        )
    elif malignant_confidence >= YELLOW_CONFIDENCE_THRESHOLD:
        tier = "Yellow"
        reason = f"Moderate-confidence visual pattern ({top_class}, {top_confidence:.0%}). Consider review soon."
    else:
        tier = "Green"
        reason = f"Low-priority visual pattern ({top_class}, {top_confidence:.0%}) and no danger signs reported."

    # Questionnaire can only escalate, and only out of the lowest tier.
    if tier == "Green" and questionnaire_result["concern_flags"]:
        factors.extend(questionnaire_result["concern_flags"])
        tier = "Yellow"
        reason = (
            "Visual pattern alone was low-priority, but reported history raised the "
            "priority level: " + "; ".join(questionnaire_result["concern_flags"]) + "."
        )
    elif questionnaire_result["concern_flags"]:
        factors.extend(questionnaire_result["concern_flags"])

    return {
        "triage_tier": tier,
        "confidence": round(top_confidence, 4),
        "reason": reason,
        "factors": factors,
    }
