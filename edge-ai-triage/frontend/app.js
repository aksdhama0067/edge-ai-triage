// Prefer the same origin when the frontend is served by FastAPI.
// The localhost fallback still supports opening index.html directly.
const API_BASE = window.location.protocol === "http:" || window.location.protocol === "https:"
  ? window.location.origin
  : "http://127.0.0.1:8000";

const elements = {
  form: document.getElementById("triageForm"),
  imageInput: document.getElementById("lesionImage"),
  uploadPrompt: document.getElementById("uploadPrompt"),
  imagePreview: document.getElementById("imagePreview"),
  cough: document.getElementById("cough"),
  coughDaysRow: document.getElementById("coughDaysRow"),
  coughDays: document.getElementById("coughDays"),
  fever: document.getElementById("fever"),
  breathingDifficulty: document.getElementById("breathingDifficulty"),
  chestIndrawing: document.getElementById("chestIndrawing"),
  runButton: document.getElementById("runButton"),
  resultEmpty: document.getElementById("resultEmpty"),
  resultLoading: document.getElementById("resultLoading"),
  resultCard: document.getElementById("resultCard"),
  resultError: document.getElementById("resultError"),
  errorText: document.getElementById("errorText"),
  tierBadge: document.getElementById("tierBadge"),
  tierLabel: document.getElementById("tierLabel"),
  confidenceValue: document.getElementById("confidenceValue"),
  qualityValue: document.getElementById("qualityValue"),
  reasonText: document.getElementById("reasonText"),
  connDot: document.getElementById("connDot"),
  connLabel: document.getElementById("connLabel"),
};

const TIER_LABELS = {
  Red: "Red — Refer urgently",
  Yellow: "Yellow — Review soon",
  Green: "Green — Routine",
};

function showResultState(state) {
  elements.resultEmpty.hidden = state !== "empty";
  elements.resultLoading.hidden = state !== "loading";
  elements.resultCard.hidden = state !== "result";
  elements.resultError.hidden = state !== "error";
}

function setConnectionStatus(online) {
  elements.connDot.classList.toggle("online", online);
  elements.connDot.classList.toggle("offline", !online);
  elements.connLabel.textContent = online ? "Device online" : "Backend unreachable";
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/api/health`, { method: "GET" });
    setConnectionStatus(response.ok);
  } catch (err) {
    setConnectionStatus(false);
  }
}

elements.imageInput.addEventListener("change", () => {
  const file = elements.imageInput.files && elements.imageInput.files[0];
  if (!file) {
    elements.imagePreview.hidden = true;
    elements.uploadPrompt.hidden = false;
    return;
  }
  const reader = new FileReader();
  reader.onload = (event) => {
    elements.imagePreview.src = event.target.result;
    elements.imagePreview.hidden = false;
    elements.uploadPrompt.hidden = true;
  };
  reader.readAsDataURL(file);
});

elements.cough.addEventListener("change", () => {
  elements.coughDaysRow.hidden = !elements.cough.checked;
  if (!elements.cough.checked) {
    elements.coughDays.value = 0;
  }
});

function tierClass(tier) {
  switch (tier) {
    case "Red":
      return "tier-red";
    case "Yellow":
      return "tier-yellow";
    case "Green":
      return "tier-green";
    default:
      return "";
  }
}

function renderResult(data) {
  elements.tierBadge.className = `tier-badge ${tierClass(data.triage_tier)}`;
  elements.tierLabel.textContent = TIER_LABELS[data.triage_tier] || data.triage_tier;
  elements.confidenceValue.textContent = `${Math.round(data.confidence * 100)}%`;
  elements.qualityValue.textContent = data.quality_gate.passed ? "Passed" : "Flagged";
  elements.reasonText.textContent = data.reason;
  showResultState("result");
}

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = elements.imageInput.files && elements.imageInput.files[0];
  if (!file) {
    return;
  }

  const symptoms = {
    cough: elements.cough.checked,
    cough_days: Number(elements.coughDays.value || 0),
    fever: elements.fever.checked,
    breathing_difficulty: elements.breathingDifficulty.value,
    chest_indrawing: elements.chestIndrawing.checked,
  };

  const formData = new FormData();
  formData.append("image", file);
  formData.append("symptoms", JSON.stringify(symptoms));

  elements.runButton.disabled = true;
  showResultState("loading");

  try {
    const response = await fetch(`${API_BASE}/api/triage`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || `Request failed with status ${response.status}`);
    }

    const data = await response.json();
    renderResult(data);
    setConnectionStatus(true);
  } catch (err) {
    elements.errorText.textContent =
      err.message === "Failed to fetch"
        ? "Could not reach the local triage server. Make sure the backend is running at " +
          `${API_BASE}.`
        : err.message;
    showResultState("error");
    setConnectionStatus(false);
  } finally {
    elements.runButton.disabled = false;
  }
});

showResultState("empty");
checkHealth();
