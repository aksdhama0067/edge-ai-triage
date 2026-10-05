/*
 * app.js
 * ------
 * Application logic for the Edge AI Triage frontend. Talks to the FastAPI
 * backend at the same origin (the backend serves this file, so relative
 * paths work whether accessed via 127.0.0.1, localhost, or a LAN IP).
 */

const API_BASE = window.location.origin;

// Holds the current working image as a Blob, independent of whether it came
// from the file input or the camera, plus the extension to send it as.
let currentImageBlob = null;
let currentImageOriginalBytes = 0;
let lastTriageContext = null; // small summary sent to the assistant for context

const el = (id) => document.getElementById(id);

const ids = [
  "lesionImage", "uploadPrompt", "imagePreview", "cameraButton", "existingPhotoButton",
  "uploadDrop", "uploadStatus", "dataSaverToggle", "sizeReadout",
  "cough", "coughDaysRow", "coughDays", "fever", "breathingDifficulty", "chestIndrawing",
  "recentChange", "sizeChange", "appearanceChange", "painful", "itchy", "bleeding", "durationDays",
  "triageForm", "runButton",
  "progressPanel", "progressSteps",
  "resultsDashboard",
  "qualityBadge", "qualityRecommendation",
  "analysisCategory", "analysisConfidence", "analysisDisclaimer", "explainButton", "explainResult",
  "explainOriginal", "explainHeatmap",
  "similarityMessage", "similarityGrid",
  "tierBadge", "triageReadText", "triageFactors",
  "limitationsList",
  "assistantChips", "assistantForm", "assistantInput", "assistantAnswer", "assistantAnswerText", "assistantProviderNote",
  "privacyButton", "privacyOverlay", "closePrivacy", "resetButton",
  "cameraOverlay", "cameraVideo", "cameraCanvas", "cameraCaptured", "cameraError",
  "cameraCaptureBtn", "cameraRetakeBtn", "cameraUseBtn", "cameraFallbackBtn", "cameraCancelBtn",
  "connDot", "connLabel", "langToggle",
];
const E = {};
ids.forEach((id) => (E[id] = el(id)));

// ---------------------------------------------------------------------------
// Language + privacy
// ---------------------------------------------------------------------------

E.langToggle.addEventListener("click", () => {
  applyLanguage(window.currentLanguage === "en" ? "hi" : "en");
});

window.onLanguageChanged = () => {
  renderAssistantChips();
};

E.privacyButton.addEventListener("click", () => {
  E.privacyOverlay.hidden = false;
  E.closePrivacy.focus();
});
E.closePrivacy.addEventListener("click", () => (E.privacyOverlay.hidden = true));
E.privacyOverlay.addEventListener("click", (e) => {
  if (e.target === E.privacyOverlay) E.privacyOverlay.hidden = true;
});
E.resetButton.addEventListener("click", () => {
  resetAnalysis();
  E.privacyOverlay.hidden = true;
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    E.privacyOverlay.hidden = true;
    if (!E.cameraOverlay.hidden) closeCamera();
  }
});

// ---------------------------------------------------------------------------
// Connection status
// ---------------------------------------------------------------------------

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    const ok = response.ok;
    E.connDot.classList.toggle("online", ok);
    E.connDot.classList.toggle("offline", !ok);
    E.connLabel.textContent = t(ok ? "deviceOnline" : "backendUnreachable");
  } catch {
    E.connDot.classList.add("offline");
    E.connLabel.textContent = t("backendUnreachable");
  }
}

// ---------------------------------------------------------------------------
// Image intake: file upload, preview, data-saver compression
// ---------------------------------------------------------------------------

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function setPreview(blob) {
  const url = URL.createObjectURL(blob);
  E.imagePreview.src = url;
  E.imagePreview.hidden = false;
  E.uploadPrompt.hidden = true;
}

async function compressIfNeeded(blob) {
  if (!E.dataSaverToggle.checked) {
    E.sizeReadout.hidden = true;
    return blob;
  }
  const originalSize = blob.size;
  const bitmap = await createImageBitmap(blob);
  const maxDim = 900;
  const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);

  const compressedBlob = await new Promise((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", 0.72)
  );

  const finalBlob = compressedBlob && compressedBlob.size < originalSize ? compressedBlob : blob;
  E.sizeReadout.hidden = false;
  E.sizeReadout.textContent = t("sizeReadoutTemplate")
    .replace("{orig}", formatBytes(originalSize))
    .replace("{compressed}", formatBytes(finalBlob.size));
  return finalBlob;
}

async function setCurrentImage(blob) {
  currentImageOriginalBytes = blob.size;
  const processed = await compressIfNeeded(blob);
  currentImageBlob = processed;
  setPreview(processed);
}

async function useExistingImage(file, source = "file") {
  if (!file || !file.type || !file.type.startsWith("image/")) {
    alert(t("invalidImageFile"));
    return;
  }
  await setCurrentImage(file);
  E.uploadStatus.textContent = source === "paste" ? t("pastedPhoto") : t("photoSelected");
}

E.lesionImage.addEventListener("change", async () => {
  const file = E.lesionImage.files && E.lesionImage.files[0];
  if (!file) return;
  await useExistingImage(file);
});

E.existingPhotoButton.addEventListener("click", () => E.lesionImage.click());

// Drag-and-drop fallback for desktop browsers.
["dragenter", "dragover"].forEach((eventName) => {
  E.uploadDrop.addEventListener(eventName, (event) => {
    event.preventDefault();
    E.uploadDrop.classList.add("drag-active");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  E.uploadDrop.addEventListener(eventName, (event) => {
    event.preventDefault();
    E.uploadDrop.classList.remove("drag-active");
  });
});
E.uploadDrop.addEventListener("drop", async (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) await useExistingImage(file, "drop");
});

// Paste a copied screenshot/photo directly into the app.
document.addEventListener("paste", async (event) => {
  const items = Array.from(event.clipboardData?.items || []);
  const imageItem = items.find((item) => item.type.startsWith("image/"));
  if (!imageItem) return;
  const file = imageItem.getAsFile();
  if (file) await useExistingImage(file, "paste");
});

E.dataSaverToggle.addEventListener("change", async () => {
  if (currentImageBlob) {
    await setCurrentImage(currentImageBlob);
  }
});

E.cough.addEventListener("change", () => {
  E.coughDaysRow.hidden = !E.cough.checked;
  if (!E.cough.checked) E.coughDays.value = 0;
});

// ---------------------------------------------------------------------------
// Camera capture
// ---------------------------------------------------------------------------

let cameraStream = null;

async function openCamera() {
  E.cameraOverlay.hidden = false;
  E.cameraError.hidden = true;
  E.cameraCaptured.hidden = true;
  E.cameraVideo.hidden = false;
  E.cameraCaptureBtn.hidden = false;
  E.cameraRetakeBtn.hidden = true;
  E.cameraUseBtn.hidden = true;

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    E.cameraError.textContent = t("cameraUnavailable");
    E.cameraError.hidden = false;
    E.cameraCaptureBtn.hidden = true;
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
    });
    E.cameraVideo.srcObject = cameraStream;
  } catch (err) {
    E.cameraError.textContent = t("cameraUnavailable");
    E.cameraError.hidden = false;
    E.cameraCaptureBtn.hidden = true;
  }
}

function stopCameraStream() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
}

function closeCamera() {
  stopCameraStream();
  E.cameraOverlay.hidden = true;
}

E.cameraButton.addEventListener("click", openCamera);
E.cameraCancelBtn.addEventListener("click", closeCamera);
E.cameraFallbackBtn.addEventListener("click", () => {
  closeCamera();
  E.lesionImage.click();
});

E.cameraCaptureBtn.addEventListener("click", () => {
  const video = E.cameraVideo;
  const canvas = E.cameraCanvas;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  E.cameraCaptured.src = canvas.toDataURL("image/jpeg", 0.9);
  E.cameraCaptured.hidden = false;
  E.cameraVideo.hidden = true;
  E.cameraCaptureBtn.hidden = true;
  E.cameraRetakeBtn.hidden = false;
  E.cameraUseBtn.hidden = false;
});

E.cameraRetakeBtn.addEventListener("click", () => {
  E.cameraCaptured.hidden = true;
  E.cameraVideo.hidden = false;
  E.cameraCaptureBtn.hidden = false;
  E.cameraRetakeBtn.hidden = true;
  E.cameraUseBtn.hidden = true;
});

E.cameraUseBtn.addEventListener("click", () => {
  E.cameraCanvas.toBlob(async (blob) => {
    if (blob) await setCurrentImage(blob);
    closeCamera();
  }, "image/jpeg", 0.9);
});

// ---------------------------------------------------------------------------
// Progress steps
// ---------------------------------------------------------------------------

function setStep(stepName, state) {
  const li = E.progressSteps.querySelector(`[data-step="${stepName}"]`);
  if (!li) return;
  li.classList.remove("active", "done");
  if (state) li.classList.add(state);
}

function resetSteps() {
  E.progressSteps.querySelectorAll("li").forEach((li) => li.classList.remove("active", "done"));
}

// ---------------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------------

function qualityKey(q) {
  return { good: "qualityGood", warning: "qualityWarning", poor: "qualityPoor" }[q] || q;
}

function renderQuality(result) {
  E.qualityBadge.textContent = t(qualityKey(result.quality));
  E.qualityBadge.className = `badge badge-${result.quality}`;
  E.qualityRecommendation.textContent = result.recommendation;
}

function renderAnalysis(modelAnalysis) {
  if (modelAnalysis.available) {
    E.analysisCategory.textContent = `${modelAnalysis.top_class} (${modelAnalysis.model_mode})`;
    E.analysisConfidence.textContent = `${Math.round(modelAnalysis.top_confidence * 100)}%`;
  } else {
    E.analysisCategory.textContent = modelAnalysis.model_mode === "unavailable" ? "AI model unavailable" : "—";
    E.analysisConfidence.textContent = "—";
  }
  E.analysisDisclaimer.textContent = modelAnalysis.disclaimer;
}

function tierClass(tier) {
  return { Red: "badge-poor", Yellow: "badge-warning", Green: "badge-good" }[tier] || "";
}

function tierKey(tier) {
  return { Red: "tierRed", Yellow: "tierYellow", Green: "tierGreen" }[tier] || tier;
}

function renderTriage(result) {
  E.tierBadge.textContent = t(tierKey(result.triage_tier));
  E.tierBadge.className = `badge ${tierClass(result.triage_tier)}`;
  E.triageReadText.textContent = result.reason;
  E.triageFactors.innerHTML = "";
  result.factors.forEach((factor) => {
    const li = document.createElement("li");
    li.textContent = factor;
    E.triageFactors.appendChild(li);
  });
}

function renderLimitations() {
  E.limitationsList.innerHTML = "";
  ["limitationModel", "limitationDataset", "limitationNoClinical", "limitationSecondOpinion"].forEach((key) => {
    const li = document.createElement("li");
    li.textContent = t(key);
    E.limitationsList.appendChild(li);
  });
}

function renderSimilarity(result) {
  E.similarityGrid.innerHTML = "";
  if (!result.available) {
    E.similarityMessage.textContent = result.message || t("similarityUnavailable");
    return;
  }
  E.similarityMessage.textContent = "";
  result.matches.forEach((match) => {
    const item = document.createElement("div");
    item.className = "similarity-item";
    if (match.thumbnail_base64) {
      const img = document.createElement("img");
      img.src = `data:image/png;base64,${match.thumbnail_base64}`;
      img.alt = match.label || "similar example";
      item.appendChild(img);
    }
    const caption = document.createElement("span");
    caption.textContent = `${match.label || "?"} · ${Math.round(match.similarity_score * 100)}%`;
    item.appendChild(caption);
    E.similarityGrid.appendChild(item);
  });
}

// ---------------------------------------------------------------------------
// Form submission -> orchestrate the pipeline
// ---------------------------------------------------------------------------

function collectSymptoms() {
  return {
    cough: E.cough.checked,
    cough_days: Number(E.coughDays.value || 0),
    fever: E.fever.checked,
    breathing_difficulty: E.breathingDifficulty.value,
    chest_indrawing: E.chestIndrawing.checked,
  };
}

function collectQuestionnaire() {
  return {
    recent_change: E.recentChange.checked,
    size_change: E.sizeChange.checked,
    appearance_change: E.appearanceChange.checked,
    painful: E.painful.checked,
    itchy: E.itchy.checked,
    bleeding: E.bleeding.checked,
    duration_days: Number(E.durationDays.value || 0),
  };
}

E.triageForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!currentImageBlob) {
    alert(t("pleaseAddPhoto"));
    return;
  }

  E.runButton.disabled = true;
  E.progressPanel.hidden = false;
  E.resultsDashboard.hidden = true;
  resetSteps();
  setStep("quality", "active");

  try {
    const symptoms = collectSymptoms();
    const questionnaire = collectQuestionnaire();

    const formData = new FormData();
    formData.append("image", currentImageBlob, "lesion.jpg");
    formData.append("symptoms", JSON.stringify(symptoms));
    formData.append("questionnaire", JSON.stringify(questionnaire));

    setStep("quality", "done");
    setStep("analyze", "active");

    const triageResponse = await fetch(`${API_BASE}/api/triage`, { method: "POST", body: formData });
    if (!triageResponse.ok) {
      const body = await triageResponse.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${triageResponse.status})`);
    }
    const triageResult = await triageResponse.json();

    setStep("analyze", "done");
    setStep("explain", "active");

    // Similarity is cheap enough to fetch eagerly; Grad-CAM is fetched lazily
    // on demand (see explainButton) since it runs an extra backward pass.
    const similarityFormData = new FormData();
    similarityFormData.append("image", currentImageBlob, "lesion.jpg");
    const similarityResponse = await fetch(`${API_BASE}/api/similarity`, {
      method: "POST",
      body: similarityFormData,
    }).catch(() => null);
    const similarityResult = similarityResponse && similarityResponse.ok
      ? await similarityResponse.json()
      : { available: false, message: t("similarityUnavailable"), matches: [] };

    setStep("explain", "done");
    setStep("prepare", "active");

    renderQuality(triageResult.quality_gate);
    renderAnalysis(triageResult.model_analysis);
    renderTriage(triageResult);
    renderLimitations();
    renderSimilarity(similarityResult);
    E.explainResult.hidden = true;
    E.explainButton.textContent = t("showAttention");

    lastTriageContext = {
      triage_tier: triageResult.triage_tier,
      model_mode: triageResult.model_analysis.model_mode,
      top_class: triageResult.model_analysis.top_class,
      image_quality: triageResult.quality_gate.quality,
    };

    setStep("prepare", "done");
    E.resultsDashboard.hidden = false;
    E.resultsDashboard.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  } catch (err) {
    alert(err.message || "Something went wrong. Is the backend running?");
  } finally {
    E.runButton.disabled = false;
    E.progressPanel.hidden = true;
  }
});

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

// ---------------------------------------------------------------------------
// Explainability (lazy)
// ---------------------------------------------------------------------------

E.explainButton.addEventListener("click", async () => {
  if (!E.explainResult.hidden) {
    E.explainResult.hidden = true;
    E.explainButton.textContent = t("showAttention");
    return;
  }
  if (!currentImageBlob) return;

  E.explainButton.disabled = true;
  try {
    const formData = new FormData();
    formData.append("image", currentImageBlob, "lesion.jpg");
    const response = await fetch(`${API_BASE}/api/explain`, { method: "POST", body: formData });
    const result = await response.json();

    if (!result.available) {
      alert(result.message || "Explainability is not available for this image.");
      return;
    }

    E.explainOriginal.src = E.imagePreview.src;
    E.explainHeatmap.src = `data:image/png;base64,${result.heatmap_overlay_base64}`;
    E.explainResult.hidden = false;
    E.explainButton.textContent = t("hideAttention");
  } catch {
    alert("Could not generate the attention map.");
  } finally {
    E.explainButton.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// AI assistant
// ---------------------------------------------------------------------------

const CHIP_QUESTIONS = {
  en: [
    "What does the attention heatmap mean?",
    "What do the triage colors mean?",
    "What are this tool's limitations?",
    "Should I see a doctor?",
  ],
  hi: [
    "ध्यान क्षेत्र (attention heatmap) का क्या मतलब है?",
    "ट्राएज रंगों का क्या मतलब है?",
    "इस उपकरण की सीमाएँ क्या हैं?",
    "क्या मुझे डॉक्टर को दिखाना चाहिए?",
  ],
};

function renderAssistantChips() {
  E.assistantChips.innerHTML = "";
  const questions = CHIP_QUESTIONS[window.currentLanguage] || CHIP_QUESTIONS.en;
  questions.forEach((question) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip-button";
    chip.textContent = question;
    chip.addEventListener("click", () => askAssistant(question));
    E.assistantChips.appendChild(chip);
  });
}

async function askAssistant(question) {
  E.assistantInput.value = "";
  E.assistantAnswer.hidden = false;
  E.assistantAnswerText.textContent = "…";
  E.assistantProviderNote.textContent = "";

  try {
    const response = await fetch(`${API_BASE}/api/assistant`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, context: lastTriageContext, language: window.currentLanguage }),
    });
    const result = await response.json();
    E.assistantAnswerText.textContent = result.answer;
    E.assistantProviderNote.textContent = t(result.provider === "openai" ? "externalAssistant" : "localAssistant");
  } catch {
    E.assistantAnswerText.textContent = "Could not reach the assistant.";
  }
}

E.assistantForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = E.assistantInput.value.trim();
  if (question) askAssistant(question);
});

// ---------------------------------------------------------------------------
// Text to speech ("Read aloud")
// ---------------------------------------------------------------------------

document.querySelectorAll(".read-aloud").forEach((button) => {
  button.addEventListener("click", () => {
    const targetId = button.getAttribute("data-target");
    const target = document.getElementById(targetId);
    if (!target || !target.textContent.trim()) return;
    if (!("speechSynthesis" in window)) {
      alert("Text-to-speech is not supported in this browser.");
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(target.textContent);
    utterance.lang = window.currentLanguage === "hi" ? "hi-IN" : "en-US";
    window.speechSynthesis.speak(utterance);
  });
});

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

function resetAnalysis() {
  currentImageBlob = null;
  currentImageOriginalBytes = 0;
  lastTriageContext = null;
  E.lesionImage.value = "";
  E.imagePreview.hidden = true;
  E.imagePreview.src = "";
  E.uploadPrompt.hidden = false;
  E.uploadStatus.textContent = t("uploadStatus");
  E.sizeReadout.hidden = true;
  E.resultsDashboard.hidden = true;
  E.explainResult.hidden = true;
  E.assistantAnswer.hidden = true;
  E.triageForm.reset();
  E.coughDaysRow.hidden = true;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

applyLanguage("en");
renderAssistantChips();
checkHealth();
