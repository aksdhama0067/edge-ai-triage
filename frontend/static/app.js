/*
 * app.js
 * ------
 * Application logic for the Edge AI Triage frontend. Talks to the FastAPI
 * backend at the same origin (the backend serves this file, so relative
 * paths work whether accessed via 127.0.0.1, localhost, or a LAN IP).
 */

const API_BASE = window.location.origin;

// ---------------------------------------------------------------------------
// Image state
// ---------------------------------------------------------------------------
// There is exactly ONE normalized representation of "the current image",
// regardless of whether it came from the camera, the file picker, a drag-
// drop, or a paste: a Blob. Both intake paths funnel into setCurrentImage().
//
//   originalImageBlob  - the untouched source Blob/File, exactly as captured
//                         or selected. Data Saver compression is always
//                         derived FROM this, never from a previously
//                         compressed result, so toggling Data Saver on/off
//                         repeatedly never progressively re-compresses.
//   currentImageBlob    - what actually gets uploaded: either
//                         originalImageBlob itself (Data Saver off) or a
//                         freshly compressed derivative of it (Data Saver
//                         on).
//   currentPreviewUrl   - the single object URL currently backing
//                         #imagePreview's src, tracked so it can be revoked
//                         exactly once, exactly when it's safe to do so.
let originalImageBlob = null;
let currentImageBlob = null;
let currentImageOriginalBytes = 0;
let currentPreviewUrl = null;
let lastTriageContext = null; // small summary sent to the assistant for context

const el = (id) => document.getElementById(id);

const ids = [
  "lesionImage", "uploadPrompt", "imagePreview", "cameraButton", "existingPhotoButton",
  "removePhotoButton", "uploadError",
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
  if (e.key !== "Escape") return;
  // There is no separate "photo picker" modal in this app (the native file
  // input is opened directly) - only these two overlays actually exist.
  if (!E.privacyOverlay.hidden) E.privacyOverlay.hidden = true;
  if (!E.cameraOverlay.hidden) closeCamera();
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

// The backend validates uploaded images by real content-type and decoded
// bytes, never by filename - so this is purely cosmetic/debuggable, not a
// functional requirement. Still worth getting right rather than always
// claiming ".jpg" for a PNG/WebP blob.
function imageFilename(blob) {
  const ext = { "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif" }[blob.type] || "jpg";
  return `lesion.${ext}`;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function revokeCurrentPreviewUrl() {
  if (currentPreviewUrl) {
    URL.revokeObjectURL(currentPreviewUrl);
    currentPreviewUrl = null;
  }
}

function setPreview(blob) {
  // Create and attach the new URL BEFORE revoking the old one, so the <img>
  // is never pointed at a dead URL even for an instant, then revoke the
  // previous URL now that nothing references it any more.
  const newUrl = URL.createObjectURL(blob);
  E.imagePreview.src = newUrl;
  E.imagePreview.hidden = false;
  E.uploadPrompt.hidden = true;
  E.removePhotoButton.hidden = false;
  revokeCurrentPreviewUrl();
  currentPreviewUrl = newUrl;
}

// Data Saver always compresses FROM the untouched original, never from a
// previously-compressed result - so toggling it on/off/on never stacks
// compression passes or loses track of the true original size.
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
  if (bitmap.close) bitmap.close();

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

// The single entry point every intake path (camera, file picker, drag-drop,
// paste) funnels through. `sourceBlob` must always be the TRUE original -
// callers never pass a previously-compressed blob back in here.
async function setCurrentImage(sourceBlob) {
  if (!sourceBlob) return false;
  originalImageBlob = sourceBlob;
  try {
    currentImageOriginalBytes = sourceBlob.size;
    const processed = await compressIfNeeded(sourceBlob);
    currentImageBlob = processed;
    setPreview(processed);
    return true;
  } catch (err) {
    console.error("Unable to process image (compression step):", err);
    // Compression is an optional optimization - if it fails for any reason,
    // the original file is still perfectly usable. Fall back to it rather
    // than blocking the user.
    currentImageBlob = sourceBlob;
    setPreview(sourceBlob);
    E.sizeReadout.hidden = true;
    return true;
  }
}

function showUploadError(message) {
  E.uploadError.textContent = message;
  E.uploadError.hidden = false;
}

function clearUploadError() {
  E.uploadError.hidden = true;
  E.uploadError.textContent = "";
}

function describeFileType(file) {
  if (file.type) return file.type;
  const name = (file.name || "").toLowerCase();
  const ext = name.includes(".") && name.split(".").pop();
  return ext ? `.${ext} file with no reported type` : "unknown file type";
}

// Clears only the image: not the questionnaire, not prior results, not the
// Privacy modal. Available from both the storage flow and the camera flow.
function clearSelectedImage() {
  revokeCurrentPreviewUrl();
  originalImageBlob = null;
  currentImageBlob = null;
  currentImageOriginalBytes = 0;
  E.lesionImage.value = "";
  E.imagePreview.hidden = true;
  E.imagePreview.src = "";
  E.uploadPrompt.hidden = false;
  E.removePhotoButton.hidden = true;
  E.sizeReadout.hidden = true;
  E.uploadStatus.textContent = t("uploadStatus");
  clearUploadError();
}

E.removePhotoButton.addEventListener("click", clearSelectedImage);

async function useExistingImage(file, source = "file") {
  if (!file) return;
  clearUploadError();

  // Authoritative validation: actually attempt to decode the file as an
  // image rather than trusting file.type (which Windows in particular can
  // report as empty or generic for otherwise-valid photos) or the file
  // extension (which a renamed non-image file could fake). A successful
  // decode is the real proof this is a usable image.
  let probeBitmap;
  try {
    probeBitmap = await createImageBitmap(file);
  } catch (err) {
    const claimedImage = (file.type && file.type.startsWith("image/")) ||
      /\.(jpe?g|png|webp|gif|bmp)$/i.test(file.name || "");
    showUploadError(
      claimedImage
        ? t("corruptImageFile")
        : `${t("invalidImageFile")} (${describeFileType(file)})`
    );
    return;
  }
  if (probeBitmap.close) probeBitmap.close();

  const ok = await setCurrentImage(file);
  if (ok) {
    E.uploadStatus.textContent = source === "paste" ? t("pastedPhoto") : t("photoSelected");
  }
}

E.lesionImage.addEventListener("change", async () => {
  const file = E.lesionImage.files && E.lesionImage.files[0];
  if (!file) return;
  await useExistingImage(file);
});

// Reset the input's value right before the native picker can possibly open,
// regardless of whether that happens via the <label> itself being clicked
// or via the "Choose existing photo" button calling .click() below. Without
// this, re-selecting the exact same file does not reliably re-fire `change`
// in Chrome/Edge/Firefox, which looks exactly like the picker "doing
// nothing" on a second attempt.
E.lesionImage.addEventListener("click", () => {
  E.lesionImage.value = "";
});

function openPhotoPicker() {
  // Use the native file picker directly. This avoids a second modal layer and
  // makes the same path work for existing photos, screenshots, and saved captures.
  E.lesionImage.click();
}

E.existingPhotoButton.addEventListener("click", openPhotoPicker);

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
  // Re-derive from the untouched original every time, never from
  // currentImageBlob (which may already be a compressed derivative) - this
  // is what prevents ON -> OFF -> ON from stacking compression passes.
  if (originalImageBlob) {
    await setCurrentImage(originalImageBlob);
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
    showCameraError();
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    E.cameraVideo.srcObject = cameraStream;
    await E.cameraVideo.play();
  } catch (err) {
    console.warn("Camera unavailable:", err);
    showCameraError();
  }
}

function showCameraError() {
  E.cameraError.textContent = `${t("cameraUnavailable")} You can choose an existing photo instead.`;
  E.cameraError.hidden = false;
  E.cameraCaptureBtn.hidden = true;
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
  openPhotoPicker();
});

E.cameraCaptureBtn.addEventListener("click", () => {
  const video = E.cameraVideo;
  const canvas = E.cameraCanvas;

  if (!video.videoWidth || !video.videoHeight) {
    E.cameraError.textContent = "The camera is not ready yet. Wait a moment and try again, or choose an existing photo.";
    E.cameraError.hidden = false;
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

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
    if (!blob) {
      E.cameraError.textContent = "Could not create the captured image. Please retake it or choose an existing photo.";
      E.cameraError.hidden = false;
      return;
    }
    // A canvas-generated JPEG from a live video frame is guaranteed
    // decodable, so it goes straight into the same normalized intake point
    // storage images use (setCurrentImage) - one representation, two sources.
    clearUploadError();
    await setCurrentImage(blob);
    E.uploadStatus.textContent = t("photoSelected");
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
    formData.append("image", currentImageBlob, imageFilename(currentImageBlob));
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
    similarityFormData.append("image", currentImageBlob, imageFilename(currentImageBlob));
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
    formData.append("image", currentImageBlob, imageFilename(currentImageBlob));
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
  // Image-specific cleanup (revoke URL, clear state, reset input, hide
  // preview/remove button) is identical to the targeted "Remove image"
  // action, so it's defined once and reused here rather than duplicated.
  clearSelectedImage();
  lastTriageContext = null;
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
