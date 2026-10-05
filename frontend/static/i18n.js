/*
 * i18n.js
 * -------
 * Translation dictionaries, kept separate from application logic so UI text
 * lives in one place rather than being scattered through app.js.
 *
 * Only interface chrome is translated. Medical/clinical terms (lesion
 * category names, WHO rule text, model disclaimers returned by the API)
 * are deliberately left in English rather than machine-translated, since
 * mistranslated medical terminology is worse than none. Hindi strings below
 * were written by hand for this UI, not run through a translation API.
 */

window.I18N = {
  en: {
    tagline: "AI-assisted health research tool — not a medical diagnosis",
    checkingDevice: "Checking device…",
    deviceOnline: "Device online",
    backendUnreachable: "Backend unreachable",
    privacy: "Privacy",
    disclaimerStrong: "Educational / research prototype.",
    disclaimerBody: "This tool does not diagnose medical conditions. Always consider professional evaluation for health concerns.",
    newAssessment: "New assessment",
    lesionPhoto: "Lesion photo",
    uploadPrompt: "Tap to add a photo of the skin lesion",
    uploadHint: "Hold steady, fill the frame, use natural light",
    takePhoto: "Take a photo",
    dataSaver: "Data Saver Mode",
    sizeReadoutTemplate: "Original: {orig} — Compressed: {compressed}",
    respiratorySymptoms: "Respiratory symptoms (optional)",
    cough: "Cough",
    coughDays: "How many days has the cough lasted?",
    fever: "Fever",
    breathingDifficulty: "Breathing difficulty",
    none: "None",
    mild: "Mild",
    severe: "Severe",
    chestIndrawing: "Chest indrawing observed",
    lesionQuestionnaire: "About this lesion",
    recentChange: "Has it changed recently?",
    sizeChange: "Has its size changed?",
    appearanceChange: "Has its appearance changed?",
    painful: "Is it painful?",
    itchy: "Is it itchy?",
    bleeding: "Is there bleeding?",
    durationDays: "How many days has it been present?",
    runAnalysis: "Run triage analysis",
    pleaseAddPhoto: "Please add or take a photo first.",
    stepQuality: "Checking image quality…",
    stepAnalyze: "Analyzing…",
    stepExplain: "Generating explanation…",
    stepPrepare: "Preparing result…",
    cardImageQuality: "Image quality",
    cardAiAnalysis: "AI analysis",
    category: "Category",
    confidence: "Confidence",
    showAttention: "Show AI attention",
    hideAttention: "Hide AI attention",
    original: "Original",
    aiAttention: "AI attention",
    gradcamDisclaimer: "The highlighted region shows areas that contributed to the model's prediction. This visualization does not prove that the highlighted region is medically abnormal.",
    cardSimilar: "Similar examples",
    similarityDisclaimer: "Visual similarity does not indicate that two lesions have the same diagnosis.",
    similarityUnavailable: "Similarity search is not configured yet.",
    cardTriage: "Triage",
    whyThisResult: "Why this result?",
    cardLimitations: "Limitations",
    askAssistant: "Ask about this result",
    assistantInputLabel: "Ask a question about this tool",
    assistantPlaceholder: "Ask a question about this tool…",
    send: "Send",
    learnHeading: "Learn",
    learnQ1: "What are skin lesions?",
    learnA1: "A lesion is any area of skin that looks different from the skin around it — a mole, spot, growth, or patch. Most lesions are harmless, but changes over time are worth having looked at.",
    learnQ2: "Benign vs. suspicious categories",
    learnA2: "\"Benign\" broadly means not cancerous. \"Suspicious\" categories are patterns that are sometimes associated with skin cancers and usually warrant a closer professional look — they are not themselves a diagnosis.",
    learnQ3: "What can this AI actually do?",
    learnA3: "It can check whether a photo is usable, sort a photo into broad visual categories, show which pixels influenced that sort, and look for visually similar reference examples. It cannot examine skin the way a clinician can, and it has not been validated for clinical use.",
    learnQ4: "What can't it do?",
    learnA4: "It cannot diagnose, cannot rule anything out, cannot feel a lesion's texture, and — unless explicitly stated as \"trained\" and even then, \"clinically validated\" — its predictions are illustrative rather than medically meaningful.",
    learnQ5: "How to take a better photo",
    learnA5: "Use natural daylight if possible, hold the camera steady, fill the frame with the area of interest, and avoid direct flash glare.",
    footerNote: "Prototype for research/educational demonstration purposes. Not a certified medical device.",
    privacyTitle: "Privacy",
    privacyBody1: "Images and answers you submit are processed temporarily, in memory, to generate a result. They are not written to disk or saved by this application, and this page does not use tracking cookies or analytics.",
    privacyBody2: "If you've enabled the optional external AI assistant (via an environment variable your operator sets), your assistant questions — not your image — may be sent to that external service. The local assistant mode sends nothing anywhere.",
    resetAnalysis: "Delete / reset this analysis",
    close: "Close",
    capture: "Capture",
    retake: "Retake",
    usePhoto: "Use photo",
    cancel: "Cancel",
    cameraUnavailable: "Camera is not available on this device or permission was denied. Please upload a photo instead.",
    qualityGood: "Good",
    qualityWarning: "Warning",
    qualityPoor: "Poor",
    tierRed: "Red — Refer urgently",
    tierYellow: "Yellow — Review soon",
    tierGreen: "Green — Routine",
    passed: "Passed",
    flagged: "Flagged",
    localAssistant: "Answered by the built-in local assistant (no external service used).",
    externalAssistant: "Answered by an external AI service configured by your operator.",
    limitationModel: "The lesion-category model may be untrained or unvalidated — see the AI analysis card's mode.",
    limitationDataset: "Any reference dataset used for similarity search is small and/or synthetic.",
    limitationNoClinical: "Nothing in this tool has undergone clinical validation.",
    limitationSecondOpinion: "Treat every output as a reason to seek a human, professional opinion — not as a result to act on directly.",
  },

  hi: {
    tagline: "एआई-सहायक स्वास्थ्य शोध उपकरण — यह चिकित्सा निदान नहीं है",
    checkingDevice: "डिवाइस जाँचा जा रहा है…",
    deviceOnline: "डिवाइस ऑनलाइन है",
    backendUnreachable: "बैकएंड से संपर्क नहीं हो पाया",
    privacy: "गोपनीयता",
    disclaimerStrong: "शैक्षणिक / शोध प्रोटोटाइप।",
    disclaimerBody: "यह उपकरण चिकित्सीय निदान नहीं करता। स्वास्थ्य संबंधी किसी भी चिंता के लिए हमेशा किसी योग्य चिकित्सक की सलाह लें।",
    newAssessment: "नया मूल्यांकन",
    lesionPhoto: "त्वचा के घाव की फोटो",
    uploadPrompt: "त्वचा के घाव की फोटो जोड़ने के लिए टैप करें",
    uploadHint: "कैमरा स्थिर रखें, फ्रेम भरें, प्राकृतिक रोशनी का उपयोग करें",
    takePhoto: "फोटो लें",
    dataSaver: "डेटा सेवर मोड",
    sizeReadoutTemplate: "मूल: {orig} — संकुचित: {compressed}",
    respiratorySymptoms: "श्वसन लक्षण (वैकल्पिक)",
    cough: "खांसी",
    coughDays: "खांसी कितने दिनों से है?",
    fever: "बुखार",
    breathingDifficulty: "सांस लेने में कठिनाई",
    none: "कोई नहीं",
    mild: "हल्की",
    severe: "गंभीर",
    chestIndrawing: "छाती धंसना देखा गया",
    lesionQuestionnaire: "इस घाव के बारे में",
    recentChange: "क्या इसमें हाल ही में बदलाव आया है?",
    sizeChange: "क्या इसका आकार बदला है?",
    appearanceChange: "क्या इसका रूप बदला है?",
    painful: "क्या इसमें दर्द होता है?",
    itchy: "क्या इसमें खुजली होती है?",
    bleeding: "क्या इसमें से खून आता है?",
    durationDays: "यह कितने दिनों से है?",
    runAnalysis: "ट्राएज विश्लेषण चलाएँ",
    pleaseAddPhoto: "कृपया पहले फोटो जोड़ें या लें।",
    stepQuality: "फोटो की गुणवत्ता जाँची जा रही है…",
    stepAnalyze: "विश्लेषण किया जा रहा है…",
    stepExplain: "व्याख्या तैयार की जा रही है…",
    stepPrepare: "परिणाम तैयार किया जा रहा है…",
    cardImageQuality: "फोटो की गुणवत्ता",
    cardAiAnalysis: "एआई विश्लेषण",
    category: "श्रेणी",
    confidence: "विश्वास स्तर",
    showAttention: "एआई का ध्यान क्षेत्र दिखाएँ",
    hideAttention: "एआई का ध्यान क्षेत्र छिपाएँ",
    original: "मूल फोटो",
    aiAttention: "एआई ध्यान क्षेत्र",
    gradcamDisclaimer: "हाइलाइट किया गया क्षेत्र दिखाता है कि मॉडल के निर्णय में किन हिस्सों का योगदान रहा। यह इस बात का प्रमाण नहीं है कि वह क्षेत्र चिकित्सकीय रूप से असामान्य है।",
    cardSimilar: "मिलते-जुलते उदाहरण",
    similarityDisclaimer: "दृश्य समानता का मतलब यह नहीं कि दो घावों का निदान एक जैसा है।",
    similarityUnavailable: "समानता खोज अभी सेट अप नहीं है।",
    cardTriage: "ट्राएज",
    whyThisResult: "यह परिणाम क्यों?",
    cardLimitations: "सीमाएँ",
    askAssistant: "इस परिणाम के बारे में पूछें",
    assistantInputLabel: "इस उपकरण के बारे में प्रश्न पूछें",
    assistantPlaceholder: "इस उपकरण के बारे में कोई प्रश्न पूछें…",
    send: "भेजें",
    learnHeading: "जानें",
    learnQ1: "त्वचा के घाव क्या होते हैं?",
    learnA1: "घाव त्वचा का कोई भी ऐसा हिस्सा है जो आसपास की त्वचा से अलग दिखता है — तिल, धब्बा, गांठ या पैच। ज़्यादातर घाव हानिरहित होते हैं, पर समय के साथ बदलाव पर ध्यान देना ज़रूरी है।",
    learnQ2: "सामान्य बनाम संदिग्ध श्रेणियाँ",
    learnA2: "\"सामान्य (Benign)\" का मोटे तौर पर मतलब है कैंसर-रहित। \"संदिग्ध\" श्रेणियाँ ऐसे पैटर्न हैं जो कभी-कभी त्वचा कैंसर से जुड़े होते हैं और आमतौर पर किसी विशेषज्ञ से जांच कराने की सलाह देते हैं — ये खुद निदान नहीं हैं।",
    learnQ3: "यह एआई वास्तव में क्या कर सकता है?",
    learnA3: "यह जांच सकता है कि फोटो उपयोग करने लायक है या नहीं, फोटो को व्यापक दृश्य श्रेणियों में बाँट सकता है, दिखा सकता है कि किन हिस्सों ने उस निर्णय को प्रभावित किया, और मिलते-जुलते संदर्भ उदाहरण खोज सकता है। यह त्वचा की वैसे जांच नहीं कर सकता जैसे कोई चिकित्सक करता है, और इसे नैदानिक उपयोग के लिए मान्य नहीं किया गया है।",
    learnQ4: "यह क्या नहीं कर सकता?",
    learnA4: "यह निदान नहीं कर सकता, किसी भी संभावना को खारिज नहीं कर सकता, घाव की बनावट महसूस नहीं कर सकता, और जब तक स्पष्ट रूप से \"प्रशिक्षित\" न बताया जाए — और तब भी \"नैदानिक रूप से मान्य\" न हो — इसके परिणाम केवल उदाहरणस्वरूप हैं, चिकित्सकीय रूप से सार्थक नहीं।",
    learnQ5: "बेहतर फोटो कैसे लें",
    learnA5: "यदि संभव हो तो प्राकृतिक दिन की रोशनी का उपयोग करें, कैमरा स्थिर रखें, फ्रेम में मुख्य हिस्से को भरें, और सीधी फ्लैश चमक से बचें।",
    footerNote: "यह प्रोटोटाइप शोध/शैक्षणिक प्रदर्शन के लिए है। यह प्रमाणित चिकित्सा उपकरण नहीं है।",
    privacyTitle: "गोपनीयता",
    privacyBody1: "आपके द्वारा भेजी गई फोटो और जवाब केवल अस्थायी रूप से, मेमोरी में, परिणाम तैयार करने के लिए संसाधित किए जाते हैं। इन्हें डिस्क पर सहेजा नहीं जाता, और यह पेज कोई ट्रैकिंग कुकीज़ या एनालिटिक्स उपयोग नहीं करता।",
    privacyBody2: "यदि आपके संचालक ने वैकल्पिक बाहरी एआई सहायक सक्षम किया है, तो आपके सहायक-प्रश्न — आपकी फोटो नहीं — उस बाहरी सेवा को भेजे जा सकते हैं। स्थानीय (local) सहायक मोड में कुछ भी कहीं नहीं भेजा जाता।",
    resetAnalysis: "इस विश्लेषण को हटाएँ / रीसेट करें",
    close: "बंद करें",
    capture: "फोटो खींचें",
    retake: "फिर से लें",
    usePhoto: "यह फोटो उपयोग करें",
    cancel: "रद्द करें",
    cameraUnavailable: "इस डिवाइस पर कैमरा उपलब्ध नहीं है या अनुमति नहीं मिली। कृपया इसके बजाय फोटो अपलोड करें।",
    qualityGood: "अच्छी",
    qualityWarning: "चेतावनी",
    qualityPoor: "खराब",
    tierRed: "लाल — तुरंत विशेषज्ञ को दिखाएँ",
    tierYellow: "पीला — जल्द समीक्षा करें",
    tierGreen: "हरा — नियमित",
    passed: "उत्तीर्ण",
    flagged: "चिह्नित",
    localAssistant: "अंतर्निहित स्थानीय सहायक द्वारा उत्तर दिया गया (कोई बाहरी सेवा उपयोग नहीं हुई)।",
    externalAssistant: "आपके संचालक द्वारा सेट की गई एक बाहरी एआई सेवा द्वारा उत्तर दिया गया।",
    limitationModel: "घाव-श्रेणी मॉडल अप्रशिक्षित या अमान्य हो सकता है — 'एआई विश्लेषण' कार्ड में मोड देखें।",
    limitationDataset: "समानता खोज के लिए उपयोग किया गया कोई भी संदर्भ डेटा छोटा और/या कृत्रिम (synthetic) है।",
    limitationNoClinical: "इस उपकरण में किसी भी चीज़ का नैदानिक सत्यापन नहीं हुआ है।",
    limitationSecondOpinion: "हर परिणाम को सीधे कार्रवाई के आधार के बजाय किसी मानव विशेषज्ञ की राय लेने का कारण मानें।",
  },
};

window.currentLanguage = "en";

function t(key) {
  const dict = window.I18N[window.currentLanguage] || window.I18N.en;
  return dict[key] || window.I18N.en[key] || key;
}

function applyLanguage(lang) {
  window.currentLanguage = lang;
  document.getElementById("htmlRoot").lang = lang;

  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.getAttribute("data-i18n"));
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.setAttribute("placeholder", t(el.getAttribute("data-i18n-placeholder")));
  });

  const toggle = document.getElementById("langToggle");
  if (toggle) {
    toggle.textContent = lang === "en" ? "🇮🇳 हिन्दी" : "🇬🇧 English";
  }

  if (typeof window.onLanguageChanged === "function") {
    window.onLanguageChanged(lang);
  }
}
