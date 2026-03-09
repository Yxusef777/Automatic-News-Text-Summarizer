/**
 * script.js – Frontend logic for the Automatic News Text Summarizer.
 *
 * Sends the article text to the FastAPI backend and renders the
 * generated abstractive summary in the results section.
 */

"use strict";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/** Base URL of the FastAPI backend. Change this if your server runs elsewhere. */
const API_BASE_URL = "http://localhost:8000";

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const articleTextarea   = document.getElementById("articleText");
const languageSelect    = document.getElementById("languageSelect");
const summarizeBtn      = document.getElementById("summarizeBtn");
const loadingIndicator  = document.getElementById("loadingIndicator");
const errorMessage      = document.getElementById("errorMessage");
const resultsSection    = document.getElementById("resultsSection");
const summaryOutput     = document.getElementById("summaryOutput");
const languageBadge     = document.getElementById("languageBadge");
const modelBadge        = document.getElementById("modelBadge");

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

summarizeBtn.addEventListener("click", handleSummarize);

// ---------------------------------------------------------------------------
// Main handler
// ---------------------------------------------------------------------------

/**
 * Handle a click on the "Summarize" button.
 * Validates input, calls the API, and renders the results.
 */
async function handleSummarize() {
  const text     = articleTextarea.value.trim();
  const language = languageSelect.value;

  // Basic client-side validation
  if (!text) {
    showError("Please paste a news article before clicking Summarize.");
    return;
  }

  // Prepare UI for loading state
  setLoading(true);
  hideError();
  hideResults();

  try {
    const data = await callSummarizeAPI(text, language);
    renderResults(data);
  } catch (err) {
    showError(
      err.message ||
        "An unexpected error occurred. Please ensure the backend server is running."
    );
  } finally {
    setLoading(false);
  }
}

// ---------------------------------------------------------------------------
// API call
// ---------------------------------------------------------------------------

/**
 * POST to /summarize and return the parsed JSON response.
 *
 * @param {string} text     - The news article to summarise.
 * @param {string} language - Language code: "auto", "en", or "ar".
 * @returns {Promise<Object>} The JSON response body.
 */
async function callSummarizeAPI(text, language) {
  const response = await fetch(`${API_BASE_URL}/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      language,
      max_length: 150,
    }),
  });

  if (!response.ok) {
    // Try to extract a meaningful error message from the response body
    let detail = `Server returned status ${response.status}.`;
    try {
      const errorBody = await response.json();
      if (errorBody.detail) {
        detail = typeof errorBody.detail === "string"
          ? errorBody.detail
          : JSON.stringify(errorBody.detail);
      }
    } catch (_) {
      // Ignore JSON parse errors on the error response
    }
    throw new Error(detail);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

/**
 * Render the summarization result in the results section.
 *
 * @param {Object} data - Response object from the API.
 * @param {string} data.summary        - The generated summary text.
 * @param {string} data.language       - Detected/specified language code.
 * @param {string} data.model_used     - Model identifier.
 * @param {number} data.original_length - Character length of the source text.
 */
function renderResults(data) {
  summaryOutput.textContent = data.summary;

  // Apply RTL direction for Arabic output
  if (data.language === "ar") {
    summaryOutput.classList.add("rtl");
  } else {
    summaryOutput.classList.remove("rtl");
  }

  // Build human-readable language label
  const languageLabel = {
    ar: "🇸🇦 Arabic",
    en: "🇬🇧 English",
  }[data.language] || `Language: ${data.language}`;

  languageBadge.textContent = languageLabel;
  modelBadge.textContent    = `Model: ${data.model_used}`;

  resultsSection.classList.remove("hidden");
  // Scroll smoothly into view
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

/**
 * Toggle the loading spinner and disable/enable the Summarize button.
 *
 * @param {boolean} isLoading - Whether the app is currently loading.
 */
function setLoading(isLoading) {
  if (isLoading) {
    loadingIndicator.classList.remove("hidden");
    summarizeBtn.disabled = true;
  } else {
    loadingIndicator.classList.add("hidden");
    summarizeBtn.disabled = false;
  }
}

/**
 * Display an error message to the user.
 *
 * @param {string} message - The error message to display.
 */
function showError(message) {
  errorMessage.textContent = `⚠️ ${message}`;
  errorMessage.classList.remove("hidden");
}

/** Hide the error message container. */
function hideError() {
  errorMessage.textContent = "";
  errorMessage.classList.add("hidden");
}

/** Hide the results section. */
function hideResults() {
  resultsSection.classList.add("hidden");
}
