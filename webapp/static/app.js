const tabButtons = document.querySelectorAll(".tab-btn");
const panels = document.querySelectorAll(".panel");

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabButtons.forEach((b) => b.classList.remove("active"));
    panels.forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`panel-${btn.dataset.tab}`).classList.add("active");
  });
});

function setStatus(modality, text) {
  document.querySelector(`[data-status-for="${modality}"]`).textContent = text;
}

function formatScore(data) {
  return data.score.toFixed(4);
}

function formatConfidence(data) {
  return `${(data.confidence * 100).toFixed(1)}%`;
}

function verdictFor(data) {
  return data.label === "ai" ? "Likely AI-generated" : "Likely human-made";
}

function resultToText(data) {
  return [
    `DetectZoo — ${data.modality} detection (${data.detector})`,
    verdictFor(data),
    `Label: ${data.label}`,
    `Score: ${formatScore(data)}`,
    `Confidence: ${formatConfidence(data)}`,
  ].join("\n");
}

function renderResult(modality, data) {
  const slot = document.querySelector(`[data-result-for="${modality}"]`);
  const isAi = data.label === "ai";
  slot.innerHTML = `
    <div class="result ${isAi ? "is-ai" : ""}">
      <p class="eyebrow">Result — ${data.detector}</p>
      <h2>${verdictFor(data)}</h2>
      <dl>
        <dt>Label</dt><dd>${data.label}</dd>
        <dt>Score</dt><dd>${formatScore(data)}</dd>
        <dt>Confidence</dt><dd>${formatConfidence(data)}</dd>
      </dl>
      <div class="copy-row">
        <button type="button" class="copy-btn" data-copy-result="${modality}">Copy findings</button>
      </div>
    </div>
  `;
  slot.querySelector(".copy-btn")._resultText = resultToText(data);
}

function renderError(modality, message) {
  const slot = document.querySelector(`[data-result-for="${modality}"]`);
  slot.innerHTML = `<p class="error">${message}</p>`;
}

document.addEventListener("click", async (e) => {
  const btn = e.target.closest(".copy-btn");
  if (!btn) return;
  try {
    await navigator.clipboard.writeText(btn._resultText || "");
    const original = btn.textContent;
    btn.textContent = "Copied!";
    btn.classList.add("copied");
    setTimeout(() => {
      btn.textContent = original;
      btn.classList.remove("copied");
    }, 1500);
  } catch {
    btn.textContent = "Copy failed";
  }
});

async function handleSubmit(modality, form, formData) {
  const submitBtn = form.querySelector("button[type=submit]");
  submitBtn.disabled = true;
  setStatus(modality, "Analyzing — first run may download model weights…");
  document.querySelector(`[data-result-for="${modality}"]`).innerHTML = "";

  try {
    const res = await fetch(`/api/detect/${modality}`, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Request failed");
    }
    const data = await res.json();
    renderResult(modality, data);
    setStatus(modality, "");
  } catch (err) {
    renderError(modality, err.message);
    setStatus(modality, "");
  } finally {
    submitBtn.disabled = false;
  }
}

["text", "image", "audio"].forEach((modality) => {
  document.getElementById(`form-${modality}`).addEventListener("submit", (e) => {
    e.preventDefault();
    handleSubmit(modality, e.target, new FormData(e.target));
  });
});

// Drag-and-drop support for the image and audio file inputs.
document.querySelectorAll(".dropzone").forEach((zone) => {
  const modality = zone.dataset.dropzoneFor;
  const input = zone.querySelector(".dropzone-input");
  const filenameEl = document.querySelector(`[data-filename-for="${modality}"]`);

  function showFilename() {
    filenameEl.textContent = input.files.length ? input.files[0].name : "";
  }

  input.addEventListener("change", showFilename);

  ["dragenter", "dragover"].forEach((evt) => {
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.add("is-dragover");
    });
  });

  ["dragleave", "dragend"].forEach((evt) => {
    zone.addEventListener(evt, () => zone.classList.remove("is-dragover"));
  });

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("is-dragover");
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      showFilename();
    }
  });
});

// Prevent the browser from navigating away if a file is dropped outside a dropzone.
["dragover", "drop"].forEach((evt) => {
  window.addEventListener(evt, (e) => {
    if (!e.target.closest(".dropzone")) e.preventDefault();
  });
});
