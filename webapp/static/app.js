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

function renderResult(modality, data) {
  const slot = document.querySelector(`[data-result-for="${modality}"]`);
  const isAi = data.label === "ai";
  slot.innerHTML = `
    <div class="result ${isAi ? "is-ai" : ""}">
      <p class="eyebrow">Result — ${data.detector}</p>
      <h2>${isAi ? "Likely AI-generated" : "Likely human-made"}</h2>
      <dl>
        <dt>Label</dt><dd>${data.label}</dd>
        <dt>Score</dt><dd>${data.score.toFixed(4)}</dd>
        <dt>Confidence</dt><dd>${(data.confidence * 100).toFixed(1)}%</dd>
      </dl>
    </div>
  `;
}

function renderError(modality, message) {
  const slot = document.querySelector(`[data-result-for="${modality}"]`);
  slot.innerHTML = `<p class="error">${message}</p>`;
}

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

document.getElementById("form-text").addEventListener("submit", (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  handleSubmit("text", e.target, formData);
});

document.getElementById("form-image").addEventListener("submit", (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  handleSubmit("image", e.target, formData);
});

document.getElementById("form-audio").addEventListener("submit", (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  handleSubmit("audio", e.target, formData);
});
