"""Minimal FastAPI front-end for DetectZoo — text/image/audio AI-detection demo."""

from __future__ import annotations

import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import markdown
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from detectzoo import load_detector

# aeroblade runs a full Stable Diffusion VAE encode/decode + LPIPS-VGG
# comparison with no internal resolution cap — a full-res phone photo
# (e.g. 4000x3000) blew past available memory and silently killed the
# container (no traceback, just a restart). Measured on the 7.7GB Haven
# host: 512px peaks ~850MB, 768px climbs past 1.6GB and gets OOM-killed.
# Capped well under that ceiling to leave headroom for concurrent requests.
MAX_IMAGE_EDGE = 512

app = FastAPI(title="DetectZoo")

STATIC_DIR = Path(__file__).parent / "static"
REPO_ROOT = Path(__file__).parent.parent

DOC_SOURCES = [
    ("README.md", "https://github.com/sadjadeb/DetectZoo/blob/main/README.md"),
    ("METHODS_AND_MODELS.md", "https://github.com/sadjadeb/DetectZoo/blob/main/METHODS_AND_MODELS.md"),
]

DETECTOR_NAMES = {
    "text": "roberta_base",
    "image": "aeroblade",
    "audio": "aasist",
}

CACHE_DIR = Path("/data/.detectzoo_data")

DETECTOR_KWARGS = {
    "text": {},
    "image": {},
    "audio": {"cache_dir": CACHE_DIR},
}


@lru_cache(maxsize=None)
def get_detector(modality: str):
    if modality not in DETECTOR_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown modality: {modality}")
    return load_detector(
        DETECTOR_NAMES[modality], device="cpu", **DETECTOR_KWARGS[modality]
    )


def _result_payload(result, modality: str) -> dict:
    return {
        "modality": modality,
        "detector": DETECTOR_NAMES[modality],
        "label": result.label,
        "score": result.score,
        "confidence": result.confidence,
    }


@app.post("/api/detect/text")
def detect_text(text: str = Form(...)):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input is empty.")
    detector = get_detector("text")
    result = detector.predict(text)
    return _result_payload(result, "text")


@app.post("/api/detect/image")
def detect_image(file: UploadFile = File(...)):
    detector = get_detector("image")
    try:
        img = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")
    img.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    result = detector.predict(img)
    return _result_payload(result, "image")


@app.post("/api/detect/audio")
def detect_audio(file: UploadFile = File(...)):
    detector = get_detector("audio")
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "audio").suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()
        result = detector.predict(tmp.name)
    return _result_payload(result, "audio")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@lru_cache(maxsize=1)
def _render_about_page() -> str:
    sections = []
    for filename, source_url in DOC_SOURCES:
        path = REPO_ROOT / filename
        if not path.exists():
            continue
        body_html = markdown.markdown(
            path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"]
        )
        sections.append(
            f'<section class="doc-source">'
            f'<p class="status">Source: <a href="{source_url}" target="_blank" rel="noopener">{filename}</a></p>'
            f"{body_html}</section>"
        )
    shell = (STATIC_DIR / "about_shell.html").read_text(encoding="utf-8")
    return shell.replace("__CONTENT__", "\n".join(sections))


@app.get("/about", response_class=HTMLResponse)
def about():
    return _render_about_page()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
