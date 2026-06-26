# DetectZoo — deployment notes

Upstream project: https://github.com/sadjadeb/DetectZoo (Python library, AI-content detection across text/image/audio — no web UI upstream).

## What was built

A FastAPI web front-end (`webapp/`) wraps the library for a public demo at **detect.patrickdell.ca**:

- `webapp/main.py` — FastAPI app. `/api/detect/{text,image,audio}` run one detector per modality via `detectzoo.load_detector()`. `/` serves the demo UI, `/about` renders `README.md` + `METHODS_AND_MODELS.md` from the repo root through `markdown` into the branded shell (`webapp/static/about_shell.html`) — this is how the GitHub docs are surfaced in-app without a build step; it re-reads the files from disk on every request, so a `git pull` + container restart is enough to refresh it.
- `webapp/static/` — UI styled per `C:\Code\handbrakedecider\branding CLAUDE.md` (Globe and Mail design system: GM Sans/Pratt fonts, globe-red accent, square corners, AAA contrast).
- Detectors chosen for CPU-friendliness and no required config: `roberta_base` (text), `aeroblade` (image), `aasist` (audio). Each downloads its model weights on first use per modality — first request per modality is slow (roberta_base ~500MB, the SD v1.1 VAE + LPIPS VGG ~300MB, aasist checkpoint ~1.3MB).

## Why these detectors (and a caught bug)

There's no detector-selection UI (single detector per modality, no model swapping) — picked the ones from the README's own quick-start examples as the simplest "it just works" defaults, then revised after a post-deploy audit (below). If more detectors are wanted later, extend `DETECTOR_NAMES` in `main.py` and add a `<select>` to the relevant panel in `index.html`.

**Text was originally `fast_detectgpt`, switched to `roberta_base`.** `fast_detectgpt`'s real defaults are `EleutherAI/gpt-neo-2.7B` + `gpt-j-6B` — multi-GB models, impractical on CPU (the library's own test suite overrides both to `gpt2` for exactly this reason, see `tests/test_text_detectors.py`). The `gpt2`/`gpt2` workaround technically ran, but degrades the method to a much weaker signal — it's designed to compare two *different*-capability models, not two identical ones. `roberta_base` (`openai-community/roberta-base-openai-detector`, 125M params) is a purpose-trained classifier for this exact task: fixed size, single fast forward pass, no risky defaults. Trade-off: it was fine-tuned on GPT-2-era text specifically, so it's somewhat dated against modern LLM output, but it's a more honest setup than the curvature hack.

**Audio was originally `rawnet2`, switched to `aasist`.** AASIST (2022) is the architecture that superseded RawNet2 (2021) and is generally benchmarked as more accurate on ASVspoof2019 LA. Its checkpoint is also **1.3MB vs RawNet2's 70MB** — a clear win with no real downside. (There's an even smaller `AASIST-L` variant, ~85k params, available via `variant="light"` if it's ever worth shaving further.)

**Model cache persistence:** the original Dockerfile set an invented `DETECTZOO_CACHE_DIR` env var that the library never reads — model weights were silently downloading into the container's writable layer (lost on every rebuild) instead of the mounted `/data` volume. Fixed by setting the env vars the underlying libraries actually read — `HF_HOME` and `TORCH_HOME` (both under `/data`) for HuggingFace/torch.hub downloads, plus an explicit `cache_dir=/data/.detectzoo_data` kwarg for `rawnet2` (the one detector using DetectZoo's own download helper, which only reads cache location from a constructor kwarg — not an env var). Verify after any Dockerfile change that `docker exec detectzoo du -sh /data/*` actually grows.

## Deployment

- `Dockerfile` — `python:3.11-slim`, installs CPU-only torch wheels explicitly (`--index-url https://download.pytorch.org/whl/cpu`) to avoid pulling CUDA packages, then `pip install -e .` for the library, then `webapp/requirements.txt`.
- `docker-compose.yml` — container `detectzoo`, host port **8084** → container 8000, named volume for model cache persistence across rebuilds.
- Cloudflare Tunnel ingress rule added in `C:\Users\patri\.cloudflared\config.yml`: `detect.patrickdell.ca` → `http://host.docker.internal:8084`. DNS: CNAME `detect` → `d72aa79e-9eb4-4536-ae32-90fc6da820eb.cfargotunnel.com`, proxied — added manually via the Cloudflare dashboard (no API token available locally).

### Redeploy after changing webapp code

```
docker compose up -d --build
```

Image build takes ~3 min cached / longer if torch/transformers/etc. change. The `cloudflared` container does **not** need restarting for app code changes — only if `config.yml` itself changes (`docker restart cloudflared`).

## Footer / credits

Site footer links the source repo, the three DetectZoo authors (`sadjadeb`, `nimajam41`, `BardiaShir`, from `git log`), and `github.com/patrickdell` (found in `patrickdell-site/html/index.html`).
