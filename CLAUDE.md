# DetectZoo — deployment notes

Upstream project: https://github.com/sadjadeb/DetectZoo (Python library, AI-content detection across text/image/audio — no web UI upstream).

## What was built

A FastAPI web front-end (`webapp/`) wraps the library for a public demo at **detect.patrickdell.ca**:

- `webapp/main.py` — FastAPI app. `/api/detect/{text,image,audio}` run one detector per modality via `detectzoo.load_detector()`. `/` serves the demo UI, `/about` renders `README.md` + `METHODS_AND_MODELS.md` from the repo root through `markdown` into the branded shell (`webapp/static/about_shell.html`) — this is how the GitHub docs are surfaced in-app without a build step; it re-reads the files from disk on every request, so a `git pull` + container restart is enough to refresh it.
- `webapp/static/` — UI styled per `C:\Code\handbrakedecider\branding CLAUDE.md` (Globe and Mail design system: GM Sans/Pratt fonts, globe-red accent, square corners, AAA contrast).
- Detectors chosen for CPU-friendliness and no required config: `fast_detectgpt` (text), `aeroblade` (image), `rawnet2` (audio). Each downloads its model weights on first use per modality — first request per modality is slow (gpt2 ~500MB, the SD v1.1 VAE + LPIPS VGG ~300MB, rawnet2 checkpoint ~80MB).

## Why these detectors (and a caught bug)

There's no detector-selection UI (single detector per modality, no model swapping) — picked the ones from the README's own quick-start examples as the simplest "it just works" defaults. If more detectors are wanted later, extend `DETECTOR_NAMES` in `main.py` and add a `<select>` to the relevant panel in `index.html`.

**`fast_detectgpt`'s real defaults are `EleutherAI/gpt-neo-2.7B` + `gpt-j-6B`** — multi-GB models, impractical on CPU (the library's own test suite overrides both to `gpt2` for exactly this reason, see `tests/test_text_detectors.py`). `webapp/main.py`'s `DETECTOR_KWARGS["text"]` pins both to `gpt2` — this was caught during a post-deploy audit when a text-detection request hung indefinitely; don't remove this override.

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
