FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch wheels — avoids pulling multi-GB CUDA packages.
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml MANIFEST.in README.md METHODS_AND_MODELS.md ./
COPY detectzoo ./detectzoo
RUN pip install --no-cache-dir -e .

COPY webapp ./webapp
RUN pip install --no-cache-dir -r webapp/requirements.txt

# HF transformers/diffusers and torch.hub both read these — pointing them at
# the mounted volume means model weights survive container rebuilds instead
# of being silently lost in the writable layer.
ENV HF_HOME=/data/hf_cache
ENV TORCH_HOME=/data/torch_cache
VOLUME ["/data"]

EXPOSE 8000
CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
