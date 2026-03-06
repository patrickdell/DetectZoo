# DetectZoo

**A unified toolkit for detecting AI-generated content**

DetectZoo is a research-oriented Python toolkit that provides **implementations of AI-generated content detectors across multiple modalities**, including **text, images, and audio**.

The goal of DetectZoo is to make detection methods **easy to use, reproducible, and extensible**, enabling researchers and practitioners to benchmark and deploy AI-generated content detectors with minimal effort.

DetectZoo aggregates detection approaches into a **single, unified API**, allowing users to load and apply detectors with just a few lines of code.

---

## Installation

```bash
pip install detectzoo
```

or install from source with all dependencies:

```bash
git clone https://github.com/sadjadeb/detectzoo.git
cd detectzoo
pip install -e ".[all]"
```

### Optional extras

Install only the dependencies you need:

```bash
pip install detectzoo[all]     # all dependencies
pip install detectzoo[text]    # transformers, accelerate
pip install detectzoo[dev]     # all + pytest, ruff
```

---

## Quick Start

### Detect AI-generated text

```python
from detectzoo import load_detector

detector = load_detector("fast_detectgpt")

text = "Large language models are transforming many fields."
result = detector.predict(text)

print(result)
# DetectionResult(score=1.2345, label='ai', confidence=0.8012)
print(result.score, result.label)
```

### List all available detectors

```python
from detectzoo import list_detectors

print(list_detectors())            # all detectors
print(list_detectors("text"))      # text-only
```

---

## Supported Detectors

DetectZoo organizes detectors by **modality**. Every detector follows the same interface: `detector.predict(input) → DetectionResult`.

### Text

Detectors for identifying LLM-generated text. Each accepts a string (or file path) and uses a HuggingFace causal language model internally.

| Name | Class | Method |
|------|-------|--------|
| `detectgpt` | `DetectGPTDetector` | Perturbation-based probability curvature (Mitchell et al., ICML 2023). Uses T5 to generate perturbations and measures log-prob curvature. |
| `fast_detectgpt` | `FastDetectGPTDetector` | Perturbation-free curvature (Bao et al., ICLR 2024). Estimates curvature from the model's own conditional distribution without generating perturbations. |
| `binoculars` | `BinocularsDetector` | Two-model perplexity ratio (Pagnoni et al., 2024). Compares an observer and performer model. |

---

## Core Components

### DetectionResult

Every `predict()` call returns a `DetectionResult` dataclass:

```python
@dataclass
class DetectionResult:
    score: float       # Higher = more likely AI-generated
    label: str         # "ai" or "human"
    confidence: float  # Confidence in the label (0–1)
    metadata: dict     # Detector-specific extra info
```

The `metadata` dictionary varies by detector and may include values like `avg_log_likelihood`, `mean_curvature`, `ppl_observer`, `hf_lf_ratio`, etc.

---

### Metrics

The `compute_metrics` utility computes standard binary-classification metrics:

```python
from detectzoo.utils import compute_metrics

metrics = compute_metrics(
    labels=[0, 0, 1, 1],
    scores=[0.1, 0.3, 0.8, 0.9],
    threshold=0.5,
)
# {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'auroc': 1.0, 'avg_precision': 1.0}
```

---

## Features

* **Multimodal detection**

  * Text (LLM-generated text)
  * Images (diffusion / GAN generated images)
  * Audio (synthetic speech / deepfake audio)

* **Unified API**

  * Consistent interface across all detectors — every detector returns a `DetectionResult` with a score, label, confidence, and metadata

* **12 detectors** spanning three modalities, including published research methods (DetectGPT, Fast-DetectGPT, Binoculars) and practical baselines

* **Reproducible implementations**

  * Clean implementations of published detection methods

* **Benchmark-ready**

  * Built-in dataset loaders and an evaluation pipeline for comparing detectors

* **Modular architecture**

  * You can easily add a new detector by subclassing `BaseDetector` and registering it with the `register_detector` decorator.

* **Lightweight and research-friendly**

  * Optional dependencies per modality — install only what you need from the following: text, image, audio, eval.

---


## Design Philosophy

DetectZoo is built around three principles.

### 1. Reproducibility

Many detection methods are difficult to reproduce due to missing implementation details. DetectZoo provides **clean and standardized implementations of published detectors** with references to the original papers.

### 2. Accessibility

Users should not need to reimplement detectors. DetectZoo provides **simple imports and unified interfaces**. Loading any detector is a single function call.

### 3. Extensibility

Adding a new detector takes a single file. Subclass `BaseDetector`, implement `predict`, and register with a decorator:

```python
from detectzoo.detectors import BaseDetector
from detectzoo.core.registry import register_detector

@register_detector("my_detector")
class MyDetector(BaseDetector):
    modality = "text"  # or "image" or "audio"

    def __init__(self, threshold=0.5, device="cpu", **kwargs):
        super().__init__(threshold=threshold, device=device, **kwargs)

    def predict(self, input_data):
        # Your detection logic here
        score = 0.0
        return self._make_result(score)
```

The detector is then immediately available via `load_detector("my_detector")`. See `examples/custom_detector.py` for a complete runnable example.

---

## Examples

The `examples/` directory contains self-contained scripts you can run immediately:

| Script | Description |
|--------|-------------|
| `text_detection.py` | Compare text detectors (log-likelihood, log-rank, entropy, fast-detectgpt) on sample human and AI passages. |

Run any example from the project root:

```bash
python examples/text_detection.py --device cuda
```

---

## Contributing

We welcome community contributions.

You can contribute by:

* Adding new detectors (see the extensibility section above)
* Improving existing implementations
* Adding benchmark datasets
* Improving documentation
* Reporting issues and suggesting features

---

## Roadmap

Planned improvements include:

* More detectors for each modality (watermark-based detectors, Deepfake-in-the-Wild models, etc.)
* Pre-trained weights for CNN and spectrogram detectors
* Built-in download and caching for common benchmark datasets (TruthfulQA, HC3, ASVspoof, GenImage, etc.)
* Training scripts and configuration files
* Leaderboard generation
* Visualization tools for detection scores and attention maps
