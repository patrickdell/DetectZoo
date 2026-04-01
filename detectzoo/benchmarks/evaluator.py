"""Benchmark evaluator for running detectors across datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from tqdm import tqdm

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.datasets.base import BaseDataset, DatasetItem
from detectzoo.utils.logger import get_logger
from detectzoo.utils.metrics import compute_metrics

logger = get_logger(__name__)


class BenchmarkEvaluator:
    """Run one or more detectors on a dataset and compute metrics.

    Example::

        evaluator = BenchmarkEvaluator(dataset)
        results = evaluator.run([detector_a, detector_b])
        for name, metrics in results.items():
            print(name, metrics)
    """

    def __init__(self, dataset: BaseDataset) -> None:
        self.dataset = dataset

    def evaluate_single(
        self,
        detector: BaseDetector,
        items: Sequence[DatasetItem] | None = None,
    ) -> Dict[str, Any]:
        """Run *detector* on all items and return a metrics dictionary."""
        if items is None:
            items = self.dataset.load()

        labels: List[int] = []
        scores: List[float] = []
        predictions: List[DetectionResult] = []

        for item in tqdm(items, desc=detector.name):
            result = detector.predict(item.data)
            labels.append(item.label)
            scores.append(result.score)
            predictions.append(result)

        metrics = compute_metrics(labels, scores, threshold=detector.threshold)
        metrics["detector"] = detector.name
        metrics["n_samples"] = len(labels)
        return metrics

    def run(
        self,
        detectors: Sequence[BaseDetector],
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate multiple detectors and return ``{name: metrics}``."""
        items = self.dataset.load()
        results: Dict[str, Dict[str, Any]] = {}
        for det in detectors:
            logger.info("Evaluating '%s' …", det.name)
            results[det.name] = self.evaluate_single(det, items=items)
        return results

    def run_and_print(self, detectors: Sequence[BaseDetector]) -> None:
        """Evaluate detectors and print a comparison table."""
        all_results = self.run(detectors)
        header_keys = ["detector", "accuracy", "precision", "recall", "f1", "auroc"]
        header = " | ".join(f"{k:>18s}" for k in header_keys)
        print(header)
        print("-" * len(header))
        for metrics in all_results.values():
            row = " | ".join(f"{metrics.get(k, ''):>18}" if isinstance(metrics.get(k), str)
                             else f"{metrics.get(k, 0):>18.4f}" for k in header_keys)
            print(row)
