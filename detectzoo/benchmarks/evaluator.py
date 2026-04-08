"""Benchmark evaluator for running detectors across datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

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
        *,
        save_scores: bool = False,
    ) -> Dict[str, Any]:
        """Run *detector* on all items and return a metrics dictionary.

        Parameters
        ----------
        save_scores:
            If ``True``, the returned dict includes a ``"samples"`` key with
            per-sample labels and scores.
        """
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

        if save_scores:
            metrics["samples"] = [
                {"label": lbl, "score": scr}
                for lbl, scr in zip(labels, scores)
            ]

        return metrics

    def run(
        self,
        detectors: Sequence[BaseDetector],
        *,
        save_scores: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate multiple detectors and return ``{name: metrics}``."""
        items = self.dataset.load()
        results: Dict[str, Dict[str, Any]] = {}
        for det in detectors:
            logger.info("Evaluating '%s' …", det.name)
            results[det.name] = self.evaluate_single(det, items=items, save_scores=save_scores)
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

    def run_and_save(
        self,
        detectors: Sequence[BaseDetector],
        output_path: Union[str, Path],
        *,
        save_scores: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate detectors and save the results to a JSON file.

        Parameters
        ----------
        detectors:
            Detectors to benchmark.
        output_path:
            Destination file path.  Parent directories are created
            automatically if they don't exist.
        save_scores:
            If ``True``, per-sample labels and scores are included in the
            saved output under each detector's ``"samples"`` key.

        Returns
        -------
        The same ``{name: metrics}`` dictionary that :meth:`run` returns.
        """
        all_results = self.run(detectors, save_scores=save_scores)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2, default=str))

        logger.info("Results saved to %s", output_path)
        return all_results
