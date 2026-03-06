#!/usr/bin/env python
"""Compare text detectors on sample human and AI-written passages.

Usage:
    python examples/text_detection.py --device cuda
"""

from __future__ import annotations

import argparse

from detectzoo import list_detectors, load_detector

HUMAN_TEXTS = [
    (
        "I went to the grocery store yesterday and forgot my wallet. "
        "Had to drive all the way back home, which was frustrating because "
        "I was already running late for dinner with friends."
    ),
    (
        "The old oak tree in our backyard has been there since before my "
        "grandparents bought the house. Every spring it drops these little "
        "helicopters that my kids love to throw in the air."
    ),
]

AI_TEXTS = [
    (
        "Large language models have demonstrated remarkable capabilities "
        "across a wide range of natural language processing tasks. These "
        "models leverage vast amounts of training data to generate coherent "
        "and contextually appropriate responses."
    ),
    (
        "The implementation of transformer architectures has revolutionized "
        "the field of artificial intelligence. By utilizing self-attention "
        "mechanisms, these models can effectively capture long-range "
        "dependencies in sequential data."
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Text detection example")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    detectors = ["binoculars", "detectgpt", "fast_detectgpt"]

    print(f"All registered text detectors: {list_detectors('text')}")
    print(f"Running: {detectors}  (device={args.device})\n")

    detectors = [
        load_detector(name, device=args.device)
        for name in detectors
    ]

    for label, texts in [("HUMAN", HUMAN_TEXTS), ("AI", AI_TEXTS)]:
        print(f"=== {label} texts ===")
        for i, text in enumerate(texts):
            print(f"\n  [{label} #{i + 1}] {text[:70]}…")
            for det in detectors:
                result = det.predict(text)
                print(
                    f"    {det.name:20s}  score={result.score:+.4f}  "
                    f"label={result.label:5s}  confidence={result.confidence:.4f}"
                )
        print()


if __name__ == "__main__":
    main()
