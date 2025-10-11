"""Metric helpers for FAIR-Chem training loops."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict


class RunningAverage:
    """Track the running average of a scalar quantity."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    def compute(self) -> float:
        return self.total / max(self.count, 1)


class BucketedMAE:
    """Track MAE per bucket (e.g., degree or composition)."""

    def __init__(self) -> None:
        self._mae: Dict[str, RunningAverage] = defaultdict(RunningAverage)

    def update(self, bucket: str, error: float, n: int = 1) -> None:
        self._mae[bucket].update(error, n=n)

    def as_dict(self) -> Dict[str, float]:
        return {bucket: tracker.compute() for bucket, tracker in self._mae.items()}
