from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar, List, Tuple

import numpy as np


class Aggregator:
    name: ClassVar[str]
    display_name: ClassVar[str] = ""
    aggregation_label: ClassVar[str] = "summary"

    def describe(self, subject: str) -> str:
        """Describe how values are combined for publication-facing metadata."""
        return f"Reported as the {self.aggregation_label} across {subject}."

    def __call__(self, X):
        raise NotImplementedError


def check_and_warn(X):
    if np.sum(np.isnan(X)) > len(X) * 0.2:
        warnings.warn("More than 20% nans received by aggregator")


class Mean(Aggregator):
    name = "mn"
    display_name = "Mean"
    aggregation_label = "mean"

    def __call__(self, X):
        if len(X) == 0:
            return None
        check_and_warn(X)
        return np.nanmean(X)


class Sum(Aggregator):
    name = "sum"
    display_name = "Sum"
    aggregation_label = "sum"

    def __call__(self, X):
        if len(X) == 0:
            return None
        check_and_warn(X)
        return np.nansum(X)


class Median(Aggregator):
    name = "md"
    display_name = "Median"
    aggregation_label = "median"

    def __call__(self, X):
        if len(X) == 0:
            return None
        check_and_warn(X)
        return np.nanmedian(X)


class Std(Aggregator):
    name = "std"
    display_name = "Std"
    aggregation_label = "standard deviation"

    def __call__(self, X):
        if len(X) == 0:
            return None
        check_and_warn(X)
        return np.nanstd(X)


mean = Mean()
sum = Sum()
median = Median()
std = Std()


@dataclass
class LengthWeightedAggregator(Aggregator):
    """Aggregate `(weight, value)` pairs with normalized weights."""

    name: ClassVar[str] = "lw"
    display_name: ClassVar[str] = "Length-Weighted"

    def describe(self, subject: str) -> str:
        return f"Reported as the mean across {subject}, weighted by vessel length."

    def __call__(self, X: List[Tuple[float, float]]):
        if len(X) == 0:
            return None

        weights = np.asarray([weight for weight, _ in X], dtype=float)
        values = np.asarray([value for _, value in X], dtype=float)

        valid = np.isfinite(weights) & np.isfinite(values)
        if not np.any(valid):
            return None

        weights = weights[valid]
        values = values[valid]
        total_weight = np.sum(weights)
        if total_weight <= 0:
            return None

        return float(np.sum(weights * values) / total_weight)
