from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar, List, Tuple

import numpy as np


class Aggregator:
    name: ClassVar[str]
    display_name: ClassVar[str] = ""

    def __call__(self, X):
        raise NotImplementedError


def check_and_warn(X):
    if np.sum(np.isnan(X)) > len(X) * 0.2:
        warnings.warn("More than 20% nans received by aggregator")


class Mean(Aggregator):
    name = "mn"
    display_name = "Mean"

    def __call__(self, X):
        if len(X) == 0:
            return None
        check_and_warn(X)
        return np.nanmean(X)


class Sum(Aggregator):
    name = "sum"
    display_name = "Sum"

    def __call__(self, X):
        if len(X) == 0:
            return None
        check_and_warn(X)
        return np.nansum(X)


class Median(Aggregator):
    name = "md"
    display_name = "Median"

    def __call__(self, X):
        if len(X) == 0:
            return None
        check_and_warn(X)
        return np.nanmedian(X)


class Std(Aggregator):
    name = "std"
    display_name = "Std"

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
