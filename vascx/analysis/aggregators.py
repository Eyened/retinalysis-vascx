from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
from typing import Callable, Union, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from vascx import Layer, Segment, Node, Bifurcation, Crossing
# FeatureType = Union[Layer, Segment, Node, Bifurcation, Crossing]


def mean(X):
    return np.mean(X)


def median(X):
    return np.median(X)


def std(X):
    return np.std(X)


def mean_std(X):
    return {"mean": mean(X), "std": std(X)}


def median_std(X):
    return {"median": median(X), "std": std(X)}


def mean_median_std(X):
    return {"mean": mean(X), "median": median(X), "std": std(X)}


class SegmentAggregator:
    pass


@dataclass
class BinnedSegmentAggregator(SegmentAggregator):
    """Aggregates measurements into bins of vessel diameter"""

    bins: Union[int, List[int]]
    fn: Callable

    def __call__(self, X: List[Tuple[float, Segment]]):
        X = [x for x in X if x[1].mean_diameter is not None]

        sorted_X = sorted(X, key=lambda x: x[1].mean_diameter)

        bins = self.bins
        if isinstance(bins, int):
            bins = np.linspace(0, 1, bins + 1)

        # X does not have enough segments.
        # return None to indicate invalid feature
        if len(X) < len(bins):
            return [None] * len(bins - 1)

        # list of quantiles passed, eg [0.25, 0.50, 0.75]
        values = []
        for i in range(len(bins) - 1):
            first = round(bins[i] * len(X))
            last = round(bins[i + 1] * len(X))

            data = sorted_X[first:last]
            data = [d[0] for d in data]
            values.append(self.fn(data))
        return values
