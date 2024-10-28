from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from vascx.shared.aggregators import mean_median_std
from vascx.shared.segment import Segment, SplineInterpolation
from vascx.shared.vessels import Vessels

from .base import FAZLayerFeature

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


class LengthMeasure(str, Enum):
    Splines = "splines"
    Skeleton = "skeleton"


class TortuosityMeasure(str, Enum):
    Distance = "distance"
    Curvature = "curvature"
    Inflections = "inflections"


@dataclass
class Tortuosity(FAZLayerFeature):
    def __init__(
        self,
        measure: TortuosityMeasure = TortuosityMeasure.Distance,
        length_measure: LengthMeasure = LengthMeasure.Splines,
        min_numpoints: int = 10,
        aggregator=mean_median_std,
        **kwargs,
    ):
        self.measure = measure
        self.length_measure = length_measure
        self.min_numpoints = min_numpoints
        self.aggregator = aggregator

    def _compute_for_segment(self, segment: Segment):
        if self.measure == TortuosityMeasure.Distance:
            if self.length_measure == LengthMeasure.Splines:
                return segment.spline.length() / segment.chord_length
            elif self.length_measure == LengthMeasure.Skeleton:
                return np.mean(segment.length / segment.chord_length)
            else:
                raise NotImplementedError()
        elif self.measure == TortuosityMeasure.Curvature:
            spline = SplineInterpolation(segment, 0.25)
            return np.mean(spline.curvatures())
        elif self.measure == TortuosityMeasure.Inflections:
            spline = SplineInterpolation(segment, 0.25)
            return len(spline.inflection_points(every=10))
        else:
            raise NotImplementedError()

    def get_segments(self, layer: VesselTreeLayer):
        return [
            segment
            for segment in layer.segments
            if len(segment.skeleton) >= self.min_numpoints
        ]

    def raw(self, layer: VesselTreeLayer):
        segments = self.get_segments(layer)
        lengths = np.array([vessel.length for vessel in segments])
        tortuosities = np.array(
            [self._compute_for_segment(vessel) for vessel in segments]
        )
        return tortuosities

    def compute(self, layer: VesselTreeLayer):
        tortuosities = self.raw(layer)
        return self.aggregator(tortuosities)

    def plot(self, ax, layer, **kwargs):
        if self.measure == TortuosityMeasure.Inflections:
            format = "{:d}"
        else:
            format = "{:.4f}"

        vessels = Vessels(layer, self.get_segments(layer))

        return vessels.plot(
            ax=ax,
            text=lambda x: format.format(self._compute_for_segment(x)),
            **{
                "show_index": True,
                "plot_endpoints": True,
                "plot_chord": True,
                "cmap": "tab20",
                **kwargs,
            },
        )
