from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from vascx.analysis.aggregators import mean_median_std
from vascx.segment import Segment, SplineInterpolation

from .base import LayerFeature

if TYPE_CHECKING:
    from vascx.retina import VesselLayer


class TortuosityMode(str, Enum):
    Segments = "segments"
    Vessels = "vessels"


class Zone(tuple, Enum):
    All = None
    A = (0.0, 0.5)
    B = (0.5, 1.0)
    C = (1.0, 2.0)


class TortuosityMeasure(str, Enum):
    Distance = "distance"
    Curvature = "curvature"
    Inflections = "inflections"


@dataclass
class Tortuosity(LayerFeature):
    # Ideas
    # tortuosity for different levels of caliber
    #   what happens when small vessels not visible
    # tortuosity for different generations

    def __init__(
        self,
        mode: TortuosityMode = TortuosityMode.Segments,
        measure: TortuosityMeasure = TortuosityMeasure.Distance,
        min_numpoints: int = 25,
        zone: Zone = Zone.All,
        aggregator=mean_median_std,
        **kwargs,
    ):
        self.mode = mode
        self.measure = measure
        self.min_numpoints = min_numpoints
        self.zone = zone
        self.aggregator = aggregator

    def _compute_for_segment(self, segment: Segment):
        if self.measure == TortuosityMeasure.Distance:
            return np.mean(segment.length / segment.chord_length)
        elif self.measure == TortuosityMeasure.Curvature:
            spline = SplineInterpolation(segment, 0.25)
            return np.mean(spline.curvatures())
        elif self.measure == TortuosityMeasure.Inflections:
            spline = SplineInterpolation(segment, 0.25)
            return len(spline.inflection_points(every=10))
        else:
            raise NotImplementedError()

    def get_segments(self, layer: VesselLayer):
        if self.mode == TortuosityMode.Segments:
            segments = layer.segments
            segments = [
                segment
                for segment in segments
                if len(segment.skeleton) >= self.min_numpoints
            ]
        elif self.mode == TortuosityMode.Vessels:
            segments = layer.vessels.filter_segments_by_numpoints(self.min_numpoints)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        return segments

    def compute(self, layer: VesselLayer):
        segments = self.get_segments(layer)
        lengths = np.array([vessel.length for vessel in segments])
        tortuosities = np.array(
            [self._compute_for_segment(vessel) for vessel in segments]
        )
        return self.aggregator(tortuosities)
        # return np.sum(lengths * tortuosities) / np.sum(lengths)

    def explain(self):
        pass

    def calc_auxiliary(self):
        pass

    def plot(self, layer, **kwargs):
        if self.measure == TortuosityMeasure.Inflections:
            format = "{:d}"
        else:
            format = "{:.4f}"

        return layer.vessels.plot(
            text=lambda x: format.format(self._compute_for_segment(x)),
            **{
                "show_index": True,
                "plot_endpoints": True,
                "plot_chord": True,
                "cmap": "tab20",
                **kwargs,
            },
        )
