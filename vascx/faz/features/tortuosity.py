from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from vascx.shared.aggregators import median
from vascx.shared.segment import Segment
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


class Tortuosity(FAZLayerFeature):
    def __init__(
        self,
        measure: TortuosityMeasure = TortuosityMeasure.Distance,
        length_measure: LengthMeasure = LengthMeasure.Splines,
        min_numpoints: int = 10,
        aggregator=median,
        spline_error_fraction: float | None = None,
        **kwargs,
    ):
        self.measure = measure
        self.length_measure = length_measure
        self.min_numpoints = min_numpoints
        self.aggregator = aggregator
        self.spline_error_fraction = (
            None
            if spline_error_fraction is None
            else float(spline_error_fraction)
        )

    def _resolved_spline_error_fraction(self) -> float:
        if self.spline_error_fraction is not None:
            return self.spline_error_fraction
        if self.measure in (
            TortuosityMeasure.Curvature,
            TortuosityMeasure.Inflections,
        ):
            return 0.25
        return 0.05

    def _compute_for_segment(self, segment: Segment):
        spline_error_fraction = self._resolved_spline_error_fraction()
        if self.measure == TortuosityMeasure.Distance:
            if self.length_measure == LengthMeasure.Splines:
                spline = segment.get_spline(error_fraction=spline_error_fraction)
                return spline.length() / segment.chord_length
            elif self.length_measure == LengthMeasure.Skeleton:
                return np.mean(segment.length / segment.chord_length)
            else:
                raise NotImplementedError()
        elif self.measure == TortuosityMeasure.Curvature:
            spline = segment.get_spline(error_fraction=spline_error_fraction)
            return np.mean(spline.curvatures())
        elif self.measure == TortuosityMeasure.Inflections:
            spline = segment.get_spline(error_fraction=spline_error_fraction)
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

    def _plot(self, ax, layer, **kwargs):
        if self.measure == TortuosityMeasure.Inflections:
            format = "{:d}"
        else:
            format = "{:.4f}"

        vessels = Vessels(layer, self.get_segments(layer))

        return vessels.plot(
            ax=ax,
            text=lambda x: f'{100*(self._compute_for_segment(x)-1):.2f}',
            **{
                "show_index": False,
                "plot_endpoints": True,
                "plot_chord": True,
                "cmap": "tab20",
                **kwargs,
            },
        )
