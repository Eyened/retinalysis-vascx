from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from vascx.shared.aggregators import mean_median_std
from vascx.shared.segment import Segment, SplineInterpolation
from vascx.shared.vessels import Vessels

from .base import LayerFeature

if TYPE_CHECKING:
    from rtnls_enface.grids.base import GridField
    from vascx.fundus.layer import VesselTreeLayer


class TortuosityMode(str, Enum):
    Segments = "segments"
    Vessels = "vessels"


class LengthMeasure(str, Enum):
    Splines = "splines"
    Skeleton = "skeleton"


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
        length_measure: LengthMeasure = LengthMeasure.Splines,
        min_numpoints: int = 25,
        grid_field: GridField = None,
        aggregator=mean_median_std,
        **kwargs,
    ):
        self.mode = mode
        self.measure = measure
        self.length_measure = length_measure
        self.min_numpoints = min_numpoints
        self.grid_field = grid_field
        self.aggregator = aggregator

    def _compute_for_segment(self, segment: Segment):
        if self.measure == TortuosityMeasure.Distance:
            if self.length_measure == LengthMeasure.Splines:
                return segment.spline.length() / segment.chord_length if segment.spline is not None else np.nan
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
        if self.mode == TortuosityMode.Segments:
            segments = layer.filter_segments(field=self.grid_field, field_threshold=0.5)
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

    def raw(self, layer: VesselTreeLayer):
        segments = self.get_segments(layer)
        tortuosities = np.array(
            [self._compute_for_segment(vessel) for vessel in segments]
        )
        return tortuosities

    def compute(self, layer: VesselTreeLayer):
        tortuosities = self.raw(layer)
        return self.aggregator(tortuosities)

    def explain(self):
        pass

    def calc_auxiliary(self):
        pass

    def plot(self, ax, layer, **kwargs):
        segments = self.get_segments(layer)

        vessels = Vessels(layer, segments)
        if self.measure == TortuosityMeasure.Inflections:
            format = "{:d}"
        else:
            format = "{:.4f}"

        ax = vessels.plot(
            ax=ax,
            text=lambda x: format.format(self._compute_for_segment(x)),
            cmap="tab20",
            min_numpoints=0,
            min_numpoints_caliber=self.min_numpoints,
            plot_endpoints=True,
            plot_chord=True,
            **kwargs,
        )

        # plot ETDRS region
        if self.grid_field is not None:
            layer.retina.grids[self.grid_field.grid()].plot(ax, self.grid_field)
        return ax
