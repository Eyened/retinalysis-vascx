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
    from rtnls_enface.grids.base import GridFieldEnum
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
    """Segment- or vessel-level tortuosity by distance ratio, curvature, or inflection counts.

    Representation: Uses segments or resolved_segments; for curvature uses Segment spline; 
    optionally length-normalized. Operates on either individual segments or merged vessel trees.

    Computation: Measures vessel tortuosity using three methods: distance ratio (arc length / chord length), 
    curvature (mean curvature along spline), or inflection counts (number of direction changes). 
    Can be computed per-segment or per-vessel (resolved segments), with optional length normalization.

    Options: mode (segments vs vessels), measure (distance/curvature/inflections), length_measure 
    (splines vs skeleton), norm_measure (length normalization), min_numpoints (length filter), 
    grid_field (spatial filtering), aggregator (statistical aggregation function).
    """
    
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
        grid_field: GridFieldEnum = None,
        norm_measure: LengthMeasure = None,
        aggregator=mean_median_std,
        **kwargs,
    ):
        self.mode = mode
        self.measure = measure
        self.length_measure = length_measure
        self.min_numpoints = min_numpoints
        self.grid_field = grid_field
        self.norm_measure = norm_measure
        self.aggregator = aggregator

    def __repr__(self) -> str:
        def fmt(v):
            import inspect, numpy as np
            from enum import Enum
            if v is None:
                return "None"
            if isinstance(v, Enum):
                return f"{v.__class__.__name__}.{v.name}"
            if callable(v):
                return getattr(v, "__name__", v.__class__.__name__)
            if isinstance(v, np.generic):
                return repr(v.item())
            return repr(v)
        return (
            f"Tortuosity(mode={fmt(self.mode)}, "
            f"measure={fmt(self.measure)}, "
            f"length_measure={fmt(self.length_measure)}, "
            f"min_numpoints={fmt(self.min_numpoints)}, "
            f"grid_field={fmt(self.grid_field)}, "
            f"norm_measure={fmt(self.norm_measure)}, "
            f"aggregator={fmt(self.aggregator)})"
        )

    def _compute_for_segment(self, segment: Segment, scale: float):
        if self.measure == TortuosityMeasure.Distance:
            if self.length_measure == LengthMeasure.Splines:
                return segment.spline.length() / segment.chord_length if segment.spline is not None else np.nan
            elif self.length_measure == LengthMeasure.Skeleton:
                return np.mean(segment.length / segment.chord_length)
            else:
                raise NotImplementedError()
        elif self.measure == TortuosityMeasure.Curvature:
            spline = SplineInterpolation(segment, 0.25)
            return np.mean(spline.curvatures()) * scale
        elif self.measure == TortuosityMeasure.Inflections:
            spline = SplineInterpolation(segment, 0.25)
            return len(spline.inflection_points(every=10))
        else:
            raise NotImplementedError()
        
    def _compute_norm_measure_for_segment(self, segment: Segment):
        if self.norm_measure == LengthMeasure.Splines:
            return segment.spline.length() if segment.spline is not None else np.nan
        elif self.norm_measure == LengthMeasure.Skeleton:
            return segment.length
        else:
            raise NotImplementedError()


    def get_segments(self, layer: VesselTreeLayer):
        if self.mode == TortuosityMode.Segments:
            field = None
            if self.grid_field is not None:
                grid = layer.retina.grids[self.grid_field.grid()]
                field = grid.field(self.grid_field)
            segments = layer.filter_segments(field=field, field_threshold=0.5)
            
        elif self.mode == TortuosityMode.Vessels:
            segments = layer.resolved_segments
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        
        segments = [
            segment
            for segment in segments
            if len(segment.skeleton) >= self.min_numpoints
        ]
        return segments

    def raw(self, layer: VesselTreeLayer):
        segments = self.get_segments(layer)

        vals = np.array(
            [self._compute_for_segment(vessel, scale=layer.retina.disc_fovea_distance) for vessel in segments]
        )

        if self.norm_measure is not None:
            norm_vals = np.array([self._compute_norm_measure_for_segment(seg) for seg in segments])
            vals = vals * norm_vals / np.nansum(norm_vals)
        return vals

    def compute(self, layer: VesselTreeLayer):
        tortuosities = self.raw(layer)
        return self.aggregator(tortuosities)

    def plot(self, ax, layer: VesselTreeLayer, **kwargs):
        segments = self.get_segments(layer)

        vessels = Vessels(layer, segments)
        if self.measure == TortuosityMeasure.Inflections:
            format = "{:d}"
        else:
            format = "{:.4f}"

        ax = vessels.plot(
            ax=ax,
            text=lambda x: format.format(self._compute_for_segment(x, scale=layer.retina.disc_fovea_distance)),
            cmap="tab20",
            min_numpoints=0,
            min_numpoints_caliber=self.min_numpoints,
            plot_endpoints=True,
            plot_chord=True,
            **kwargs,
        )

        # plot ETDRS region
        if self.grid_field is not None:
            grid = layer.retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
            grid.plot(ax, field)
        return ax
