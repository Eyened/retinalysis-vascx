from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from vascx.shared.aggregators import LengthWeightedAggregator, median
from vascx.shared.segment import Segment, SplineInterpolation
from vascx.shared.vessels import Vessels

from .base import LayerFeature, grid_field_fraction_in_bounds

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


class TortuosityMode(str, Enum):
    Segments = "segments"
    Vessels = "vessels"


class LengthMeasure(str, Enum):
    Splines = "spl"
    Skeleton = "skl"


class TortuosityMeasure(str, Enum):
    Distance = "dist"
    Curvature = "curv"
    Inflections = "infl"


class Tortuosity(LayerFeature):
    """Segment- or vessel-level tortuosity by distance ratio, curvature, or inflection counts.

    Representation: Uses segments or resolved_segments; for curvature uses Segment spline;
    optionally length-weighted. Operates on either individual segments or merged vessel trees.

    Computation: Measures vessel tortuosity using three methods: distance ratio (arc length / chord length),
    curvature (mean curvature along spline), or inflection counts (number of direction changes).
    Can be computed per-segment or per-vessel (resolved segments), with optional spline-length weighting.

    Args (constructor):
    - mode: `TortuosityMode` selecting per-segment or per-vessel computation.
    - measure: `TortuosityMeasure` (distance, curvature, inflections).
    - length_measure: length source (splines or skeleton) for distance-based measure.
    - min_numpoints: minimum skeleton points required for inclusion.
    - grid_field: optional `GridFieldEnum` restricting segments to a region.
    - aggregator: callable aggregator returning a single scalar over per-entity values.
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
        grid_field: Optional[BaseGridFieldSpecification] = None,
        aggregator: Callable = median,
        **kwargs,
    ):
        self.mode = mode
        self.measure = measure
        self.length_measure = length_measure
        self.min_numpoints = min_numpoints
        super().__init__(grid_field_spec=grid_field)
        self.aggregator = aggregator

    def _compute_for_segment(self, segment: Segment, scale: float):
        if self.measure == TortuosityMeasure.Distance:
            if self.length_measure == LengthMeasure.Splines:
                return (
                    segment.spline.length() / segment.chord_length
                    if segment.spline is not None
                    else np.nan
                )
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

    def _compute_weight_for_segment(self, segment: Segment) -> float:
        return segment.spline.length() if segment.spline is not None else np.nan

    def get_segments(self, layer: VesselTreeLayer):
        if self.mode == TortuosityMode.Segments:
            field = self._get_grid_field(layer)
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
        if isinstance(self.aggregator, LengthWeightedAggregator) and len(segments) < 5:
            return None

        vals = [
            self._compute_for_segment(vessel, scale=layer.retina.disc_fovea_distance)
            for vessel in segments
        ]

        if isinstance(self.aggregator, LengthWeightedAggregator):
            weights = [self._compute_weight_for_segment(seg) for seg in segments]
            return list(zip(weights, vals))

        return np.asarray(vals)

    def compute(self, layer: VesselTreeLayer):
        if self.grid_field_spec is not None:
            frac = grid_field_fraction_in_bounds(layer.retina, self.grid_field_spec)
            if frac < 0.5:
                return None
        tortuosities = self.raw(layer)
        if tortuosities is None:
            return None
        return self.aggregator(tortuosities)

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_aggregator_prefix, get_grid_field_suffix, get_layer_suffix

        agg = get_aggregator_prefix(self.aggregator)
        if isinstance(self.aggregator, LengthWeightedAggregator):
            agg = "Length-Weighted "

        field = get_grid_field_suffix(self.grid_field_spec)
        layer = get_layer_suffix(layer_name)
        measure = self.measure.name.title()
        return f"{agg}{measure} Tortuosity{field}{layer}"

    def name_prefix_tokens(self) -> list[str]:
        from .base import get_aggregator_tokens

        if isinstance(self.aggregator, LengthWeightedAggregator):
            return ["lw"]
        return get_aggregator_tokens(self.aggregator)

    def feature_name_tokens(self) -> list[str]:
        return ["tort"]

    def parameter_name_tokens(self) -> list[str]:
        tokens = [self.measure.value]
        if self.mode != TortuosityMode.Segments:
            tokens.append(self.mode.value)
        if self.length_measure != LengthMeasure.Splines:
            tokens.extend(["length", self.length_measure.value])
        if self.min_numpoints != 25:
            tokens.extend(["min_numpoints", str(self.min_numpoints)])
        return tokens

    def _plot(self, ax, layer: VesselTreeLayer, **kwargs):
        segments = self.get_segments(layer)

        vessels = Vessels(layer, segments)
        ax = vessels.plot(
            ax=ax,
            show_index=False,
            text=lambda x: (
                f"{100 * (self._compute_for_segment(x, scale=layer.retina.disc_fovea_distance) - 1):.2f}"
            ),
            cmap="tab20",
            min_numpoints=0,
            min_numpoints_caliber=self.min_numpoints,
            endpoints=True,
            chords=True,
            **kwargs,
        )

        # plot ETDRS region
        field = self._get_grid_field(layer)
        if field is not None:
            field.plot(ax)
        return ax
