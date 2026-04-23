from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from vascx.shared.aggregators import LengthWeightedAggregator, median
from vascx.shared.segment import Segment
from vascx.shared.vessels import Vessels

from .base import LayerFeature, grid_field_fraction_in_bounds

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


class Caliber(LayerFeature):
    """Aggregate vessel caliber within a region.

    Representation: Uses per-segment median_diameter computed on the undirected/directed segments
    of a VesselTreeLayer; segments are filtered by length (min_numpoints) and optionally by
    GridField visibility.

    Computation: Collects median_diameter over eligible segments and aggregates with the given
    aggregator (e.g., median or std). With `LengthWeightedAggregator`, values are weighted by
    spline arc length per segment (same convention as tortuosity). Segment diameters are computed
    from skeleton-derived measurements using spline interpolation and perpendicular distance sampling.

    Args (constructor):
    - min_numpoints: minimum number of skeleton points required for segment inclusion.
    - grid_field: optional `GridFieldEnum` restricting segments to a predefined retinal region.
    - aggregator: callable aggregator returning a single scalar from diameters, or
      `LengthWeightedAggregator` for length-weighted mean of per-segment median diameters.
    """

    def __init__(
        self,
        min_numpoints=10,
        grid_field: Optional[BaseGridFieldSpecification] = None,
        aggregator: Callable = median,
        spline_error_fraction: float = 0.05,
    ):
        self.min_numpoints = min_numpoints
        self.aggregator = aggregator
        self.spline_error_fraction = float(spline_error_fraction)
        super().__init__(grid_field_spec=grid_field)

    def _get_segments(self, layer: VesselTreeLayer):
        segments = layer.get_region_segments(self.grid_field_spec)
        segments = [seg for seg in segments if len(seg.skeleton) > self.min_numpoints]
        return segments

    def _compute_weight_for_segment(self, segment: Segment) -> float:
        spline = segment.get_spline(error_fraction=self.spline_error_fraction)
        return spline.length() if spline is not None else np.nan

    def compute(self, layer: VesselTreeLayer):
        # raise NotImplementedError("Caliber is not implemented")
        if self.grid_field_spec is not None:
            frac = grid_field_fraction_in_bounds(layer.retina, self.grid_field_spec)
            if frac < 0.5:
                return None
        segments = self._get_segments(layer)
        if isinstance(self.aggregator, LengthWeightedAggregator) and len(segments) < 5:
            return None

        calibers = [
            s.get_median_diameter(error_fraction=self.spline_error_fraction)
            for s in segments
        ]
        # median diameters should never be nan
        if np.isnan(calibers).any():
            raise ValueError("Some median diameters are nan")

        if isinstance(self.aggregator, LengthWeightedAggregator):
            weights = [self._compute_weight_for_segment(s) for s in segments]
            return self.aggregator(list(zip(weights, calibers)))

        return self.aggregator(np.asarray(calibers))

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_aggregator_prefix, get_grid_field_suffix, get_layer_suffix

        agg = get_aggregator_prefix(self.aggregator)
        if isinstance(self.aggregator, LengthWeightedAggregator):
            agg = "Length-Weighted "

        field = get_grid_field_suffix(self.grid_field_spec)
        layer = get_layer_suffix(layer_name)
        return f"{agg}Caliber{field}{layer}"

    def name_prefix_tokens(self) -> list[str]:
        from .base import get_aggregator_tokens

        if isinstance(self.aggregator, LengthWeightedAggregator):
            return ["lw"]
        return get_aggregator_tokens(self.aggregator)

    def feature_name_tokens(self) -> list[str]:
        return ["diam"]

    def parameter_name_tokens(self) -> list[str]:
        from .base import format_name_value

        tokens: list[str] = []
        if self.min_numpoints != 10:
            tokens.extend(["min_numpoints", str(self.min_numpoints)])
        if self.spline_error_fraction != 0.05:
            tokens.extend(
                [
                    "spline_error_fraction",
                    format_name_value(self.spline_error_fraction),
                ]
            )
        return tokens

    def _plot(self, ax, layer: VesselTreeLayer, **kwargs):
        segments = self._get_segments(layer)
        vessels = Vessels(layer, segments)
        ax = vessels.plot(
            ax=ax,
            # text=lambda x: f"{x.get_median_diameter(self.spline_error_fraction):.2f}",
            show_index=False,
            cmap="tab20",
            bounds=True,
            min_numpoints=0,
            min_numpoints_caliber=self.min_numpoints,
            arcs=True,
            spline_points=True,
            spline_error_fraction=self.spline_error_fraction,
            **kwargs,
        )

        # plot ETDRS region
        field = self._get_grid_field(layer)
        if field is not None:
            field.plot(ax)
        return ax
