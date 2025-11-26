from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from vascx.shared.aggregators import mean_median_std
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
    aggregator (e.g., median/std). Segment diameters are computed from skeleton-derived measurements
    using spline interpolation and perpendicular distance sampling.

    Args (constructor):
    - min_numpoints: minimum number of skeleton points required for segment inclusion.
    - grid_field: optional `GridFieldEnum` restricting segments to a predefined retinal region.
    - aggregator: function that aggregates the set of diameters (e.g., mean/median/std tuple).
    """

    def __init__(
        self,
        min_numpoints=10,
        grid_field: Optional[BaseGridFieldSpecification] = None,
        aggregator=mean_median_std,
    ):
        self.min_numpoints = min_numpoints
        self.aggregator = aggregator
        super().__init__(grid_field_spec=grid_field)

    def __repr__(self) -> str:
        def fmt(v):
            from enum import Enum

            import numpy as np

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
            f"Caliber(min_numpoints={fmt(self.min_numpoints)}, "
            f"grid_field_spec={fmt(self.grid_field_spec)}, "
            f"aggregator={fmt(self.aggregator)})"
        )

    def _get_segments(self, layer: VesselTreeLayer):
        field = self._get_grid_field(layer)
        segments = layer.filter_segments(field=field)
        segments = [seg for seg in segments if len(seg.skeleton) > self.min_numpoints]
        return segments

    def compute(self, layer: VesselTreeLayer):
        if self.grid_field_spec is not None:
            frac = grid_field_fraction_in_bounds(layer.retina, self.grid_field_spec)
            if frac < 0.5:
                return None
        segments = self._get_segments(layer)
        calibers = [s.median_diameter for s in segments]
        # median diameters should never be nan
        if np.isnan(calibers).any():
            raise ValueError("Some median diameters are nan")
        return self.aggregator(calibers)

    def _plot(self, ax, layer: VesselTreeLayer, **kwargs):
        segments = self._get_segments(layer)
        vessels = Vessels(layer, segments)
        ax = vessels.plot(
            ax=ax,
            text=lambda x: f"{x.median_diameter:.2f}",
            show_index=False,
            cmap="tab20",
            min_numpoints=0,
            min_numpoints_caliber=self.min_numpoints,
            plot_spline_points=True,
            **kwargs,
        )

        # plot ETDRS region
        field = self._get_grid_field(layer)
        if field is not None:
            field.plot(ax)
        return ax
