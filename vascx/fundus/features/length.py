from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from .base import LayerFeature, grid_field_fraction_in_bounds

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


class Length(LayerFeature):
    """Mean segment length (skeleton by default; spline alternative available).

    Representation: Uses segments with skeleton or spline-based length measurements from the
    VesselTreeLayer directed graph representation.

    Computation: Computes mean length across all segments that meet the minimum length threshold.
    Length can be measured either along the skeleton centerline points or via fitted spline curves
    for smoother length estimation.

    Args (constructor):
    - min_numpoints: minimum number of skeleton points required for a segment to be included.
    - grid_field: optional `GridFieldEnum` restricting segments to a predefined retinal region.

    Notes: `compute` uses skeleton lengths; `compute_with_spline` provides a spline-based alternative.
    """

    # Ideas
    # tortuosity for different levels of caliber
    #   what happens when small vessels not visible
    # tortuosity for different generations
    def __init__(
        self,
        min_numpoints: int = 25,
        grid_field: Optional[BaseGridFieldSpecification] = None,
        **kwargs,
    ):
        """Mean segment length within optional ETDRS grid_field using skeleton lengths.

        If grid_field is provided, segments are filtered to those sufficiently inside the field
        via `layer.filter_segments(field=self.grid_field)` prior to aggregation.
        """
        self.min_numpoints = min_numpoints
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
            f"Length(min_numpoints={fmt(self.min_numpoints)}, "
            f"grid_field_spec={fmt(self.grid_field_spec)})"
        )

    def compute_with_spline(self, layer: VesselTreeLayer):
        field = None
        if self.grid_field_spec is not None:
            frac = grid_field_fraction_in_bounds(layer.retina, self.grid_field_spec)
            if frac < 0.5:
                return None
            field = self._get_grid_field(layer)
        segments = [
            segment
            for segment in layer.filter_segments(field=field)
            if len(segment.skeleton) >= self.min_numpoints
        ]
        return np.mean([segment.spline.length() for segment in segments])

    def compute(self, layer: VesselTreeLayer):
        field = None
        if self.grid_field_spec is not None:
            frac = grid_field_fraction_in_bounds(layer.retina, self.grid_field_spec)
            if frac < 0.5:
                return None
            field = self._get_grid_field(layer)
        segments = [
            segment
            for segment in layer.filter_segments(field=field)
            if len(segment.skeleton) >= self.min_numpoints
        ]
        return np.mean([segment.length for segment in segments])

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_grid_field_suffix, get_layer_suffix

        field = get_grid_field_suffix(self.grid_field_spec)
        layer = get_layer_suffix(layer_name)
        return f"Mean Segment Length{field}{layer}"

    def calc_auxiliary(self, parameters):
        pass

    def _plot(self, ax, layer: "VesselTreeLayer", **kwargs):
        """Draw segments used in computation and overlay ETDRS field if set."""
        field = self._get_grid_field(layer)
        segments = [
            segment
            for segment in layer.filter_segments(field=field)
            if len(segment.skeleton) >= self.min_numpoints
        ]

        ax = layer.plot_segments(ax=ax, segments=segments)

        # plot ETDRS region
        if field is not None:
            field.plot(ax)
        return ax
