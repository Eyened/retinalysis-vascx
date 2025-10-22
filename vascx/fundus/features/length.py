from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import LayerFeature

if TYPE_CHECKING:
    from rtnls_enface.grids.base import GridFieldEnum
    from vascx.fundus.layer import VesselTreeLayer


class Length(LayerFeature):
    """Mean segment length (skeleton by default; spline alternative available).

    Representation: Uses segments with skeleton or spline-based length measurements from the 
    VesselTreeLayer directed graph representation.

    Computation: Computes mean length across all segments that meet the minimum length threshold. 
    Length can be measured either along the skeleton centerline points or via fitted spline curves 
    for smoother length estimation.

    Options: min_numpoints (minimum segment length filter for inclusion in computation).
    """
    
    # Ideas
    # tortuosity for different levels of caliber
    #   what happens when small vessels not visible
    # tortuosity for different generations
    def __init__(
        self,
        min_numpoints: int = 25,
        grid_field: 'GridFieldEnum' = None,
        **kwargs,
    ):
        """Mean segment length within optional ETDRS grid_field using skeleton lengths.

        If grid_field is provided, segments are filtered to those sufficiently inside the field
        via `layer.filter_segments(field=self.grid_field)` prior to aggregation.
        """
        self.min_numpoints = min_numpoints
        self.grid_field = grid_field

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
            f"Length(min_numpoints={fmt(self.min_numpoints)}, "
            f"grid_field={fmt(self.grid_field)})"
        )

    def compute_with_spline(self, layer: VesselTreeLayer):
        field = None
        if self.grid_field is not None:
            grid = layer.retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
        segments = [
            segment
            for segment in layer.filter_segments(field=field)
            if len(segment.skeleton) >= self.min_numpoints
        ]
        return np.mean([segment.spline.length() for segment in segments])

    def compute(self, layer: VesselTreeLayer):
        field = None
        if self.grid_field is not None:
            grid = layer.retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
        segments = [
            segment
            for segment in layer.filter_segments(field=field)
            if len(segment.skeleton) >= self.min_numpoints
        ]
        return np.mean([segment.length for segment in segments])

    def calc_auxiliary(self, parameters):
        pass

    def plot(self, ax, layer: 'VesselTreeLayer', **kwargs):
        """Plot only the segments used in computation and overlay ETDRS field if set."""
        field = None
        if self.grid_field is not None:
            grid = layer.retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
        segments = [
            segment
            for segment in layer.filter_segments(field=field)
            if len(segment.skeleton) >= self.min_numpoints
        ]

        ax = layer.plot_segments(ax=ax, segments=segments)

        # annotate mean length to match compute()
        if len(segments) > 0:
            mean_len = float(np.mean([s.length for s in segments]))
            ax.text(5, 15, f"mean={mean_len:.2f}", color="white", fontsize=6)

        # plot ETDRS region
        if self.grid_field is not None:
            grid = layer.retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
            grid.plot(ax, field)
        return ax
