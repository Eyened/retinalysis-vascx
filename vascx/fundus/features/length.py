from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import LayerFeature

if TYPE_CHECKING:
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
        **kwargs,
    ):
        self.min_numpoints = min_numpoints

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
        return f"Length(min_numpoints={fmt(self.min_numpoints)})"

    def compute_with_spline(self, layer: VesselTreeLayer):
        segments = [
            segment
            for segment in layer.segments
            if len(segment.skeleton) >= self.min_numpoints
        ]
        return np.mean([segment.spline.length() for segment in segments])

    def compute(self, layer: VesselTreeLayer):
        segments = [
            segment
            for segment in layer.segments
            if len(segment.skeleton) >= self.min_numpoints
        ]
        return np.mean([segment.length for segment in segments])

    def calc_auxiliary(self, parameters):
        pass

    def plot(self, layer, **kwargs):
        layer.plot_segments()
