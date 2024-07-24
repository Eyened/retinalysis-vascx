from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vascx.layer import VesselLayer

from .base import LayerFeature

if TYPE_CHECKING:
    pass


class Length(LayerFeature):
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

    def compute_with_spline(self, layer: VesselLayer):
        segments = [
            segment
            for segment in layer.segments
            if len(segment.skeleton) >= self.min_numpoints
        ]
        return np.mean([segment.spline.length() for segment in segments])

    def compute(self, layer: VesselLayer):
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
