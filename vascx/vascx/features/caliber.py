from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vascx.analysis.aggregators import mean_median_std

from .base import LayerFeature
from .base import LayerFeature

if TYPE_CHECKING:
    from vascx.retina import VesselLayer


class Caliber(LayerFeature):
    def __init__(self, min_numpoints=10, aggregator=mean_median_std):
        self.min_numpoints = min_numpoints
        self.aggregator = aggregator

    def compute(self, layer: VesselLayer):
        segments = layer.vessels.filter_segments_by_numpoints(self.min_numpoints)
        calibers = [s.median_diameter for s in segments]
        # median diameters should never be nan
        if np.isnan(calibers).any():
            raise ValueError("Some median diameters are nan")
        return self.aggregator(calibers)

    def plot(self, layer: VesselLayer, **kwargs):
        return layer.vessels.plot(
            text=lambda x: f"{x.median_diameter:.2f}",
            cmap="tab20",
            min_numpoints=0,
            min_numpoints_caliber=self.min_numpoints,
            plot_spline_points=True,
            **kwargs,
        )
