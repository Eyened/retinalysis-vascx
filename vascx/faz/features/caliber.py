from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vascx.faz.features.base import FAZLayerFeature
from vascx.shared.aggregators import median
from vascx.shared.vessels import Vessels

if TYPE_CHECKING:
    from vascx.faz.layer import FAZLayer


class Caliber(FAZLayerFeature):
    def __init__(self, min_numpoints=10, aggregator=median):
        self.min_numpoints = min_numpoints
        self.aggregator = aggregator

    def _get_segments(self, layer: FAZLayer):
        return layer.segments

    def compute(self, layer: FAZLayer):
        segments = self._get_segments(layer)
        calibers = [s.median_diameter for s in segments]
        # median diameters should never be nan
        if np.isnan(calibers).any():
            raise ValueError("Some median diameters are nan")
        return self.aggregator(calibers)

    def _plot(self, ax, layer: FAZLayer, **kwargs):
        vessels = Vessels(layer, self._get_segments(layer))
        return vessels.plot(
            ax=ax,
            text=lambda x: f"{x.median_diameter:.2f}",
            cmap="tab20",
            min_numpoints=0,
            min_numpoints_caliber=self.min_numpoints,
            plot_spline_points=True,
            **kwargs,
        )
