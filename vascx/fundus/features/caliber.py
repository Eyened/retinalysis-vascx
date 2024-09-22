from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vascx.shared.aggregators import mean_median_std
from vascx.shared.vessels import Vessels

from .base import LayerFeature

if TYPE_CHECKING:
    from rtnls_enface.grids.base import GridField
    from vascx.fundus.layer import VesselTreeLayer


class Caliber(LayerFeature):
    def __init__(
        self, min_numpoints=10, grid_field: GridField = None, aggregator=mean_median_std
    ):
        self.min_numpoints = min_numpoints
        self.aggregator = aggregator
        self.grid_field = grid_field

    def _get_segments(self, layer: VesselTreeLayer):
        segments = layer.filter_segments(field=self.grid_field)
        return [seg for seg in segments if len(seg.skeleton) > self.min_numpoints]

    def compute(self, layer: VesselTreeLayer):
        segments = self._get_segments(layer)
        calibers = [s.median_diameter for s in segments]
        # median diameters should never be nan
        if np.isnan(calibers).any():
            raise ValueError("Some median diameters are nan")
        return self.aggregator(calibers)

    def plot(self, layer: VesselTreeLayer, **kwargs):
        vessels = Vessels(layer, self._get_segments(layer))
        ax = vessels.plot(
            text=lambda x: f"{x.median_diameter:.2f}",
            cmap="tab20",
            min_numpoints=0,
            min_numpoints_caliber=self.min_numpoints,
            plot_spline_points=True,
            **kwargs,
        )

        # plot ETDRS region
        if self.grid_field is not None:
            layer.retina.grids[self.grid_field.grid()].plot(ax, self.grid_field)
        return ax
