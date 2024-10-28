from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from rtnls_enface.grids.base import GridField
from vascx.faz.layer import FAZLayer

from .base import FAZLayerFeature

if TYPE_CHECKING:
    pass


@dataclass
class VascularDensity(FAZLayerFeature):
    def __init__(self, grid_field: GridField = None, cut_mask: bool = False):
        self.grid_field = grid_field
        self.cut_mask = cut_mask

    def get_mask(self, layer: FAZLayer):
        if self.grid_field is None:
            return np.ones(layer.retina.resolution, dtype=np.uint8) * 255
        return layer.retina.grids[self.grid_field.grid()].field(self.grid_field).astype(np.uint8) * 255
        
    def plot(self, ax, layer: FAZLayer, **kwargs):
        ax = layer.retina.plot(ax=ax, layers=[])
        mask = self.get_mask(layer)
        density = self.compute(layer)

        binary = layer.binary.astype(np.uint8)
        selected_pixels = cv2.bitwise_and(binary, binary, mask=mask)
        colors = [(0, 0, 0, 0), (0, 0, 1, 1)]
        cmap = LinearSegmentedColormap.from_list("binary", colors, N=2)
        ax.imshow(selected_pixels, cmap=cmap)

        # plot ETDRS region
        if self.grid_field is not None:
            layer.retina.grids[self.grid_field.grid()].plot(ax, self.grid_field)

        ax.text(10, 30, f"{density:.3f}", color="white", fontsize=6)

    def compute(self, layer: FAZLayer):
        mask = self.get_mask(layer)

        # Use the mask to select pixels within the region in the image
        binary = layer.binary.astype(np.uint8)
        selected_pixels = cv2.bitwise_and(binary, binary, mask=mask)

        return np.sum(selected_pixels[mask == 255]) / np.sum(mask == 255)
