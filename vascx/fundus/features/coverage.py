from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import numpy as np



from .base import VesselsLayerFeature

if TYPE_CHECKING:
    from rtnls_enface.grids.base import GridField
    from vascx.fundus.layer import VesselTreeLayer
    from vascx.fundus.vessels_layer import FundusVesselsLayer


class Coverage(VesselsLayerFeature):
    ''' Quantifies vessel coverage by calculating a distance transform that computes, for each pixel, the distance to its nearest white or on pixel on the vessel mask.
    '''
    def __init__(
        self, ignore_fovea=False
    ):
        self.ignore_fovea = ignore_fovea


    def compute(self, layer: FundusVesselsLayer):
        return layer.mean_distance_to_vessel

    def plot(self, layer: FundusVesselsLayer, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)

        ax.imshow(layer.distance_transform)
        ax.text(5, 45, f'{layer.mean_distance_to_vessel:.4f}')

        return ax
    
    def plot_heatmap(self, layer: FundusVesselsLayer, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
        
        ax.imshow(layer.invisibility_map, cmap='jet', alpha=0.5, vmin=0, vmax=0.1)
        return ax
