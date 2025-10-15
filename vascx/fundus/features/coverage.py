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
    """Mean of distance transform to nearest vessel pixel normalized by OD–fovea distance.

    Representation: Uses FundusVesselsLayer.distance_transform which computes the distance from each 
    retinal pixel to the nearest vessel pixel, normalized by the OD-fovea distance.

    Computation: Calculates a distance transform that measures, for each pixel in the retina, the 
    distance to its nearest vessel pixel. The distances are normalized by the OD-fovea distance to 
    provide scale-invariant measurements. Returns the mean distance across all retinal pixels, 
    providing a measure of vessel coverage density.

    Options: ignore_fovea (reserved for future use to exclude foveal region from computation).
    """
    
    def __init__(
        self, ignore_fovea=False
    ):
        self.ignore_fovea = ignore_fovea

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
        return f"Coverage(ignore_fovea={fmt(self.ignore_fovea)})"

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
