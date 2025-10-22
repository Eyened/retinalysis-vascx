from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import warnings
import numpy as np



from .base import VesselsLayerFeature

if TYPE_CHECKING:
    from rtnls_enface.grids.base import GridFieldEnum
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
        self, ignore_fovea: bool = False, grid_field: 'GridFieldEnum' = None
    ):
        """Mean distance to nearest vessel, optionally restricted to an ETDRS grid field.

        When grid_field is provided, the computation is performed over pixels within the
        ETDRS field that are also inside the retina mask. If less than 80% of the field
        lies within the retina, a warning is issued and NaN is returned.
        """
        self.ignore_fovea = ignore_fovea
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
            f"Coverage(ignore_fovea={fmt(self.ignore_fovea)}, "
            f"grid_field={fmt(self.grid_field)})"
        )

    def compute(self, layer: FundusVesselsLayer):
        if self.grid_field is None:
            return layer.mean_distance_to_vessel

        retina = layer.retina
        grid = retina.grids[self.grid_field.grid()]
        field = grid.field(self.grid_field)
        field_mask = field.mask.astype(bool)
        # fraction of ETDRS field inside retina
        total = np.sum(field_mask)
        if total == 0:
            return np.nan
        in_retina = np.sum(field_mask & retina.mask)
        frac = in_retina / total
        if frac > 0.8:
            dt = layer.distance_transform
            vals = dt[field_mask & retina.mask]
            return float(np.nanmean(vals))
        else:
            warnings.warn(
                "ETDRS field largely outside retina for Coverage; returning NaN"
            )
            return np.nan

    def plot(self, ax, layer: FundusVesselsLayer, **kwargs):
        dt = layer.distance_transform
        if self.grid_field is not None:
            retina = layer.retina
            grid = retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
            field_mask = field.mask.astype(bool)
            to_plot = np.where(field_mask, dt, np.nan)
            ax.imshow(to_plot)
            # overlay ETDRS region
            retina.grids[self.grid_field.grid()].plot(ax, field)
            try:
                val = self.compute(layer)
                if val is not None and not np.isnan(val):
                    ax.text(5, 45, f'{val:.4f}', color='white', fontsize=6)
            except Exception:
                pass
        else:
            ax.imshow(dt)
            ax.text(5, 45, f'{layer.mean_distance_to_vessel:.4f}')

        return ax
    
    def plot_heatmap(self, layer: FundusVesselsLayer, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
        
        ax.imshow(layer.invisibility_map, cmap='jet', alpha=0.5, vmin=0, vmax=0.1)
        return ax
