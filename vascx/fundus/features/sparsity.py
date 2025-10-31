from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import warnings
import numpy as np
import scipy.ndimage as ndi
from enum import Enum



from .base import VesselsLayerFeature, grid_field_fraction_in_bounds

if TYPE_CHECKING:
    from rtnls_enface.grids.base import GridFieldEnum
    from vascx.fundus.layer import VesselTreeLayer
    from vascx.fundus.vessels_layer import FundusVesselsLayer


class SparsityMode(Enum):
    MEAN = "mean"
    MAX = "max"


class Sparsity(VesselsLayerFeature):
    """Sparsity over distance-to-nearest-vessel normalized by the OD–fovea distance.

    Representation: uses `FundusVesselsLayer.distance_transform` (DT), where each retinal pixel stores
    the distance to the nearest vessel pixel, normalized by the OD–fovea distance.

    Computation:
    - mode == "mean": S_mean = mean(DT over selected pixels)
    - mode == "max": S_max = max(DT over regional maxima within selected pixels)

    Args (constructor):
    - grid_field: optional `GridFieldEnum` restricting computation to a predefined retinal region.
    - mode: `SparsityMode` controlling aggregation ("mean" or "max").
    """
    
    def __init__(
        self, grid_field: 'GridFieldEnum' = None, mode: 'SparsityMode' = SparsityMode.MEAN
    ):
        """Coverage of distance transform, optionally restricted to an ETDRS grid field.

        When grid_field is provided, the computation is performed over pixels within the
        ETDRS field that are also inside the retina mask. If less than 50% of the field
        lies within the retina, a warning is issued and None is returned. Mode controls
        whether the mean is taken over all selected pixels ("mean") or only over local
        maxima of the distance transform ("max").
        """
        self.grid_field = grid_field
        self.mode = mode

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
            f"Sparsity(grid_field={fmt(self.grid_field)}, "
            f"mode={fmt(self.mode)})"
        )

    @staticmethod
    def _disk_footprint(radius: int = 2) -> np.ndarray:
        """Return a boolean disk footprint with given radius (5x5 when radius=2)."""
        y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        return (x * x + y * y) <= radius * radius

    @staticmethod
    def _local_maxima(dt: np.ndarray, radius: int = 4) -> np.ndarray:
        """Boolean mask of regional maxima using a disk footprint; NaN-safe."""
        # Import here to avoid hard dependency unless max mode is used
        import scipy.ndimage as ndi  # type: ignore

        dt_filled = np.where(np.isfinite(dt), dt, -np.inf)
        fp = Sparsity._disk_footprint(radius)
        max_img = ndi.maximum_filter(dt_filled, footprint=fp, mode="nearest")
        maxima = (dt_filled == max_img) & np.isfinite(dt)
        return maxima

    def _smooth_distance_transform(self, dt: np.ndarray) -> np.ndarray:
        return ndi.gaussian_filter(dt, sigma=1.0)

    def _compute_maxima_mask(self, layer: FundusVesselsLayer) -> np.ndarray:
        dt = layer.distance_transform
        dt_smoothed = self._smooth_distance_transform(dt)
        maxima_mask = Sparsity._local_maxima(dt_smoothed, radius=2)
        retina = layer.retina

        maxima_mask = maxima_mask & retina.mask

        if self.grid_field is not None:
            grid = retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
            field_mask = field.mask.astype(bool)
            maxima_mask = maxima_mask & field_mask

        return maxima_mask


    def compute(self, layer: FundusVesselsLayer):
        if self.grid_field is not None:
            frac = grid_field_fraction_in_bounds(layer.retina, self.grid_field)
            if frac < 0.5:
                return None
                
        maxima_mask = self._compute_maxima_mask(layer)
        vals = layer.distance_transform[maxima_mask]

        if self.mode == SparsityMode.MAX:
            return float(np.nanmax(vals)) if vals.size > 0 else np.nan
        else:
            return float(np.nanmean(vals)) if vals.size > 0 else np.nan

    def _plot(self, ax, layer: FundusVesselsLayer, **kwargs):
        layer.plot(ax=ax, image=True)
        dt = self._smooth_distance_transform(layer.distance_transform)
        maxima_mask = self._compute_maxima_mask(layer)

        if self.grid_field is not None:
            retina = layer.retina
            grid = retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
            field_mask = field.mask.astype(bool)
            to_plot = np.where(field_mask, dt, np.nan)
            ax.imshow(to_plot)
            # overlay ETDRS region
            retina.grids[self.grid_field.grid()].plot(ax, field)
            # overlay maxima points in max mode
            
        ys, xs = np.nonzero(maxima_mask)
        if ys.size > 0:
            ax.scatter(xs, ys, s=6, c="cyan", edgecolors="black", linewidths=0.2, alpha=0.8)

        return ax
    
    def plot_heatmap(self, layer: FundusVesselsLayer, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
        
        ax.imshow(layer.invisibility_map, cmap='jet', alpha=0.5, vmin=0, vmax=0.1)
        return ax
