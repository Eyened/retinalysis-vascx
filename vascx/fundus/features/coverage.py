from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import warnings
import numpy as np
from enum import Enum



from .base import VesselsLayerFeature, grid_field_fraction_in_bounds

if TYPE_CHECKING:
    from rtnls_enface.grids.base import GridFieldEnum
    from vascx.fundus.layer import VesselTreeLayer
    from vascx.fundus.vessels_layer import FundusVesselsLayer


class CoverageMode(Enum):
    MEAN = "mean"
    MAX = "max"


class Coverage(VesselsLayerFeature):
    """Coverage over distance transform to nearest vessel pixel normalized by OD–fovea distance.

    Representation: Uses FundusVesselsLayer.distance_transform which computes the distance from each
    retinal pixel to the nearest vessel pixel, normalized by the OD-fovea distance.

    Computation:
    - mode == "mean": mean of the distance transform across selected pixels.
    - mode == "max": mean of regional maxima of the distance transform across selected pixels.

    Options: ignore_fovea (reserved); grid_field to restrict to ETDRS field; mode selects behavior.
    """
    
    def __init__(
        self, ignore_fovea: bool = False, grid_field: 'GridFieldEnum' = None, mode: 'CoverageMode' = CoverageMode.MEAN
    ):
        """Coverage of distance transform, optionally restricted to an ETDRS grid field.

        When grid_field is provided, the computation is performed over pixels within the
        ETDRS field that are also inside the retina mask. If less than 50% of the field
        lies within the retina, a warning is issued and None is returned. Mode controls
        whether the mean is taken over all selected pixels ("mean") or only over local
        maxima of the distance transform ("max").
        """
        self.ignore_fovea = ignore_fovea
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
            f"Coverage(ignore_fovea={fmt(self.ignore_fovea)}, "
            f"grid_field={fmt(self.grid_field)}, "
            f"mode={fmt(self.mode)})"
        )

    @staticmethod
    def _disk_footprint(radius: int = 2) -> np.ndarray:
        """Return a boolean disk footprint with given radius (5x5 when radius=2)."""
        y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        return (x * x + y * y) <= radius * radius

    @staticmethod
    def _local_maxima(dt: np.ndarray, radius: int = 2) -> np.ndarray:
        """Boolean mask of regional maxima using a disk footprint; NaN-safe."""
        # Import here to avoid hard dependency unless max mode is used
        import scipy.ndimage as ndi  # type: ignore

        dt_filled = np.where(np.isfinite(dt), dt, -np.inf)
        fp = Coverage._disk_footprint(radius)
        max_img = ndi.maximum_filter(dt_filled, footprint=fp, mode="nearest")
        maxima = (dt_filled == max_img) & np.isfinite(dt)
        return maxima

    def compute(self, layer: FundusVesselsLayer):
        if self.grid_field is not None:
            frac = grid_field_fraction_in_bounds(layer.retina, self.grid_field)
            if frac < 0.5:
                return None
                
        dt = layer.distance_transform
        # Mean mode preserves existing behavior
        if self.mode == CoverageMode.MEAN:
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
            if frac > 0.5:
                vals = dt[field_mask & retina.mask]
                return float(np.nanmean(vals))
            else:
                warnings.warn(
                    "ETDRS field largely outside retina for Coverage; returning None"
                )
                return None

        # Max mode: compute mean of regional maxima within selection
        maxima_mask = Coverage._local_maxima(dt, radius=2)
        retina = layer.retina
        if self.grid_field is None:
            sel = maxima_mask & retina.mask
            vals = dt[sel]
            return float(np.nanmean(vals)) if vals.size > 0 else np.nan

        grid = retina.grids[self.grid_field.grid()]
        field = grid.field(self.grid_field)
        field_mask = field.mask.astype(bool)
        total = np.sum(field_mask)
        if total == 0:
            return np.nan
        in_retina = np.sum(field_mask & retina.mask)
        frac = in_retina / total
        if frac > 0.5:
            sel = maxima_mask & field_mask & retina.mask
            vals = dt[sel]
            return float(np.nanmean(vals)) if vals.size > 0 else np.nan
        else:
            warnings.warn(
                "ETDRS field largely outside retina for Coverage; returning None"
            )
            return None

    def plot(self, ax, layer: FundusVesselsLayer, **kwargs):
        layer.plot(ax=ax, image=True)
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
            # overlay maxima points in max mode
            if self.mode == CoverageMode.MAX:
                maxima_mask = Coverage._local_maxima(dt, radius=2)
                sel = maxima_mask & field_mask & retina.mask
                ys, xs = np.nonzero(sel)
                if ys.size > 0:
                    ax.scatter(xs, ys, s=6, c="cyan", edgecolors="black", linewidths=0.2, alpha=0.8)
            try:
                val = self.compute(layer)
                if val is not None and not np.isnan(val):
                    ax.text(5, 45, f'{val:.4f}', color='white', fontsize=6)
            except Exception:
                pass
        else:
            ax.imshow(dt)
            # overlay maxima points in max mode
            if self.mode == CoverageMode.MAX:
                retina = layer.retina
                maxima_mask = Coverage._local_maxima(dt, radius=2)
                sel = maxima_mask & retina.mask
                ys, xs = np.nonzero(sel)
                if ys.size > 0:
                    ax.scatter(xs, ys, s=6, c="cyan", edgecolors="black", linewidths=0.2, alpha=0.8)
            try:
                val = self.compute(layer)
                if val is not None and not np.isnan(val):
                    ax.text(5, 45, f'{val:.4f}')
            except Exception:
                pass

        return ax
    
    def plot_heatmap(self, layer: FundusVesselsLayer, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
        
        ax.imshow(layer.invisibility_map, cmap='jet', alpha=0.5, vmin=0, vmax=0.1)
        return ax
