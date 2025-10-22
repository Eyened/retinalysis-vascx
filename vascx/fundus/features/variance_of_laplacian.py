from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import numpy as np

from .base import RetinaFeature

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina
    from rtnls_enface.grids.base import GridFieldEnum
    from vascx.fundus.layer import VesselTreeLayer
    from vascx.fundus.vessels_layer import FundusVesselsLayer


class VarianceOfLaplacian(RetinaFeature):
    """Global image sharpness proxy; variance of Laplacian map.

    Representation: Uses Retina.laplacian - global image Laplacian operator applied to the 
    fundus image to detect edges and texture variations.

    Computation: Applies the Laplacian operator (second derivative) to the retinal image to 
    highlight regions of rapid intensity change, then computes the variance of the resulting 
    Laplacian map. Higher variance indicates sharper, more detailed images with better focus.

    Options: grid_field (optional ETDRS region limiting computation/visualization).
    """
    
    def __init__(self, grid_field: 'GridFieldEnum' = None):
        """Variance of Laplacian, optionally restricted to an ETDRS grid_field.

        When grid_field is provided, the variance is computed over the Laplacian
        values inside the ETDRS field intersected with the retinal mask.
        """
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
        return f"VarianceOfLaplacian(grid_field={fmt(self.grid_field)})"

    def compute(self, retina: 'Retina'):
        if self.grid_field is None:
            return float(np.nanvar(retina.laplacian))

        grid = retina.grids[self.grid_field.grid()]
        field = grid.field(self.grid_field)
        field_mask = field.mask.astype(bool)
        mask = field_mask & retina.mask
        if not np.any(mask):
            return float('nan')
        vals = np.where(mask, retina.laplacian, np.nan)
        return float(np.nanvar(vals))

    def plot(self, ax, retina: 'Retina', **kwargs):
        if self.grid_field is None:
            ax.imshow(retina.laplacian)
            ax.text(5, 45, f"{self.compute(retina):.4f}")
        else:
            grid = retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
            field_mask = field.mask.astype(bool)
            masked = np.where(field_mask, retina.laplacian, np.nan)
            ax.imshow(masked)
            # overlay ETDRS region
            grid.plot(ax, field)
            try:
                val = self.compute(retina)
                if val is not None and not np.isnan(val):
                    ax.text(5, 45, f"{val:.4f}", color='white', fontsize=6)
            except Exception:
                pass

        return ax
    
