from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from rtnls_enface.grids.specifications import BaseGridFieldSpecification
from skimage.exposure import equalize_adapthist

from .base import RetinaFeature, grid_field_masks_and_fraction

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina


class VarianceOfLaplacian(RetinaFeature):
    """Global image sharpness proxy; variance of Laplacian map.

    Representation: Uses Retina.laplacian - global image Laplacian operator applied to the
    fundus image to detect edges and texture variations.

    Computation: Applies the Laplacian operator (second derivative) to the retinal image to
    highlight regions of rapid intensity change, then computes the variance of the resulting
    Laplacian map. Higher variance indicates sharper, more detailed images with better focus.

    Args (constructor):
    - grid_field: optional `GridFieldEnum` limiting computation/visualization to a predefined region
      (applied within the retina mask).
    """

    def __init__(self, grid_field: Optional[BaseGridFieldSpecification] = None):
        """Variance of Laplacian, optionally restricted to an ETDRS grid_field.

        When grid_field is provided, the variance is computed over the Laplacian
        values inside the ETDRS field intersected with the retinal mask.
        """
        super().__init__(grid_field_spec=grid_field)

    def __repr__(self) -> str:
        def fmt(v):
            from enum import Enum

            import numpy as np

            if v is None:
                return "None"
            if isinstance(v, Enum):
                return f"{v.__class__.__name__}.{v.name}"
            if callable(v):
                return getattr(v, "__name__", v.__class__.__name__)
            if isinstance(v, np.generic):
                return repr(v.item())
            return repr(v)

        return f"VarianceOfLaplacian(grid_field_spec={fmt(self.grid_field_spec)})"

    def compute(self, retina: "Retina"):
        if self.grid_field_spec is None:
            return float(np.nanvar(retina.laplacian))

        field_mask, in_bounds_mask, frac = grid_field_masks_and_fraction(
            retina, self.grid_field_spec
        )
        if frac < 0.5:
            return None
        mask = in_bounds_mask
        if not np.any(mask):
            return None
        vals = np.where(mask, retina.laplacian, np.nan)
        return float(np.nanvar(vals))

    def display_name(self, key: str = None, **kwargs) -> str:
        from .base import get_grid_field_suffix

        field = get_grid_field_suffix(self.grid_field_spec)
        return f"Laplacian Variance{field} - IM"

    def _plot(self, ax, retina: "Retina", **kwargs):
        L = retina.laplacian.astype(np.float32)  # may contain NaNs outside mask
        mask = np.isfinite(L)

        M = np.abs(L)  # use magnitude for edge strength
        low, high = np.nanpercentile(M[mask], (1, 99))  # robust, ignore NaNs
        scale = max(high - low, 1e-6)
        M = np.clip((M - low) / scale, 0, 1)  # normalize to [0,1]
        M[~mask] = 0.0  # fill NaNs

        M_eq = equalize_adapthist(M, clip_limit=0.02, nbins=256)
        ax.imshow(M_eq, cmap="gray", vmin=0, vmax=1)

        field = self._get_grid_field(retina)
        if field is not None:
            field.plot(ax)

        return ax
