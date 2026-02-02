from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy.ndimage as ndi
from matplotlib import pyplot as plt
from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from vascx.shared.masks import fill_small_holes

from .base import VesselsLayerFeature, grid_field_fraction_in_bounds

if TYPE_CHECKING:
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
    - mode == "max": S_max = max(DT measured at one EDT peak per vessel-free connected component
      (blob) inside the selected pixels)

    Args (constructor):
    - grid_field: optional `GridFieldEnum` restricting computation to a predefined retinal region.
    - mode: `SparsityMode` controlling aggregation ("mean" or "max").
    """

    def __init__(
        self,
        grid_field: Optional[BaseGridFieldSpecification] = None,
        mode: "SparsityMode" = SparsityMode.MEAN,
        maxima_smoothing_sigma: float = 1.0,
    ):
        """Coverage of distance transform, optionally restricted to an ETDRS grid field.

        When grid_field is provided, the computation is performed over pixels within the
        ETDRS field that are also inside the retina mask. If less than 50% of the field
        lies within the retina, a warning is issued and None is returned. Mode controls
        whether the mean is taken over all selected pixels ("mean") or only over blob-wise
        maxima of the distance transform ("max"). `maxima_smoothing_sigma` controls the
        Gaussian smoothing applied before peak finding
        """
        super().__init__(grid_field_spec=grid_field)
        self.mode = mode
        self.maxima_smoothing_sigma = maxima_smoothing_sigma

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

        return f"Sparsity(grid_field_spec={fmt(self.grid_field_spec)}, mode={fmt(self.mode)})"

    def _smooth_distance_transform(self, dt: np.ndarray) -> np.ndarray:
        return ndi.gaussian_filter(dt, sigma=self.maxima_smoothing_sigma)

    def _get_field_mask(self, layer: FundusVesselsLayer) -> Optional[np.ndarray]:
        field = self._get_grid_field(layer)
        if field is None:
            return None
        return field.mask.astype(bool)

    def _compute_maxima_mask(self, layer: FundusVesselsLayer) -> np.ndarray:
        dt = layer.distance_transform
        retina = layer.retina

        retina_mask = retina.mask.astype(bool)
        disc_mask = (
            retina.disc.mask.astype(bool)
            if retina.disc is not None
            else np.zeros_like(retina_mask, dtype=bool)
        )
        roi_mask = retina_mask & ~disc_mask

        vessels_mask = layer.binary.astype(bool)
        background_inverted = ~((~vessels_mask) & roi_mask)
        background = ~fill_small_holes(background_inverted, area_threshold=250)

        if not np.any(background):
            return np.zeros_like(background, dtype=bool)

        labels, num_labels = ndi.label(background)
        maxima_mask = np.zeros_like(background, dtype=bool)

        for label_id in range(1, num_labels + 1):
            region_coords = np.argwhere(labels == label_id)
            if region_coords.size == 0:
                continue
            region_vals = dt[region_coords[:, 0], region_coords[:, 1]]
            if np.all(np.isnan(region_vals)):
                continue

            peak_idx = int(np.nanargmax(region_vals))
            y, x = region_coords[peak_idx]
            maxima_mask[y, x] = True

        if self.grid_field_spec is not None:
            field_mask = self._get_field_mask(layer)
            return maxima_mask & field_mask

        return maxima_mask

    def get_fovea_mask(self, layer: FundusVesselsLayer) -> np.ndarray:
        """Compute a circular binary mask centered at the fovea with radius relative to the distance between fovea and optic disc."""
        retina = layer.retina
        h, w = retina.resolution
        fovea = retina.fovea_location
        radius = retina.disc_fovea_distance / 6

        yy = np.arange(h)[:, None]
        xx = np.arange(w)[None, :]
        mask = (xx - fovea.x) ** 2 + (yy - fovea.y) ** 2 <= radius**2

        return mask.astype(bool)

    def compute(self, layer: FundusVesselsLayer):
        if self.grid_field_spec is not None:
            frac = grid_field_fraction_in_bounds(layer.retina, self.grid_field_spec)
            if frac < 0.5:
                return None

        fovea_mask = self.get_fovea_mask(layer)
        if self.mode == SparsityMode.MAX:
            maxima_mask = self._compute_maxima_mask(layer)
            maxima_mask = maxima_mask & ~fovea_mask
            vals = layer.distance_transform[maxima_mask]
            return float(np.nanmax(vals)) if vals.size > 0 else np.nan
        else:
            dt = layer.distance_transform
            dt = np.where(~fovea_mask, dt, np.nan)
            field_mask = self._get_field_mask(layer)
            if field_mask is not None:
                dt = np.where(field_mask, dt, np.nan)

            return float(np.nanmean(dt))

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_grid_field_suffix, get_layer_suffix

        field = get_grid_field_suffix(self.grid_field_spec)
        layer = get_layer_suffix(layer_name)
        # Capitalize only first letter of Enum name
        mode = self.mode.name.title()
        return f"{mode} Sparsity{field}{layer}"

    def _plot(self, ax, layer: FundusVesselsLayer, **kwargs):
        layer.plot(ax=ax, image=True)
        dt = layer.distance_transform
        fovea_mask = self.get_fovea_mask(layer)
        dt = np.where(~fovea_mask, dt, np.nan)
        field_mask = self._get_field_mask(layer)

        if field_mask is not None:
            ax.imshow(np.where(field_mask, dt, np.nan))
            field = self._get_grid_field(layer)
            if field is not None:
                field.plot(ax)
        else:
            ax.imshow(dt)

        # only show points in max mode
        if self.mode == SparsityMode.MAX:
            maxima_mask = self._compute_maxima_mask(layer)
            maxima_mask = maxima_mask & ~fovea_mask
            ys, xs = np.nonzero(maxima_mask)
            if ys.size > 0:
                ax.scatter(
                    xs, ys, s=6, c="cyan", edgecolors="black", linewidths=0.2, alpha=0.8
                )

        return ax

    def plot_heatmap(self, layer: FundusVesselsLayer, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)

        ax.imshow(layer.invisibility_map, cmap="jet", alpha=0.5, vmin=0, vmax=0.1)
        return ax
