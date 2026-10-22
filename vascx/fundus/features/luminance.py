from __future__ import annotations

from vascx.shared.outputs import FeatureOutput

from typing import TYPE_CHECKING, Optional

import numpy as np
from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from .base import (
    RetinaFeature,
    get_grid_field_suffix,
    grid_field_masks_and_fraction,
    resolve_min_area_within_bounds,
    validate_min_area_within_bounds,
)

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina

# Rec. 601 luma coefficients (same as quality-control luminance score).
_LUMA_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Convert an RGB or grayscale image to Rec. 601 luminance."""
    img = np.asarray(image)
    if img.ndim == 2:
        return img.astype(np.float64)
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0].astype(np.float64)
    if img.ndim == 3 and img.shape[2] >= 3:
        rgb = img[:, :, :3].astype(np.float64)
        return rgb @ _LUMA_WEIGHTS
    raise ValueError(f"Unsupported image shape: {img.shape}")


class Luminance(RetinaFeature):
    """Mean Rec. 601 luminance inside the visible fundus mask.

    Representation: Uses the fundus RGB (or grayscale) image and the retina FOV mask.

    Computation: Converts the image to Rec. 601 luminance and returns the mean over
    visible mask pixels, optionally restricted to a grid field. Higher values
    indicate a brighter image.

    Grid visibility uses ``min_area_within_bounds`` from the feature, field, or
    grid, in that order, falling back to 0.80. Insufficient visibility or an
    empty measurement region produces a missing value.
    """

    general_description = (
        "Fundus luminance describes the overall brightness of the retinal image. "
        "Higher values indicate a brighter image."
    )

    outputs = (FeatureOutput(units="intensity"),)

    def implementation_description(self, **kwargs) -> str:
        return "Calculated by converting the fundus image to luminance."

    def aggregation_description(self, **kwargs) -> str:
        if self.grid_field_spec is not None:
            return "Reported as the mean across the visible retinal area within the selected region."
        return "Reported as the mean across the visible retinal area."

    default_min_area_within_bounds = 0.80

    def __init__(
        self,
        grid_field: Optional[BaseGridFieldSpecification] = None,
        *,
        min_area_within_bounds: Optional[float] = None,
        plot: bool = False,
    ) -> None:
        """Masked mean luminance over the visible retina or a selected grid field."""
        self.min_area_within_bounds = validate_min_area_within_bounds(
            min_area_within_bounds
        )
        super().__init__(grid_field_spec=grid_field, plot=plot)

    def compute(self, retina: "Retina") -> Optional[float]:
        """Return mean Rec. 601 luminance inside the visible measurement region."""
        image = retina.image
        if image is None:
            return None
        visible = np.asarray(retina.mask).astype(bool)
        if self.grid_field_spec is not None:
            _, visible, fraction = grid_field_masks_and_fraction(
                retina, self.grid_field_spec
            )
            if fraction < resolve_min_area_within_bounds(
                self.grid_field_spec,
                self.min_area_within_bounds,
                self.default_min_area_within_bounds,
            ):
                return None
        if not np.any(visible):
            return None
        luma = to_luminance(image)
        if luma.shape[:2] != visible.shape[:2]:
            raise ValueError(
                f"Image/mask shape mismatch: {luma.shape[:2]} vs {visible.shape[:2]}"
            )
        return float(luma[visible].mean())

    def display_name(self, key: str = None, **kwargs) -> str:
        field = get_grid_field_suffix(self.grid_field_spec)
        return f"Luminance{field} - IM"

    def feature_name_tokens(self) -> list[str]:
        return ["luminance"]

    def _plot(self, ax, retina: "Retina", **kwargs):
        """Grayscale luminance is shown within the visible measurement region, with
        image bounds and the configured grid field overlaid.
        """
        if retina.image is None:
            return ax
        luma = to_luminance(retina.image)
        visible = np.asarray(retina.mask).astype(bool)
        field = self._get_grid_field(retina)
        if field is not None:
            visible &= field.mask.astype(bool)
        display = np.where(visible, luma, np.nan)
        ax.imshow(display, cmap="gray")
        retina.plot(ax=ax, image=False, bounds=True, av=False)
        if field is not None:
            field.plot(ax)
        return ax
