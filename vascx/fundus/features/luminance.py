from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .base import RetinaFeature

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
    visible mask pixels. Higher values indicate a brighter image.
    """

    general_description = (
        "Fundus luminance describes the overall brightness of the retinal image. "
        "Higher values indicate a brighter image."
    )

    def implementation_description(self, **kwargs) -> str:
        return "Calculated by converting the fundus image to luminance."

    def aggregation_description(self, **kwargs) -> str:
        return "Reported as the mean across the visible retinal area."

    def __init__(self, *, plot: bool = False) -> None:
        """Masked mean Rec. 601 luminance over the full retina FOV."""
        super().__init__(plot=plot)

    def compute(self, retina: "Retina") -> Optional[float]:
        """Return mean Rec. 601 luminance inside the retina mask."""
        image = retina.image
        if image is None:
            return None
        visible = np.asarray(retina.mask).astype(bool)
        if not np.any(visible):
            return None
        luma = to_luminance(image)
        if luma.shape[:2] != visible.shape[:2]:
            raise ValueError(
                f"Image/mask shape mismatch: {luma.shape[:2]} vs {visible.shape[:2]}"
            )
        return float(luma[visible].mean())

    def display_name(self, key: str = None, **kwargs) -> str:
        return "Luminance - IM"

    def feature_name_tokens(self) -> list[str]:
        return ["luminance"]

    def parameter_name_tokens(self) -> list[str]:
        return []

    def _plot(self, ax, retina: "Retina", **kwargs):
        if retina.image is None:
            return ax
        luma = to_luminance(retina.image)
        visible = np.asarray(retina.mask).astype(bool)
        display = np.where(visible, luma, np.nan)
        ax.imshow(display, cmap="gray")
        retina.plot(ax=ax, image=False, bounds=True, av=False)
        return ax
