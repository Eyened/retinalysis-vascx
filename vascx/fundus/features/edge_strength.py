from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from rtnls_enface.grids.specifications import BaseGridFieldSpecification
from skimage.morphology import binary_dilation, binary_erosion, binary_opening, disk

from .base import (
    VesselsLayerFeature,
    format_name_value,
    grid_field_fraction_in_bounds,
    resolve_min_area_within_bounds,
    validate_min_area_within_bounds,
)

if TYPE_CHECKING:
    from vascx.fundus.vessels_layer import FundusVesselsLayer


class EdgeStrength(VesselsLayerFeature):
    """Vessel-edge strength from median Scharr magnitude on a thin edge band.

    Representation: Uses the green channel of the fundus image and the vessels binary
    mask. Thin vessels are removed by morphological opening; a thin annular band is
    taken around the remaining vessel edges.

    Computation: median of M / I^beta on the edge band, where M is Scharr magnitude and
    I is a smoothed green-channel intensity. beta=0 is raw Scharr; beta=1 is full
    brightness normalization; intermediate values (e.g. 0.5) partially correct for
    brightness. Higher values indicate stronger vessel edges (contrast and/or sharpness).
    """

    default_min_area_within_bounds = 0.80
    default_min_radius = 2
    default_band_width = 1
    default_beta = 0.0
    default_intensity_smooth_sigma = 1.0
    _min_intensity = 1.0

    def __init__(
        self,
        grid_field: Optional[BaseGridFieldSpecification] = None,
        min_radius: int = default_min_radius,
        band_width: int = default_band_width,
        beta: float = default_beta,
        intensity_smooth_sigma: float = default_intensity_smooth_sigma,
        min_area_within_bounds: Optional[float] = None,
    ):
        """Edge strength on vessel edges, optionally restricted to a grid field.

        min_radius is the morphological opening radius in pixels used to drop thin
        vessels. band_width is the half-width of the edge band in pixels.
        beta controls brightness normalization via M / I^beta (0 = raw, 1 = full).
        intensity_smooth_sigma is the Gaussian sigma for I when beta > 0.
        """
        if min_radius < 1:
            raise ValueError("min_radius must be >= 1")
        if band_width < 1:
            raise ValueError("band_width must be >= 1")
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError("beta must be in [0.0, 1.0]")
        if intensity_smooth_sigma < 0:
            raise ValueError("intensity_smooth_sigma must be >= 0")
        self.min_radius = int(min_radius)
        self.band_width = int(band_width)
        self.beta = float(beta)
        self.intensity_smooth_sigma = float(intensity_smooth_sigma)
        self.min_area_within_bounds = validate_min_area_within_bounds(
            min_area_within_bounds
        )
        super().__init__(grid_field_spec=grid_field)

    def _green_channel(self, layer: FundusVesselsLayer) -> Optional[np.ndarray]:
        image = layer.retina.image
        if image is None:
            return None
        if image.ndim == 2:
            return image.astype(np.float64)
        return image[..., 1].astype(np.float64)

    def _scharr_magnitude(self, gray: np.ndarray) -> np.ndarray:
        gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        return np.hypot(gx, gy)

    def _smoothed_intensity(self, gray: np.ndarray) -> np.ndarray:
        """Local intensity used in the brightness-normalization denominator."""
        if self.intensity_smooth_sigma <= 0:
            return gray
        return cv2.GaussianBlur(
            gray,
            ksize=(0, 0),
            sigmaX=self.intensity_smooth_sigma,
            sigmaY=self.intensity_smooth_sigma,
        )

    def _thick_vessels(self, layer: FundusVesselsLayer) -> np.ndarray:
        """Vessel mask after removing vessels thinner than ``min_radius``."""
        binary = layer.binary.astype(bool)
        thick = binary_opening(binary, disk(self.min_radius))
        if layer.retina.disc is not None:
            thick = thick & ~layer.retina.disc.mask.astype(bool)
        return thick

    def _edge_band(self, thick: np.ndarray) -> np.ndarray:
        """Thin annular band around thick-vessel edges."""
        selem = disk(self.band_width)
        dilated = binary_dilation(thick, selem)
        eroded = binary_erosion(thick, selem)
        return dilated & ~eroded

    def _roi_mask(self, layer: FundusVesselsLayer) -> Optional[np.ndarray]:
        """Retina ROI, optionally intersected with a grid field after QC."""
        mask = layer.retina.mask.astype(bool)
        if self.grid_field_spec is None:
            return mask

        frac = grid_field_fraction_in_bounds(layer.retina, self.grid_field_spec)
        if frac < resolve_min_area_within_bounds(
            self.grid_field_spec,
            self.min_area_within_bounds,
            self.default_min_area_within_bounds,
        ):
            return None
        field = self._get_grid_field(layer)
        if field is None:
            return None
        return mask & field.mask.astype(bool)

    def _evaluation_mask(
        self, layer: FundusVesselsLayer, thick: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Mask of pixels used for the edge-strength aggregate."""
        if thick is None:
            thick = self._thick_vessels(layer)
        if not np.any(thick):
            return None

        roi = self._roi_mask(layer)
        if roi is None:
            return None

        band = self._edge_band(thick) & roi
        if not np.any(band):
            return None
        return band

    def compute(self, layer: FundusVesselsLayer) -> Optional[float]:
        gray = self._green_channel(layer)
        if gray is None:
            return None

        band = self._evaluation_mask(layer)
        if band is None:
            return None

        mag = self._scharr_magnitude(gray)
        if self.beta == 0.0:
            vals = mag[band]
        else:
            intensity = np.maximum(
                self._smoothed_intensity(gray), self._min_intensity
            )
            vals = (mag / np.power(intensity, self.beta))[band]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_grid_field_suffix, get_layer_suffix

        field = get_grid_field_suffix(self.grid_field_spec)
        layer = get_layer_suffix(layer_name)
        if self.beta != self.default_beta:
            return f"Edge Strength (β={self.beta:g}){field}{layer}"
        return f"Edge Strength{field}{layer}"

    def feature_name_tokens(self) -> list[str]:
        return ["edge_strength"]

    def parameter_name_tokens(self) -> list[str]:
        tokens: list[str] = []
        if self.min_area_within_bounds is not None:
            tokens.extend(
                [
                    "min_area_within_bounds",
                    format_name_value(self.min_area_within_bounds),
                ]
            )
        if self.min_radius != self.default_min_radius:
            tokens.extend(["min_radius", str(self.min_radius)])
        if self.band_width != self.default_band_width:
            tokens.extend(["band_width", str(self.band_width)])
        if self.beta != self.default_beta:
            tokens.extend(["beta", format_name_value(self.beta)])
        if self.intensity_smooth_sigma != self.default_intensity_smooth_sigma:
            tokens.extend(
                [
                    "intensity_smooth_sigma",
                    format_name_value(self.intensity_smooth_sigma),
                ]
            )
        return tokens

    def _plot(self, ax, layer: FundusVesselsLayer, **kwargs):
        layer.plot(ax=ax, image=True, bounds=True, skeleton=False, mask=False)

        band = self._evaluation_mask(layer)
        if band is not None:
            overlay = np.zeros((*band.shape, 4), dtype=float)
            overlay[band, :] = (0.0, 1.0, 1.0, 0.45)
            ax.imshow(overlay)

        field = self._get_grid_field(layer)
        if field is not None:
            field.plot(ax)
        return ax
