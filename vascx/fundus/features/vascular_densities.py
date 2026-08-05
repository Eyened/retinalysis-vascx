from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from .base import (
    LayerFeature,
    grid_field_passes_qc,
    resolve_min_area_within_bounds,
    validate_min_area_within_bounds,
)

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


class VascularDensity(LayerFeature):
    """Fraction of vessel pixels over a region: full retina mask, or a grid field mask if specified.

    Representation: Uses VesselTreeLayer.binary vessel mask for pixel counting within the region.

    Computation: Ratio of vessel pixels to valid (masked) pixels. With no `grid_field`, the region is the
    whole fundus mask (`retina.mask`). With a grid field, counting is restricted to that field’s mask.

    Args (constructor):
    - grid_field: optional grid field specification; if None, density is over the full retina mask.
    """

    default_min_area_within_bounds = 1.0

    def __init__(
        self,
        grid_field: Optional[BaseGridFieldSpecification] = None,
        min_area_within_bounds: Optional[float] = None,
    ):
        """Configure optional grid region; default is full retina (no grid field)."""
        self.min_area_within_bounds = validate_min_area_within_bounds(
            min_area_within_bounds
        )
        super().__init__(grid_field_spec=grid_field)

    def get_mask(self, layer: VesselTreeLayer):
        if self.grid_field_spec is None:
            return layer.retina.mask.astype(np.uint8) * 255
        field = self._get_grid_field(layer)
        return field.mask.astype(np.uint8) * 255

    def compute_for_mask(self, layer: VesselTreeLayer, mask: np.ndarray):
        # Use the mask to select pixels within the region in the image
        binary = layer.binary.astype(np.uint8)
        selected_pixels = cv2.bitwise_and(binary, binary, mask=mask)

        return np.sum(selected_pixels[mask == 255]) / np.sum(mask == 255)

    def compute(self, layer: VesselTreeLayer):
        if self.grid_field_spec is not None:
            if not grid_field_passes_qc(
                layer.retina,
                self.grid_field_spec,
                min_fraction_in_bounds=resolve_min_area_within_bounds(
                    self.grid_field_spec,
                    self.min_area_within_bounds,
                    self.default_min_area_within_bounds,
                ),
            ):
                return None
        mask = self.get_mask(layer)

        mask[~layer.retina.mask] = 0
        return self.compute_for_mask(layer, mask)

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_grid_field_suffix, get_layer_suffix

        field = get_grid_field_suffix(self.grid_field_spec)
        layer = get_layer_suffix(layer_name)
        return f"Vessel Density{field}{layer}"

    def feature_name_tokens(self) -> list[str]:
        return ["vd"]

    def _plot(self, ax, layer: VesselTreeLayer, **kwargs):
        field = self._get_grid_field(layer)
        return layer.plot(
            ax=ax,
            image=True,
            bounds=True,
            mask=True,
            grid_field=field,
        )
