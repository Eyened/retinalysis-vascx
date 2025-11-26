from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from rtnls_enface.base import Circle, Line
from rtnls_enface.grids.ellipse import EllipseField
from rtnls_enface.grids.specifications import (
    BaseGridFieldSpecification,
    EllipseGridSpecification,
    GridFieldSpecification,
)

from .base import LayerFeature, grid_field_masks_and_fraction

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


def _default_ellipse_field_spec() -> GridFieldSpecification:
    """Return the default ellipse grid field specification."""
    return GridFieldSpecification(
        grid_spec=EllipseGridSpecification(),
        field=EllipseField.FullGrid,
    )


class VascularDensity(LayerFeature):
    """Fraction of vessel pixels within an OD–fovea-oriented ellipse (or ETDRS GridField) over its area.

    Representation: Uses VesselTreeLayer.binary vessel mask for pixel counting within specified regions.

    Computation: Calculates the ratio of vessel pixels to total pixels within an elliptical region centered
    between the optic disc and fovea, or within a specified ETDRS GridField. The ellipse extends from the
    optic disc with 1.5x aspect ratio along the OD-fovea axis.

    Args (constructor):
    - grid_field: optional `GridFieldEnum` selecting a predefined region; if None, uses an OD–fovea
      oriented ellipse (`EllipseGrid`).
    """

    def __init__(self, grid_field: Optional[BaseGridFieldSpecification] = None):
        """Configure optional ETDRS/other grid region; default uses an OD–fovea ellipse."""
        spec = grid_field if grid_field is not None else _default_ellipse_field_spec()
        super().__init__(grid_field_spec=spec)

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

        return f"VascularDensity(grid_field_spec={fmt(self.grid_field_spec)})"

    def get_circle(self, layer: VesselTreeLayer):
        disc = layer.retina.disc
        assert disc is not None

        od_to_fovea = Line(disc.center_of_mass, layer.retina.fovea_location)
        center = od_to_fovea.point_at(0.5)

        radius = od_to_fovea.length / 2 + disc.circle.r

        circle = Circle(center=center, r=radius)
        return circle

    def get_ellipse_mask(self, layer: VesselTreeLayer):
        grid = layer.retina.get_grid(EllipseGridSpecification())
        return grid.grid.astype(np.uint8) * 255

    def get_mask(self, layer: VesselTreeLayer):
        if self.grid_field_spec is None:
            return self.get_ellipse_mask(layer)
        else:
            field = self._get_grid_field(layer)
            return field.mask.astype(np.uint8) * 255

    def compute_for_mask(self, layer: VesselTreeLayer, mask: np.ndarray):
        # Use the mask to select pixels within the region in the image
        binary = layer.binary.astype(np.uint8)
        selected_pixels = cv2.bitwise_and(binary, binary, mask=mask)

        return np.sum(selected_pixels[mask == 255]) / np.sum(mask == 255)

    def compute(self, layer: VesselTreeLayer):
        # If using a grid_field, ensure at least 50% is within bounds
        if self.grid_field_spec is not None:
            _, in_bounds_mask, frac = grid_field_masks_and_fraction(
                layer.retina, self.grid_field_spec
            )
            if frac < 0.5:
                return None
        mask = self.get_mask(layer)

        mask[~layer.retina.mask] = 0
        return self.compute_for_mask(layer, mask)

    def _plot(self, ax, layer: VesselTreeLayer, **kwargs):
        field = self._get_grid_field(layer)
        ax = layer.plot(
            ax=ax,
            image=True,
            mask=True,
            grid_field=field,
        )
        return ax
