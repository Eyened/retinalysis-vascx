from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import matplotlib as mpl
import numpy as np

from rtnls_enface.base import Circle, Line
from rtnls_enface.grids.ellipse import EllipseGrid

from .base import LayerFeature, grid_field_masks_and_fraction

if TYPE_CHECKING:
    from rtnls_enface.grids.base import GridFieldEnum
    from vascx.fundus.layer import VesselTreeLayer


class VascularDensity(LayerFeature):
    """Fraction of vessel pixels within an OD–fovea-oriented ellipse (or ETDRS GridField) over its area.

    Representation: Uses VesselTreeLayer.binary vessel mask for pixel counting within specified regions.

    Computation: Calculates the ratio of vessel pixels to total pixels within an elliptical region centered 
    between the optic disc and fovea, or within a specified ETDRS GridField. The ellipse extends from the 
    optic disc with 1.5x aspect ratio along the OD-fovea axis.

    Options: grid_field (region selection).
    """
    
    def __init__(self, grid_field: GridFieldEnum = None):
        """
        reduce_mask: If true, trim the mask when parts of it are out of bounds of the CFI.
            Otherwise, an exception is generated.
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
        return (
            f"VascularDensity(grid_field={fmt(self.grid_field)})"
        )

    def get_circle(self, layer: VesselTreeLayer):
        disc = layer.retina.disc
        assert disc is not None

        od_to_fovea = Line(disc.center_of_mass, layer.retina.fovea_location)
        center = od_to_fovea.point_at(0.5)

        radius = od_to_fovea.length / 2 + disc.circle.r

        circle = Circle(center=center, r=radius)
        return circle

    def get_ellipse_mask(self, layer: VesselTreeLayer):
        grid = layer.retina.grids[EllipseGrid]
        return grid.grid.astype(np.uint8) * 255

    def get_mask(self, layer: VesselTreeLayer):
        if self.grid_field is None:
            return self.get_ellipse_mask(layer)
        else:
            grid = layer.retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
            return field.mask.astype(np.uint8) * 255

    

    def compute_for_mask(self, layer: VesselTreeLayer, mask: np.ndarray):
        # Use the mask to select pixels within the region in the image
        binary = layer.binary.astype(np.uint8)
        selected_pixels = cv2.bitwise_and(binary, binary, mask=mask)

        return np.sum(selected_pixels[mask == 255]) / np.sum(mask == 255)

    def compute(self, layer: VesselTreeLayer):
        # If using a grid_field, ensure at least 50% is within bounds
        if self.grid_field is not None:
            _, in_bounds_mask, frac = grid_field_masks_and_fraction(layer.retina, self.grid_field)
            if frac < 0.5:
                return None
        mask = self.get_mask(layer)

        mask[~layer.retina.mask] = 0
        return self.compute_for_mask(layer, mask)

    def _plot(self, ax, layer: VesselTreeLayer, **kwargs):
        field = None
        if self.grid_field is not None:
            grid = layer.retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
        ax = layer.plot(
            ax=ax,
            image=True,
            mask=True,
            grid_field=field,
        )
        return ax
