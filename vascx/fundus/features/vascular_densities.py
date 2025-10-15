from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import matplotlib as mpl
import numpy as np

from rtnls_enface.base import Circle, Line

from .base import LayerFeature

if TYPE_CHECKING:
    from rtnls_enface.grids.base import GridField
    from vascx.fundus.layer import VesselTreeLayer


@dataclass
class VascularDensity(LayerFeature):
    """Fraction of vessel pixels within an OD–fovea-oriented ellipse (or ETDRS GridField) over its area.

    Representation: Uses VesselTreeLayer.binary vessel mask for pixel counting within specified regions.

    Computation: Calculates the ratio of vessel pixels to total pixels within an elliptical region centered 
    between the optic disc and fovea, or within a specified ETDRS GridField. The ellipse extends from the 
    optic disc with 1.5x aspect ratio along the OD-fovea axis.

    Options: grid_field (region selection), cut_mask (handle out-of-bounds regions by masking vs. warning).
    """
    
    def __init__(self, grid_field: GridField = None, cut_mask: bool = False):
        """
        reduce_mask: If true, trim the mask when parts of it are out of bounds of the CFI.
            Otherwise, an exception is generated.
        """
        self.grid_field = grid_field
        self.cut_mask = cut_mask

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
            f"VascularDensity(grid_field={fmt(self.grid_field)}, "
            f"cut_mask={fmt(self.cut_mask)})"
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
        disc = layer.retina.disc
        assert disc is not None

        od_to_fovea = Line(disc.center_of_mass, layer.retina.fovea_location)
        center = od_to_fovea.point_at(0.5)

        radius = od_to_fovea.length / 2 + disc.circle.r
        angle = od_to_fovea.orientation()

        # Create an empty mask with the same dimensions as the image
        mask = np.zeros(layer.binary.shape[:2], dtype=np.uint8)

        # Calculate the axes lengths from width and height
        axes = (int(radius), int(radius * 1.5))

        center = [int(e) for e in center.tuple_xy]
        # Draw the ellipse on the mask
        cv2.ellipse(
            mask,
            center,
            axes,
            angle,
            0,
            360,
            (255, 255, 255),
            -1,
        )

        return mask

    def get_mask(self, layer: VesselTreeLayer):
        if self.grid_field is None:
            return self.get_ellipse_mask(layer)
        else:
            return (
                layer.retina.grids[self.grid_field.grid()]
                .field(self.grid_field)
                .astype(np.uint8)
                * 255
            )

    def plot(self, ax, layer: VesselTreeLayer, **kwargs):
        # ax = layer.retina.plot(ax=ax)
        layer.plot(ax=ax, image=True)
        mask = self.get_mask(layer)
        density = self.compute(layer)

        # Plot the ellipse outline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ax.add_patch(
            mpl.patches.Polygon(
                contours[0][:, 0, :],
                closed=True,
                fill=False,
                edgecolor="w",
                linewidth=0.5,
            )
        )

        # ax.text(10, 30, f"{density:.3f}", color="white", fontsize=6)

    def compute_for_mask(self, layer: VesselTreeLayer, mask: np.ndarray):
        # Use the mask to select pixels within the region in the image
        binary = layer.binary.astype(np.uint8)
        selected_pixels = cv2.bitwise_and(binary, binary, mask=mask)

        return np.sum(selected_pixels[mask == 255]) / np.sum(mask == 255)

    def compute(self, layer: VesselTreeLayer):
        mask = self.get_mask(layer)

        if self.cut_mask:
            # zero-out everything outside the fundus mask
            mask[~layer.retina.mask] = 0
        else:
            # check that mask is within fundus bounds
            if not np.all(mask.astype(bool) <= layer.retina.mask):
                warnings.warn(
                    "Region for vascular density computation is out of bounds"
                )
                return np.nan

        return self.compute_for_mask(layer, mask)
