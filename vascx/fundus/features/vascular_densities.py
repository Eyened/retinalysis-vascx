from __future__ import annotations

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
    def __init__(self, grid_field: GridField = None):
        self.grid_field = grid_field

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

    def plot(self, layer: VesselTreeLayer, ax=None, **kwargs):
        ax = layer.retina.plot(ax=ax)
        mask = self.get_ellipse_mask(layer)
        density = self.compute_for_mask(layer, mask)

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

        ax.text(10, 30, f"{density:.3f}", color="white", fontsize=6)

    def compute_for_mask(self, layer: VesselTreeLayer, mask: np.ndarray):
        # Use the mask to select pixels within the region in the image
        binary = layer.binary.astype(np.uint8)
        selected_pixels = cv2.bitwise_and(binary, binary, mask=mask)

        return np.sum(selected_pixels[mask == 255]) / np.sum(mask == 255)

    def compute(self, layer: VesselTreeLayer):
        mask = self.get_ellipse_mask(layer)

        return self.compute_for_mask(layer, mask)
