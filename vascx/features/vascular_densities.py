from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from rtnls_enface.types import Circle, Ellipse, Line

from .base import LayerFeature

if TYPE_CHECKING:
    from vascx.retina import VesselLayer


@dataclass
class VascularDensity(LayerFeature):
    def get_circle(self, layer: VesselLayer):
        disc = layer.retina.disc
        assert disc is not None

        od_to_fovea = Line(disc.center_of_mass, layer.retina.fovea_location)
        center = od_to_fovea.point_at(0.5)

        radius = od_to_fovea.length / 2 + disc.circle.r

        circle = Circle(center=center, r=radius)
        return circle

    def get_ellipse(self, layer):
        disc = layer.retina.disc
        assert disc is not None

        od_to_fovea = Line(disc.center_of_mass, layer.retina.fovea_location)
        center = od_to_fovea.point_at(0.5)

        radius = od_to_fovea.length / 2 + disc.circle.r
        angle = od_to_fovea.orientation()

        ellipse = Ellipse(
            center=center, width=2 * radius, height=2 * radius * 1.5, angle=angle
        )
        return ellipse

    def plot_circle(self, layer: VesselLayer, fig=None, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()

        circle = self.get_circle(layer)
        layer.retina.plot_fundus(fig=fig, ax=ax)
        ax.add_patch(
            plt.Circle(circle.center.tuple_xy, circle.r, color="w", fill=False, lw=0.5)
        )

    def plot(self, layer: VesselLayer, fig=None, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()

        ellipse = self.get_ellipse(layer)
        density = self.compute_for_ellipse(layer, ellipse)
        layer.retina.plot_fundus(fig=fig, ax=ax)
        ax.add_patch(
            mpl.patches.Ellipse(
                ellipse.center.tuple_xy,
                ellipse.width,
                ellipse.height,
                ellipse.angle,
                color="w",
                fill=False,
                lw=0.5,
            )
        )
        ax.text(10, 30, f"{density:.3f}", color="white", fontsize=6)

    def compute_for_ellipse(self, layer: VesselLayer, ellipse: Ellipse):
        # Create an empty mask with the same dimensions as the image
        mask = np.zeros(layer.binary.shape[:2], dtype=np.uint8)

        # Calculate the axes lengths from width and height
        axes = (int(ellipse.width / 2), int(ellipse.height / 2))

        center = ellipse.center.tuple_xy
        center = [int(e) for e in center]
        # Draw the ellipse on the mask
        # angle is the rotation angle in degrees
        # 0 and 360 define the start and end of the ellipse arc, drawing the full ellipse
        cv2.ellipse(
            mask,
            center,
            axes,
            ellipse.angle,
            0,
            360,
            (255, 255, 255),
            -1,
        )

        # Use the mask to select pixels within the ellipse in the image
        selected_pixels = cv2.bitwise_and(layer.binary, layer.binary, mask=mask)

        return np.sum(selected_pixels[mask == 255]) / np.sum(mask == 255)

    def compute(self, layer: VesselLayer):
        ellipse = self.get_ellipse(layer)

        return self.compute_for_ellipse(layer, ellipse)
