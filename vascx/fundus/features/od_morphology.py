"""Compound optic disc features derived from the existing segmentation and ellipse.

Torsion follows the major-axis deviation from the disc–fovea perpendicular:
https://doi.org/10.1038/s41598-019-38960-6
https://doi.org/10.1371/journal.pone.0213714
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Arc, Rectangle
from skimage.measure import perimeter_crofton

from vascx.shared.features import FeatureOutput
from .base import RetinaFeature


class ODMorphology(RetinaFeature):
    """Optic disc features as a fixed vector in ``outputs`` order.

    ``valid`` denotes an accepted VascX optic-disc ellipse, including clipped
    discs. Rejected or missing discs yield valid=0 and otherwise NaN. For a
    clipped disc, only fraction_outside_bounds, centroid coordinates, and the
    available fovea_distance are reported. The centroid and distance are
    estimates from the reconstructed ellipse when fraction_outside_bounds > 0.

    Fully visible means the reconstructed ellipse has visible_fraction == 1.
    Axes a and b are semi-axis lengths, not diameters. Area and circularity
    measure the cleaned segmentation, not its fitted ellipse. Pixel units refer
    to the loaded/preprocessed image (normally 1024x1024), not the original file.
    """

    outputs = (
        FeatureOutput("valid", "Valid", "1 if VascX accepted an optic-disc ellipse, otherwise 0; always reported."),
        FeatureOutput("fraction_outside_bounds", "Fraction outside bounds", "Fraction of reconstructed ellipse area outside the fundus ROI or image; 0 is fully visible.", "fraction"),
        FeatureOutput("centroid_x", "Centroid x", "Fitted ellipse centroid x (column) coordinate from the processed image’s top-left origin; estimated from the visible contour when fraction outside bounds > 0.", "px"),
        FeatureOutput("centroid_y", "Centroid y", "Fitted ellipse centroid y (row) coordinate from the processed image’s top-left origin; estimated from the visible contour when fraction outside bounds > 0.", "px"),
        FeatureOutput("fovea_distance", "Fovea distance", "Euclidean distance from the fitted ellipse centroid to the fovea, if available. Estimated when fraction outside bounds > 0.", "px"),
        FeatureOutput("ellipse_a", "Ellipse a", "Semi-major axis length of the fitted ellipse; reported only for fully visible discs.", "px"),
        FeatureOutput("ellipse_b", "Ellipse b", "Semi-minor axis length of the fitted ellipse; reported only for fully visible discs.", "px"),
        FeatureOutput("ovality_index", "Ovality index", "Semi-major / semi-minor axis ratio (a/b); 1 is circular. Reported only for fully visible discs."),
        FeatureOutput("circularity", "Circularity", "4*pi*A/P^2 from the cleaned disc segmentation's pixel area A and four-direction Crofton perimeter P. Approximately 1 for a circle; discretization can give values slightly above 1. Reported only for fully visible discs."),
        FeatureOutput("area", "Area", "Area of the cleaned optic-disc segmentation, natively counted in pixels; reported only for fully visible discs.", "px²"),
        FeatureOutput("area_norm", "Normalized area", "Segmentation area divided by squared centroid–fovea distance; requires a fully visible disc and a finite, nonzero distance."),
        FeatureOutput("torsion", "Torsion", "Unsigned acute angle (0–90 degrees) between the ellipse major axis and the perpendicular to the centroid–fovea vector. Requires a fully visible disc, a distinct fovea, and unequal ellipse axes; orientation is unstable for near-circular discs.", "degrees"),
    )
    general_description = "Optic disc features describe its visibility, position, size, and shape."

    def __init__(self, *, plot: bool = False):
        super().__init__(plot=plot)

    def feature_name_tokens(self):
        return ["od"]

    def parameter_name_tokens(self):
        return []

    def display_name(self, **kwargs):
        return "Optic disc - IM"

    def implementation_description(self, **kwargs):
        return (
            "Uses the accepted ellipse reconstruction and cleaned optic-disc segmentation. "
            "Centroid coordinates and fovea distance are estimates when any ellipse area "
            "is outside the visible bounds; shape and area are reported only for fully visible discs."
        )

    def region_description(self, **kwargs):
        return "Measured on the optic disc relative to the processed image’s top-left origin; coordinates do not anatomically align images."

    def compute(self, retina) -> np.ndarray:
        result = np.full(len(self.outputs), np.nan)
        result[0] = 0.0
        disc = retina.disc
        if disc is None or disc.ellipse.is_poor_fit:
            return result
        fit = disc.ellipse
        result[:4] = [1.0, 1.0 - fit.visible_fraction, fit.center.x, fit.center.y]
        fovea = retina.fovea_location
        distance = np.nan
        if fovea is not None:
            distance = np.hypot(fovea.x - fit.center.x, fovea.y - fit.center.y)
            if np.isfinite(distance):
                result[4] = distance
        if result[1] > 0:
            return result
        a, b = fit.semi_axes
        mask = np.asarray(disc.mask, dtype=bool)
        area = float(np.count_nonzero(mask))
        perimeter = perimeter_crofton(mask, directions=4)
        circularity = 4 * np.pi * area / perimeter**2 if perimeter > 0 else np.nan
        result[5:10] = [a, b, a / b, circularity, area]
        if np.isfinite(distance) and distance > 0:
            result[10] = area / distance**2
            if not np.isclose(a, b, rtol=1e-6, atol=0):
                reference = np.degrees(np.arctan2(fovea.y-fit.center.y, fovea.x-fit.center.x)) + 90
                result[11] = abs((fit.angle_degrees - reference + 90) % 180 - 90)
        return result

    def _plot_fovea_location(self, ax, layer):
        # Landmarks are already drawn with the optic-disc geometry.
        return ax

    def _plot(self, ax, retina, **kwargs):
        """The fundus shows segmentation (cyan), ellipse (yellow), bounds (magenta),
        and the disc–fovea line (white). Fully visible discs show axes a/b and
        torsion (green). The bottom box lists measurements and units; ~ denotes
        estimates and N/A missing values. Overlays use image pixels.
        """
        return self._draw(ax, retina, self.compute(retina))

    def _plot_with_value(self, ax, retina, value, **kwargs):
        return self._draw(ax, retina, np.asarray(self.output_values(value)))

    def _draw(self, ax, retina, values):
        ax.set_axis_off()
        image = retina.image
        if image is None:
            image = np.zeros((*retina.resolution, 3), dtype=np.uint8)
        h, w = image.shape[:2]
        ax.imshow(image)
        # Padding exposes the ROI boundary even when it meets an image edge.
        roi = np.pad(np.asarray(retina.mask, dtype=float), 1)
        ax.contour(np.arange(-1, w+1), np.arange(-1, h+1), roi,
                   levels=[0.5], colors=["#ee78d8"], linewidths=0.8)
        disc = retina.disc
        if disc is not None:
            ax.contour(disc.mask, levels=[0.5], colors=["#40e8e0"], linewidths=1.0)
            fit = disc.ellipse
            outline = fit.outline_xy()
            ax.plot(outline[:, 0], outline[:, 1], color="#ffd166", lw=1.3, ls="--")
            cx, cy = fit.center.x, fit.center.y
            ax.plot(cx, cy, marker="+", color="white", ms=8, mew=1.4)
            fovea = retina.fovea_location
            if fovea is not None and np.isfinite(values[4]):
                ax.plot([cx, fovea.x], [cy, fovea.y], color="white", lw=1, ls=":")
                ax.plot(fovea.x, fovea.y, marker="x", color="white", ms=6)
            if np.isfinite(values[5]):
                a, b = fit.semi_axes
                angle = np.radians(fit.angle_degrees)
                major = np.array([np.cos(angle), np.sin(angle)])
                minor = np.array([-np.sin(angle), np.cos(angle)])
                for direction, length, colour, label in (
                    (major, a, "#ffd166", "a"), (minor, b, "#72b7ff", "b")
                ):
                    end = np.array([cx, cy]) + length*direction
                    opposite = np.array([cx, cy]) - length*direction
                    ax.plot([opposite[0], end[0]], [opposite[1], end[1]], color=colour, lw=1.2)
                    ax.annotate(label, end, xytext=(3, 3), textcoords="offset points",
                                color=colour, fontsize=8, clip_on=True,
                                bbox=dict(facecolor="#14202d", alpha=0.65, edgecolor="none", pad=1))
                if np.isfinite(values[11]):
                    reference = np.arctan2(fovea.y-cy, fovea.x-cx) + np.pi/2
                    signed = (angle-reference+np.pi/2) % np.pi - np.pi/2
                    ref_vec = np.array([np.cos(reference), np.sin(reference)])
                    endpoints = np.array([cx, cy]) + np.array([-1, 1])[:, None]*a*1.3*ref_vec
                    ax.plot(endpoints[:, 0], endpoints[:, 1], color="#a6e86c", lw=1, ls="-.")
                    start, stop = sorted(np.degrees([reference, reference+signed]))
                    ax.add_patch(Arc((cx, cy), a*1.1, a*1.1, theta1=start, theta2=stop,
                                     color="#a6e86c", lw=2))
        # Keep the full fundus framing, including for an off-image fitted centroid.
        ax.set_xlim(-0.5, w-0.5)
        ax.set_ylim(h-0.5, -0.5)
        ax.text(0.02, 0.98, "Cyan: segmentation   Yellow: ellipse   Magenta: bounds",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color="white",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3))
        return ax

    def _annotate_value(self, ax, retina, value):
        values = self.output_values(value)
        schema = getattr(ax, "_vascx_output_schema", self.output_schema())
        clipped = np.isfinite(values[1]) and values[1] > 0
        status = "Clipped disc — position estimated" if clipped else "Fully visible optic disc"
        if not np.isfinite(values[0]) or values[0] == 0:
            status = "No valid optic disc"
        ax.add_patch(Rectangle((0, 0), 1, 0.36, transform=ax.transAxes,
                               facecolor="#09111c", edgecolor="none", alpha=0.76,
                               zorder=10, gid="od-measurements-background"))
        ax.text(0.025, 0.338, status, transform=ax.transAxes, va="top", fontsize=9,
                weight="bold", color="white", zorder=11)
        labels = ["Valid", "Outside bounds", "Centroid x", "Centroid y", "Fovea distance",
                  "Semi-major a", "Semi-minor b", "Ovality a/b", "Circularity", "Area", "Area / distance²", "Torsion"]
        # All twelve outputs remain visible, including missing measurements.
        for column, indices in enumerate(((0, 1, 2, 3, 4, 10), (5, 6, 7, 8, 9, 11))):
            for row, i in enumerate(indices):
                v = values[i]
                if not np.isfinite(v):
                    text = "N/A"
                elif i == 0:
                    text = "Yes (1)" if v else "No (0)"
                elif i == 1:
                    text = f"{v:.1%}"
                elif i in (2, 3, 4, 5, 6):
                    text = f"{v:.4g} {schema[i].units}"
                elif i == 9:
                    text = f"{v:.4g} {schema[i].units}"
                elif i == 11:
                    text = f"{v:.1f}°"
                else:
                    text = f"{v:.4g}"
                if clipped and i in (2, 3, 4) and np.isfinite(v):
                    text = "~" + text
                x, y = 0.025 + column*0.50, 0.277-row*0.041
                ax.text(x, y, labels[i], transform=ax.transAxes, fontsize=7.2, color="#e2e8f0", va="center", zorder=11)
                ax.text(x+0.45, y, text, transform=ax.transAxes, fontsize=7.2, color="white", ha="right", va="center", weight="medium", zorder=11)
        if not np.isfinite(values[0]) or values[0] == 0:
            footer = "Measurements unavailable: no accepted ellipse."
        elif clipped:
            footer = "Shape/area withheld: disc is clipped."
        else:
            footer = "Torsion: major axis vs disc–fovea perpendicular (green)."
        ax.text(0.5, 0.015, footer, transform=ax.transAxes, fontsize=6.5,
                ha="center", va="bottom", color="#cbd5e1", zorder=11)
