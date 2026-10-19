"""Compound measurements of fundus vessel bifurcations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from matplotlib.artist import Artist
from matplotlib.patches import Arc
from matplotlib.text import Annotation
from matplotlib.transforms import Bbox
from rtnls_enface.grids.specifications import BaseGridFieldSpecification
from skimage.util import img_as_float

from vascx.shared.features import FeatureOutput
from .base import (
    LayerFeature, format_name_value, get_grid_field_suffix, get_layer_suffix,
    grid_field_passes_qc, resolve_min_area_within_bounds, validate_min_area_within_bounds,
)

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer
    from vascx.shared.nodes import Bifurcation as BifurcationNode


@dataclass(frozen=True)
class _Measurement:
    node: BifurcationNode
    angle: float
    branching_coefficient: float = np.nan
    jed: float = np.nan


class _BifurcationLabels(Artist):
    """Place compact measurement labels beside junctions, including after resizing."""

    def __init__(self, ax, measurements, marker_radius=0):
        super().__init__()
        self.marker_radius = marker_radius
        self.labels = []
        self.drawn_extents = []
        for measurement in measurements:
            coefficient = (f"{measurement.branching_coefficient:.2f}"
                           if np.isfinite(measurement.branching_coefficient) else "N/A")
            jed = f"{measurement.jed:.2f}" if np.isfinite(measurement.jed) else "N/A"
            label = Annotation(
                f"{measurement.angle:.0f}° | {coefficient} | {jed}",
                measurement.node.position.tuple_xy,
                xytext=(7, 3), textcoords="offset points", fontsize=7.2,
                color="white", annotation_clip=True,
                bbox=dict(boxstyle="round,pad=0.12", fc="#14202d", ec="none", alpha=0.5),
            )
            label.axes = ax
            label.set_figure(ax.figure)
            self.labels.append(label)
        self.set_zorder(7)

    def get_children(self):
        # Let report typography include these dynamically positioned annotations.
        return self.labels

    def draw(self, renderer):
        self.drawn_extents = []
        radius = renderer.points_to_pixels(3.5)
        summary_boxes = [text.get_window_extent(renderer).expanded(1.05, 1.1)
                         for text in self.axes.texts
                         if text.get_gid() == "bifurcation-summary"]
        marker_boxes = []
        for label in self.labels:
            x, y = self.axes.transData.transform(label.xy)
            edge = self.axes.transData.transform(np.asarray(label.xy) + self.marker_radius)
            extent = np.maximum(np.abs(edge - (x, y)), radius)
            marker_boxes.append(Bbox.from_extents(x-extent[0], y-extent[1],
                                                 x+extent[0], y+extent[1]))
        for label in self.labels:
            font_scale = label.get_fontsize() / 7.2
            best = None
            # Try nearby positions first; preserve every label even in dense clusters.
            for distance in (7, 13, 21, 30, 42, 58):
                candidates = ((distance, 3, "left", "bottom"),
                              (-distance, 3, "right", "bottom"),
                              (distance, -3, "left", "top"),
                              (-distance, -3, "right", "top"),
                              (0, distance, "center", "bottom"),
                              (0, -distance, "center", "top"),
                              (distance, distance, "left", "bottom"),
                              (-distance, distance, "right", "bottom"),
                              (distance, -distance, "left", "top"),
                              (-distance, -distance, "right", "top"))
                for dx, dy, ha, va in candidates:
                    label.set_position((dx * font_scale, dy * font_scale))
                    label.set_horizontalalignment(ha)
                    label.set_verticalalignment(va)
                    bbox = label.get_window_extent(renderer).expanded(1.03, 1.25)
                    if not self.axes.bbox.contains(bbox.x0, bbox.y0) or not self.axes.bbox.contains(bbox.x1, bbox.y1):
                        continue
                    overlap = (100 * sum(bbox.overlaps(other) for other in self.drawn_extents)
                               + sum(bbox.overlaps(other) for other in marker_boxes)
                               + 1000 * sum(bbox.overlaps(other) for other in summary_boxes))
                    if best is None or overlap < best[0]:
                        best = (overlap, (dx * font_scale, dy * font_scale), ha, va, bbox)
                    if overlap == 0:
                        break
                if best is not None and best[0] == 0:
                    break
            if best is None or best[0] > 0:
                # Large report fonts can exhaust the nearby positions. Search
                # the remaining panel space, nearest to the junction first.
                label.set_horizontalalignment("left")
                label.set_verticalalignment("bottom")
                label.set_position((0, 0))
                box = label.get_window_extent(renderer).expanded(1.03, 1.25)
                x, y = self.axes.transData.transform(label.xy)
                points = renderer.points_to_pixels(1)
                bounds = self.axes.bbox
                if box.width < bounds.width and box.height < bounds.height:
                    xs = np.linspace(bounds.x0, bounds.x1 - box.width, 24)
                    ys = np.linspace(bounds.y0, bounds.y1 - box.height, 32)
                    positions = sorted(((px, py) for px in xs for py in ys),
                                       key=lambda p: (p[0] + box.width/2 - x)**2 + (p[1] + box.height/2 - y)**2)
                    for px, py in positions:
                        bbox = Bbox.from_bounds(px, py, box.width, box.height)
                        overlap = (100 * sum(bbox.overlaps(other) for other in self.drawn_extents)
                                   + sum(bbox.overlaps(other) for other in marker_boxes)
                                   + 1000 * sum(bbox.overlaps(other) for other in summary_boxes))
                        if best is None or overlap < best[0]:
                            best = (overlap, ((px-box.x0)/points, (py-box.y0)/points),
                                    "left", "bottom", bbox)
                        if overlap == 0:
                            break
            if best is not None:
                _, position, ha, va, bbox = best
                label.set_position(position)
                label.set_horizontalalignment(ha)
                label.set_verticalalignment(va)
                self.drawn_extents.append(bbox)
            label.draw(renderer)
        self.stale = False


class Bifurcation(LayerFeature):
    """Means of angle, branching coefficient and absolute cubic JED, plus count.

    All components use bifurcations inside the requested region with both
    daughters longer than delta and finite opening angle <= max_angle. Count
    includes eligible nodes lacking a usable diameter triplet. The angle mean
    and diameter-based means each require min_bifurcations valid measurements.
    An adequately visible empty region has count=0; an unavailable region has
    all NaNs. Diameters are whole-segment means, not measurements at delta.
    """

    outputs = (
        FeatureOutput("angle_mean", "Mean angle",
                      "Arithmetic mean of eligible daughter–daughter opening angles.", "degrees"),
        FeatureOutput("branching_coefficient_mean", "Mean branching coefficient",
                      "Mean of (d1²+d2²)/d0² per eligible bifurcation, using whole-segment mean diameters; d0 is the parent.", "dimensionless"),
        FeatureOutput("jed_mean", "Mean JED",
                      "Mean absolute cubic junctional exponent deviation: cuberoot(abs(d0³-d1³-d2³))/d0. Zero satisfies Murray's law. Uses whole-segment mean diameters, not a fitted exponent.", "dimensionless"),
        FeatureOutput("count", "Bifurcation count",
                      "Number passing region, daughter-length and angle filters, including nodes without valid diameter measurements. Reported even below the minimum required for means.", "count"),
    )
    general_description = "Retinal bifurcation features describe branch opening, parent–daughter calibre relationships, and the number of eligible junctions."
    default_min_area_within_bounds = 0.95
    preferred_plot_size = (8, 8)

    def __init__(
        self, delta: int = 20, grid_field: Optional[BaseGridFieldSpecification] = None,
        max_angle: int = 135, min_bifurcations: int = 3,
        spline_error_fraction: float = 0.05,
        min_area_within_bounds: Optional[float] = None, plot: bool = False,
    ):
        self.delta = delta
        self.max_angle = max_angle
        self.min_bifurcations = min_bifurcations
        self.spline_error_fraction = float(spline_error_fraction)
        self.min_area_within_bounds = validate_min_area_within_bounds(min_area_within_bounds)
        super().__init__(grid_field_spec=grid_field, plot=plot)

    def feature_name_tokens(self):
        return ["bif"]

    def parameter_name_tokens(self):
        tokens = []
        for name, default in (("min_area_within_bounds", None), ("delta", 20),
                              ("max_angle", 135), ("min_bifurcations", 3),
                              ("spline_error_fraction", 0.05)):
            value = getattr(self, name)
            if value != default:
                tokens.extend([name, format_name_value(value)])
        return tokens

    def display_name(self, layer_name: str, **kwargs):
        return f"Bifurcation{get_grid_field_suffix(self.grid_field_spec)}{get_layer_suffix(layer_name)}"

    def implementation_description(self, **kwargs):
        return (
            f"Daughter directions are sampled {self.delta:g} pixels from the junction using "
            f"splines (error fraction {self.spline_error_fraction:g}). Both daughters must be "
            f"longer than this distance and the opening angle must be at most {self.max_angle:g} degrees. "
            "Parent and daughter diameters are existing whole-segment mean diameters. "
            f"Each mean requires at least {self.min_bifurcations} valid observations; "
            "missing diameters do not remove valid angles or reduce the filtered count."
        )

    def _measurements(self, layer: VesselTreeLayer):
        # Cache on the layer so records expire with the image, not with a global feature set.
        cache = layer.__dict__.setdefault("_bifurcation_feature_cache", {})
        key = (self, self.delta, self.max_angle, self.spline_error_fraction,
               id(self.grid_field_spec), self.min_area_within_bounds)
        if key in cache:
            return cache[key]
        if self.grid_field_spec is not None:
            try:
                valid_region = grid_field_passes_qc(
                    layer.retina, self.grid_field_spec,
                    min_fraction_in_bounds=resolve_min_area_within_bounds(
                        self.grid_field_spec, self.min_area_within_bounds,
                        self.default_min_area_within_bounds))
                field = self._get_grid_field(layer) if valid_region else None
            except (ValueError, AttributeError):
                valid_region, field = False, None
            if not valid_region:
                cache[key] = None
                return None
        else:
            field = None
        measurements = []
        for bif in layer.filter_bifurcations(field):
            if not bif.outgoing_min_length > self.delta:
                continue
            try:
                angle = float(bif.angle(self.delta, self.spline_error_fraction))
            except (ValueError, RuntimeError, FloatingPointError):
                continue
            if not np.isfinite(angle) or angle > self.max_angle:
                continue
            bc = jed = np.nan
            try:
                diameters = np.array([
                    segment.get_mean_diameter(self.spline_error_fraction)
                    for segment in (bif.incoming, *bif.outgoing[:2])
                ], dtype=float)
                if np.isfinite(diameters).all() and (diameters > 0).all():
                    r1, r2 = diameters[1:] / diameters[0]
                    with np.errstate(over="ignore", invalid="ignore"):
                        bc = r1**2 + r2**2
                        jed = np.cbrt(abs(1 - r1**3 - r2**3))
                    if not np.isfinite([bc, jed]).all():
                        bc = jed = np.nan
            except (AttributeError, ValueError, RuntimeError, FloatingPointError):
                # Diameter failure is local; it must not discard an otherwise usable angle.
                pass
            measurements.append(_Measurement(bif, angle, bc, jed))
        cache[key] = tuple(measurements)
        return cache[key]

    def compute(self, layer: VesselTreeLayer) -> np.ndarray:
        records = self._measurements(layer)
        result = np.full(4, np.nan)
        if records is None:
            return result
        result[3] = len(records)
        if records and len(records) >= self.min_bifurcations:
            result[0] = np.mean([r.angle for r in records])
        valid = [r for r in records if np.isfinite(r.branching_coefficient) and np.isfinite(r.jed)]
        if valid and len(valid) >= self.min_bifurcations:
            result[1:3] = np.mean([[r.branching_coefficient, r.jed] for r in valid], axis=0)
        return result

    def _plot_fovea_location(self, ax, layer):
        return ax

    def _annotate_value(self, ax, layer, value):
        values = self.output_values(value)
        labels = ("Mean angle", "Mean coefficient", "Mean JED", "Eligible count")
        lines = []
        for i, (label, v) in enumerate(zip(labels, values)):
            text = "N/A" if not np.isfinite(v) else (f"{v:.1f}°" if i == 0 else f"{v:.3g}")
            lines.append(f"{label}: {text}")
        records = self._measurements(layer)
        if records is None:
            lines.append("Region unavailable or insufficiently visible")
        else:
            valid = sum(np.isfinite(r.branching_coefficient) for r in records)
            lines.append(f"Angles: n={len(records)}; diameter triplets: n={valid}")
        ax.text(.02, .98, "\n".join(lines), transform=ax.transAxes,
                va="top", ha="left", color="white", fontsize=8, zorder=9,
                gid="bifurcation-summary",
                bbox=dict(facecolor="black", alpha=.65, edgecolor="none", pad=4))

    def _plot(self, ax, layer, **kwargs):
        """Two short daughter-direction rays and an arc mark each opening angle.
        Markers are unfilled so the underlying vessels remain visible.
        Each label gives angle (integer degrees) | branching coefficient | JED
        (two decimals). N/A denotes unavailable diameter measurements. The outline
        shows the measurement region; the summary reports all four components.
        """
        return self._draw(ax, layer)

    def _draw(self, ax, layer):
        ax.set_axis_off()
        records = self._measurements(layer)
        image = layer.retina.image
        if image is None:
            image = np.zeros((*layer.retina.resolution, 3))
        image = img_as_float(image) * .85
        try:
            field = self._get_grid_field(layer)
        except (ValueError, AttributeError):
            field = None
        panel = ax
        panel.imshow(image)
        region_mask = field.mask if field is not None else layer.retina.mask
        roi = np.pad(np.asarray(region_mask, dtype=float), 1)
        h, w = roi.shape
        panel.contour(np.arange(-1, w-1), np.arange(-1, h-1), roi,
                      levels=[.5], colors=["#bed5e5"], linewidths=.65, alpha=.75)
        if records:
            for record in records:
                line1, line2 = record.node.lines(self.delta, self.spline_error_fraction)
                for line in (line1, line2):
                    line.plot(panel, color="white", linewidth=.75, alpha=.85, zorder=4)
                radius = .55 * min(line1.length, line2.length)
                start, end = line1.orientation(), line2.orientation()
                if (end - start) % 360 > 180:
                    start, end = end, start
                panel.add_patch(Arc(record.node.position.tuple_xy, 2*radius, 2*radius,
                                    theta1=start, theta2=end,
                                    color="white", linewidth=.65, alpha=.85, zorder=4))
            panel.add_artist(_BifurcationLabels(panel, records, marker_radius=self.delta))
        panel.set_xlim(-.5, image.shape[1]-.5)
        panel.set_ylim(image.shape[0]-.5, -.5)
        return ax
