from __future__ import annotations

import warnings
from itertools import combinations
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from rtnls_enface.base import Circle, Line, Point

from vascx.shared.vessels import Vessels

from .base import LayerFeature

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer
    from vascx.shared.segment import Segment


class TemporalAngle(LayerFeature):
    """Median angle between the two dominant (largest caliber) temporal arcades at circles from 2/3 OD–fovea distance outward.

    Representation: Uses resolved_segments with circle intersections and OD–fovea spatial geometry.
    Operates on the resolved vessel graph to identify major temporal vessel arcades.

    Computation: At concentric circles starting from 2/3 of the OD-fovea distance, identifies the two
    largest-caliber temporal vessels (fovea-side, angle < 90°) crossing each circle, measures the angle
    between them, and returns the median angle across all sampled radii. Captures the characteristic
    temporal arcade geometry.

    Args (constructor):
    - inner_circle: inner sampling radius as a fraction of the OD–fovea distance.
    - outer_circle: outer sampling radius as a fraction of the OD–fovea distance.
    - num_circles: total number of circles sampled between inner and outer radii, inclusive.
    """

    inner_circle: float = 2 / 3
    outer_circle: float = 2 / 3 + 4 * 0.03
    num_circles: int = 5

    def __init__(
        self,
        inner_circle: float = 2 / 3,
        outer_circle: float = 2 / 3 + 4 * 0.03,
        num_circles: int = 5,
        spline_error_fraction: float = 0.05,
    ):
        """Configure temporal-angle sampling circles."""
        super().__init__(grid_field_spec=None)
        self.inner_circle = float(inner_circle)
        self.outer_circle = float(outer_circle)
        self.num_circles = int(num_circles)
        self.spline_error_fraction = float(spline_error_fraction)
        if self.num_circles < 1:
            raise ValueError("num_circles must be at least 1")
        if self.outer_circle < self.inner_circle:
            raise ValueError(
                "outer_circle must be greater than or equal to inner_circle"
            )

    def get_circle(self, layer: VesselTreeLayer, fractional_distance: float = 0.5):
        disc = layer.retina.disc
        assert disc is not None

        disc_center = disc.center_of_mass
        radius = (
            disc_center.distance_to(layer.retina.fovea_location) * fractional_distance
        )

        circle = Circle(center=disc_center, r=radius)
        return circle

    def get_circle_fractions(self) -> list[float]:
        """Return the OD–fovea distance fractions used for sampling."""
        return np.linspace(
            self.inner_circle, self.outer_circle, num=self.num_circles
        ).tolist()

    def _get_disc_fovea_axis(self, layer: VesselTreeLayer) -> Line:
        disc = layer.retina.disc
        fovea = layer.retina.fovea_location
        if disc is None or fovea is None:
            raise ValueError("Disc and fovea location are required for temporal angles")
        return Line(disc.center_of_mass, fovea)

    def _get_circle_mask(self, layer: VesselTreeLayer, circle: Circle) -> np.ndarray:
        retina = layer.retina
        disc = retina.disc
        if (
            disc is not None
            and np.isclose(circle.center.y, disc.center_of_mass.y)
            and np.isclose(circle.center.x, disc.center_of_mass.x)
            and getattr(retina, "disc_center_dist_sq", None) is not None
        ):
            return retina.disc_center_dist_sq <= circle.r**2

        yy_xx = getattr(retina, "yy_xx", None)
        if yy_xx is None:
            h, w = retina.resolution
            yy_xx = (np.arange(h)[:, None], np.arange(w)[None, :])
        yy, xx = yy_xx
        return (yy - circle.center.y) ** 2 + (xx - circle.center.x) ** 2 <= circle.r**2

    def _get_temporal_half_plane_mask(self, layer: VesselTreeLayer) -> np.ndarray:
        retina = layer.retina
        disc = retina.disc
        if disc is None:
            raise ValueError("Disc is required for temporal angles")

        yy_xx = getattr(retina, "yy_xx", None)
        if yy_xx is None:
            h, w = retina.resolution
            yy_xx = (np.arange(h)[:, None], np.arange(w)[None, :])
        yy, xx = yy_xx

        axis = np.array(
            [
                disc.temporal_point.y - disc.center_of_mass.y,
                disc.temporal_point.x - disc.center_of_mass.x,
            ],
            dtype=float,
        )
        return (
            (yy - disc.nasal_point.y) * axis[0] + (xx - disc.nasal_point.x) * axis[1]
        ) >= 0.0

    def get_valid_region_mask(
        self, layer: VesselTreeLayer, circle: Circle
    ) -> np.ndarray:
        return self._get_circle_mask(
            layer, circle
        ) & self._get_temporal_half_plane_mask(layer)

    def circle_region_in_bounds(self, layer: VesselTreeLayer, circle: Circle) -> bool:
        roi_mask = layer.retina.roi_mask
        if roi_mask is None:
            return True

        region_mask = self.get_valid_region_mask(layer, circle)
        return not np.any(region_mask & ~roi_mask.astype(bool))

    def _point_in_mask(self, point: Point, mask: np.ndarray) -> bool:
        y = int(np.rint(point.y))
        x = int(np.rint(point.x))
        if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
            return False
        return bool(mask[y, x])

    def _is_valid_temporal_intersection(
        self, layer: VesselTreeLayer, point: Point
    ) -> bool:
        axis = self._get_disc_fovea_axis(layer)
        angle = axis.angle_to(Line(layer.retina.disc.nasal_point, point))
        return angle < 90  # and self._point_in_mask(point, region_mask)

    def get_filtered_segments(self, layer: VesselTreeLayer, circle: Circle):
        # to speed it up only at the segments with one endpoint inside and one outside of the circle
        filtered_segments = [
            seg
            for seg in layer.get_resolved_segments(self.spline_error_fraction)
            if (circle.contains(seg.start) and not circle.contains(seg.end))
            or (not circle.contains(seg.start) and circle.contains(seg.end))
        ]
        return filtered_segments

    def get_intersections(self, layer: VesselTreeLayer, circle: Circle):
        if not self.circle_region_in_bounds(layer, circle):
            return []

        filtered_segments = self.get_filtered_segments(layer, circle)

        # for now we measure on the segments directly
        pairs = []
        for segment in filtered_segments:
            spline = segment.get_spline(error_fraction=self.spline_error_fraction)
            if spline is None:
                continue

            _, intersection_ts = segment.intersections_with_circle(
                circle, error_fraction=self.spline_error_fraction
            )
            if intersection_ts is None:
                continue

            t = float(intersection_ts[0])
            point = Point(*spline.get_point(t))
            if self._is_valid_temporal_intersection(layer, point):
                pairs.append((segment, t))
        return pairs

    def get_pair(
        self, layer: VesselTreeLayer, intersections: List[Tuple[Segment, float]]
    ):
        od_center = layer.retina.disc.center_of_mass
        pairs = []
        for (s1, t1), (s2, t2) in combinations(intersections, 2):
            p1 = Point(
                *s1.get_spline(error_fraction=self.spline_error_fraction).get_point(t1)
            )
            p2 = Point(
                *s2.get_spline(error_fraction=self.spline_error_fraction).get_point(t2)
            )
            angle = Line(od_center, p1).angle_to(Line(od_center, p2))

            if 75 < angle < 200:
                mean_diameter = s1.get_mean_diameter(
                    error_fraction=self.spline_error_fraction
                ) + s2.get_mean_diameter(error_fraction=self.spline_error_fraction)
                pairs.append((p1, p2, angle, mean_diameter))

        if len(pairs) > 0:
            return sorted(pairs, key=lambda x: x[3])[-1]
        else:
            return None

    def plot_filtered_segments(self, layer: VesselTreeLayer, **kwargs):
        circle = self.get_circle(layer, self.inner_circle)
        segments = self.get_filtered_segments(layer, circle)
        ax = layer.retina.plot(av=False)
        Vessels(layer, segments).plot(
            **{
                "show_index": True,
                "cmap": "tab20",
                "ax": ax,
                **kwargs,
            },
        )

    def compute(self, layer: VesselTreeLayer, **kwargs):
        angles = []
        for fractional_distance in self.get_circle_fractions():
            circle = self.get_circle(layer, fractional_distance)
            if not self.circle_region_in_bounds(layer, circle):
                continue
            intersections = self.get_intersections(layer, circle)
            pair = self.get_pair(layer, intersections)
            if pair is not None:
                angles.append(pair[2])

        if len(angles) > 0:
            return np.median(angles)
        else:
            warnings.warn("Couldn't find valid angles for any circle.")
            return None

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_layer_suffix

        layer = get_layer_suffix(layer_name)
        return f"Median Temporal Angle{layer}"

    def name_prefix_tokens(self) -> list[str]:
        return ["median"]

    def feature_name_tokens(self) -> list[str]:
        return ["temporal", "angle"]

    def parameter_name_tokens(self) -> list[str]:
        from .base import format_name_value

        tokens: list[str] = []
        if self.inner_circle != 2 / 3:
            tokens.extend(["inner_circle", format_name_value(self.inner_circle)])
        if self.outer_circle != 2 / 3 + 4 * 0.03:
            tokens.extend(["outer_circle", format_name_value(self.outer_circle)])
        if self.num_circles != 5:
            tokens.extend(["num_circles", format_name_value(self.num_circles)])
        if self.spline_error_fraction != 0.05:
            tokens.extend(
                [
                    "spline_error_fraction",
                    format_name_value(self.spline_error_fraction),
                ]
            )
        return tokens

    def _plot(self, ax, layer, **kwargs):
        pairs, segments, circles = [], [], []

        for fractional_distance in self.get_circle_fractions():
            circle = self.get_circle(layer, fractional_distance)
            circles.append(circle)
            if not self.circle_region_in_bounds(layer, circle):
                continue
            intersections = self.get_intersections(layer, circle)
            segments += [p[0] for p in intersections]
            pair = self.get_pair(layer, intersections)
            if pair is not None:
                pairs.append(pair)

        segments = list(set(segments))
        Vessels(layer, segments).plot(
            **{
                "show_index": True,
                "segments": True,
                "bounds": True,
                "ax": ax,
                **kwargs,
            },
        )

        largest_circle = self.get_circle(layer, self.outer_circle)
        largest_region_mask = self.get_valid_region_mask(layer, largest_circle)
        overlay = np.zeros((*largest_region_mask.shape, 4), dtype=float)
        overlay[..., 1] = 1.0
        overlay[..., 3] = largest_region_mask.astype(float) * 0.12
        ax.imshow(overlay)

        od = layer.retina.disc.center_of_mass

        for circle in circles:
            # ax.add_patch(

            # )
            circle = plt.Circle(
                circle.center.tuple_xy, circle.r, color="w", fill=False, lw=0.5
            )
            circle.set_clip_path(ax.patch)  # ensure anything outside axes isn't drawn
            ax.add_artist(circle)  # does not update data limits

        for pair in pairs:
            ax.plot(
                [pair[0].x, od.x],
                [pair[0].y, od.y],
                marker=None,
                linewidth=0.2,
                color="green",
            )
            ax.plot(
                [pair[1].x, od.x],
                [pair[1].y, od.y],
                marker=None,
                linewidth=0.2,
                color="green",
            )
        return ax

    def plot_intersections(self, layer: VesselTreeLayer, **kwargs):
        circle = self.get_circle(layer, self.inner_circle)
        intersections = self.get_intersections(layer, circle)

        segments = [p[0] for p in intersections]
        points = [
            p[0].get_spline(error_fraction=self.spline_error_fraction).get_point(p[1])
            for p in intersections
        ]

        ax = layer.retina.plot(av=False)
        Vessels(layer, segments).plot(
            **{
                "show_index": True,
                "cmap": "tab20",
                "ax": ax,
                **kwargs,
            },
        )

        ax.add_patch(
            plt.Circle(circle.center.tuple_xy, circle.r, color="w", fill=False, lw=0.5)
        )

        for p in points:
            ax.scatter(x=p[1], y=p[0], c="blue", s=1)
