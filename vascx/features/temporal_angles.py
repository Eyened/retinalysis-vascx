from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from rtnls_enface.types import Circle, Line, Point

from vascx.vessels import Vessels

from .base import LayerFeature

if TYPE_CHECKING:
    from vascx.retina import VesselLayer
    from vascx.segment import Segment


@dataclass
class TemporalAngle(LayerFeature):
    od_to_fovea_fraction: float = 2 / 3
    increment: float = 0.03

    def get_circle(self, layer: VesselLayer, fractional_distance=0.5):
        disc = layer.retina.disc
        assert disc is not None

        disc_center = disc.center_of_mass
        radius = (
            disc_center.distance_to(layer.retina.fovea_location) * fractional_distance
        )

        circle = Circle(center=disc_center, r=radius)
        return circle

    def get_filtered_segments(self, layer: VesselLayer, circle: Circle):
        # to speed it up only at the segments with one endpoint inside and one outside of the circle
        filtered_segments = [
            seg
            for seg in layer.vessels.segments
            if (circle.contains(seg.start) and not circle.contains(seg.end))
            or (not circle.contains(seg.start) and circle.contains(seg.end))
        ]

        filtered_segments = [
            seg
            for seg in filtered_segments
            if seg.fod_angle() is not None and seg.fod_angle() < 90
        ]
        return filtered_segments

    def get_intersections(self, layer: VesselLayer, circle: Circle):
        filtered_segments = self.get_filtered_segments(layer, circle)

        # for now we measure on the segments directly
        pairs = []
        for segment in filtered_segments:
            _, intersection_ts = segment.intersections_with_circle(circle)
            if intersection_ts is not None:
                for t in intersection_ts:
                    pairs.append((segment, t))
        return pairs

    def get_pair(self, layer: VesselLayer, intersections: List[Tuple[Segment, float]]):
        od_center = layer.disc.center_of_mass
        pairs = []
        for (s1, t1), (s2, t2) in combinations(intersections, 2):
            p1 = Point(*s1.spline.get_point(t1))
            p2 = Point(*s2.spline.get_point(t2))
            angle = Line(od_center, p1).angle_to(Line(od_center, p2))

            if 75 < angle < 200:
                mean_diameter = s1.mean_diameter + s2.mean_diameter
                pairs.append((p1, p2, angle, mean_diameter))

        if len(pairs) > 0:
            return sorted(pairs, key=lambda x: x[3])[-1]
        else:
            return None

    def plot_filtered_segments(self, layer: VesselLayer, **kwargs):
        circle = self.get_circle(layer, self.od_to_fovea_fraction)
        segments = self.get_filtered_segments(layer, circle)
        fig, ax = layer.retina.plot_fundus()
        Vessels(layer, segments).plot(
            **{
                "show_index": True,
                "cmap": "tab20",
                "ax": ax,
                "fig": fig,
                **kwargs,
            },
        )

    def plot(self, layer, fig=None, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()
            ax.imshow(np.zeros_like(layer.binary))

        pairs, segments, circles = [], [], []

        for i in range(0, 5):
            circle = self.get_circle(
                layer, self.od_to_fovea_fraction + i * self.increment
            )
            intersections = self.get_intersections(layer, circle)
            segments += [p[0] for p in intersections]
            circles.append(circle)
            pair = self.get_pair(layer, intersections)
            if pair is not None:
                pairs.append(pair)

        segments = list(set(segments))
        layer.retina.plot_fundus(ax=ax)
        Vessels(layer, segments).plot(
            **{
                "show_index": True,
                "cmap": "tab20",
                "ax": ax,
                "fig": fig,
                **kwargs,
            },
        )

        od = layer.retina.disc.center_of_mass

        for circle in circles:
            ax.add_patch(
                plt.Circle(
                    circle.center.tuple_xy, circle.r, color="w", fill=False, lw=0.5
                )
            )

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
        return fig, ax

    def plot_intersections(self, layer: VesselLayer, **kwargs):
        circle = self.get_circle(layer, 0.5)
        intersections = self.get_intersections(layer, circle)

        segments = [p[0] for p in intersections]
        points = [p[0].spline.get_point(p[1]) for p in intersections]

        fig, ax = layer.retina.plot_fundus()
        Vessels(layer, segments).plot(
            **{
                "show_index": True,
                "cmap": "tab20",
                "ax": ax,
                "fig": fig,
                **kwargs,
            },
        )

        ax.add_patch(
            plt.Circle(circle.center.tuple_xy, circle.r, color="w", fill=False, lw=0.5)
        )

        for p in points:
            ax.scatter(x=p[1], y=p[0], c="blue", s=1)

    def compute(self, layer: VesselLayer, **kwargs):
        angles = []
        for i in range(0, 5):
            circle = self.get_circle(
                layer, self.od_to_fovea_fraction + i * self.increment
            )
            intersections = self.get_intersections(layer, circle)
            pair = self.get_pair(layer, intersections)
            if pair is not None:
                angles.append(pair[2])

        if len(angles) > 0:
            return np.median(angles)
        else:
            warnings.warn("Couldn't find valid angles for any circle.")
            return None
