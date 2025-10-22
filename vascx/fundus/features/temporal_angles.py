from __future__ import annotations

import warnings
from dataclasses import dataclass
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


@dataclass
class TemporalAngle(LayerFeature):
    """Median angle between the two dominant (largest caliber) temporal arcades at circles from 2/3 OD–fovea distance outward.

    Representation: Uses resolved_segments with circle intersections and OD–fovea spatial geometry. 
    Operates on the resolved vessel graph to identify major temporal vessel arcades.

    Computation: At concentric circles starting from 2/3 of the OD-fovea distance, identifies the two 
    largest-caliber temporal vessels (fovea-side, angle < 90°) crossing each circle, measures the angle 
    between them, and returns the median angle across all sampled radii. Captures the characteristic 
    temporal arcade geometry.

    Options: od_to_fovea_fraction (starting distance as fraction of OD-fovea distance), 
    increment (spacing between concentric sampling circles).
    """
    
    od_to_fovea_fraction: float = 2 / 3
    increment: float = 0.03

    def __init__(self, od_to_fovea_fraction: float = 2 / 3, increment: float = 0.03):
        """Configure starting fraction of OD–fovea distance and concentric circle spacing."""
        self.od_to_fovea_fraction = float(od_to_fovea_fraction)
        self.increment = float(increment)

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
            f"TemporalAngle(od_to_fovea_fraction={fmt(self.od_to_fovea_fraction)}, "
            f"increment={fmt(self.increment)})"
        )

    def get_circle(self, layer: VesselTreeLayer, fractional_distance=0.5):
        disc = layer.retina.disc
        assert disc is not None

        disc_center = disc.center_of_mass
        radius = (
            disc_center.distance_to(layer.retina.fovea_location) * fractional_distance
        )

        circle = Circle(center=disc_center, r=radius)
        return circle

    def get_filtered_segments(self, layer: VesselTreeLayer, circle: Circle):
        # to speed it up only at the segments with one endpoint inside and one outside of the circle
        filtered_segments = [
            seg
            for seg in layer.resolved_segments
            if (circle.contains(seg.start) and not circle.contains(seg.end))
            or (not circle.contains(seg.start) and circle.contains(seg.end))
        ]

        filtered_segments = [
            seg
            for seg in filtered_segments
            if seg.fod_angle() is not None and seg.fod_angle() < 90
        ]
        return filtered_segments

    def get_intersections(self, layer: VesselTreeLayer, circle: Circle):
        filtered_segments = self.get_filtered_segments(layer, circle)

        # for now we measure on the segments directly
        pairs = []
        for segment in filtered_segments:
            _, intersection_ts = segment.intersections_with_circle(circle)
            if intersection_ts is not None:
                for t in intersection_ts:
                    pairs.append((segment, t))
        return pairs

    def get_pair(
        self, layer: VesselTreeLayer, intersections: List[Tuple[Segment, float]]
    ):
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

    def plot_filtered_segments(self, layer: VesselTreeLayer, **kwargs):
        circle = self.get_circle(layer, self.od_to_fovea_fraction)
        segments = self.get_filtered_segments(layer, circle)
        fig, ax = layer.retina.plot_fundus()
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

    def plot(self, ax, layer, **kwargs):
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
        Vessels(layer, segments).plot(
            **{
                "show_index": True,
                "ax": ax,
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
        return ax

    def plot_intersections(self, layer: VesselTreeLayer, **kwargs):
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
                **kwargs,
            },
        )

        ax.add_patch(
            plt.Circle(circle.center.tuple_xy, circle.r, color="w", fill=False, lw=0.5)
        )

        for p in points:
            ax.scatter(x=p[1], y=p[0], c="blue", s=1)
