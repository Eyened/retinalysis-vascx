from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from rtnls_enface.base import Circle, LayerType
from vascx.shared.segment import Segment
from vascx.shared.vessels import Vessels

from .base import LayerFeature

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


def recursive_cre(lst, cte):
    if len(lst) == 0:
        return None
    # Base case: if the list is reduced to a single element, return that element
    if len(lst) == 1:
        return lst[0]

    # Initialize a new list to store sums of pairs
    new_list = []

    # Calculate the middle index
    mid = len(lst) // 2

    # Add the first and last, second and second-to-last, etc.
    for i in range(
        mid + len(lst) % 2
    ):  # Adjust for odd-length lists by adding 1 if odd
        # If we're at the middle of an odd-length list, just append the middle element
        if len(lst) % 2 != 0 and i == mid:
            new_list.append(lst[i])
        else:
            new_list.append(cte * np.sqrt(lst[i] ** 2 + lst[-i - 1] ** 2))

    # Recursively call the function with the new list
    return recursive_cre(new_list, cte)


@dataclass
class CRE(LayerFeature):
    def get_circle(self, layer: VesselTreeLayer, od_multiple=0.5):
        disc = layer.retina.disc
        assert disc is not None

        disc_center = disc.center_of_mass
        radius = 2 * disc.circle.r * (0.5 + od_multiple)

        circle = Circle(center=disc_center, r=radius)
        return circle

    def get_filtered_segments(self, layer: VesselTreeLayer, circle: Circle):
        # to speed it up only at the segments with one endpoint inside and one outside of the circle
        filtered_segments = [
            seg
            for seg in layer.segments
            if (circle.contains(seg.start) and not circle.contains(seg.end))
            or (not circle.contains(seg.start) and circle.contains(seg.end))
        ]

        filtered_segments = [
            seg
            for seg in filtered_segments
            if seg.fod_angle() is not None
            and seg.fod_angle() < 100
            and seg.orientation() is not None
            and seg.orientation() < 90
        ]

        return filtered_segments

    def get_intersections(
        self, layer: VesselTreeLayer, circle: Circle
    ) -> List[Tuple[Segment, float]]:
        filtered_segments = self.get_filtered_segments(layer, circle)

        # for now we measure on the segments directly
        pairs = []
        for segment in filtered_segments:
            _, intersection_ts = segment.intersections_with_circle(circle)
            if intersection_ts is not None:
                for t in intersection_ts:
                    pairs.append((segment, t))
        return pairs

    def plot_filtered_segments(self, layer: VesselTreeLayer, **kwargs):
        circle = self.get_circle(layer, 2 / 3)
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

    def recursive_cre(self, calibers: List[float], cte: float):
        sc = sorted(calibers)
        return recursive_cre(sc, cte)

    def compute_cre_for_circle(self, layer: VesselTreeLayer, circle: Circle):
        if layer.type == LayerType.ARTERIES:
            cte = 0.88
        elif layer.type == LayerType.VEINS:
            cte = 0.95
        else:
            raise ValueError("Unrecognized layer type for CRE computation")

        intersections = self.get_intersections(layer, circle)
        segments = [p[0] for p in intersections]
        if len(segments) == 0:
            warnings.warn("Could not find intersecting segmentes for CRE circle.")
            return None, []

        # select the max four largest segments
        calibers = [s.median_diameter for s in segments]
        return self.recursive_cre(calibers, cte), intersections

    def plot(self, layer: VesselTreeLayer, fig=None, ax=None, **kwargs):
        """This plot shows the circles used to compute CRE,
        the segments used in the CRE computation and the CRE next to each circle
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()
            ax.imshow(np.zeros_like(layer.binary))

        segments, circles, cres, points = [], [], [], []
        for i in range(0, 6):
            circle = self.get_circle(layer, 0.5 + 0.1 * i)

            cre, intersections = self.compute_cre_for_circle(layer, circle)
            if cre is None:
                continue
            segments += [p[0] for p in intersections]
            points += [p[0].spline.get_point(p[1]) for p in intersections]
            circles.append(circle)
            cres.append(cre)

        segments = list(set(segments))
        layer.retina.plot_fundus(ax=ax)
        Vessels(layer, segments).plot(
            **{
                "show_index": True,
                "cmap": "tab20",
                "ax": ax,
                "text": lambda s: f"{s.orientation():.2f}",
                "plot_endpoints": True,
                **kwargs,
            },
        )

        for circle in circles:
            ax.add_patch(
                plt.Circle(
                    circle.center.tuple_xy, circle.r, color="w", fill=False, lw=0.5
                )
            )

        for i, cre in enumerate(reversed(cres)):
            ax.text(10, 20 + 30 * i, f"{cre:.2f}", color="white", fontsize=6)

        for p in points:
            ax.scatter(x=p[1], y=p[0], c="green", s=1)

        return fig, ax

    def compute(self, layer: VesselTreeLayer, fig=None, ax=None, **kwargs):
        cres = []
        for i in range(0, 6):
            circle = self.get_circle(layer, 0.5 + 0.1 * i)

            cre, _ = self.compute_cre_for_circle(layer, circle)
            if cre is not None:
                cres.append(cre)

        if len(cres) == 0:
            return None
        else:
            return np.median(cres).item()
