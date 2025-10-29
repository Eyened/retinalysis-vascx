from __future__ import annotations

from enum import Enum
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from rtnls_enface.base import Circle, LayerType
from rtnls_enface.grids.hemifields import HemifieldGrid, HemifieldField
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


class CREMode(str, Enum):
    Nasal = "nasal"
    Temporal = "temporal"
    Full = "full"

class CRE(LayerFeature):
    """Central retinal equivalents with temporal/nasal/full modes and optional hemifield filtering.

    Representation: Uses circle–segment intersections around the optic disc and each segment's `median_diameter`.

    Computation: Across concentric radii around the disc, identifies intersecting segments, optionally filters by
    a superior/inferior hemifield, retains up to the largest `max_vessels` by `median_diameter`, and recursively
    combines diameters using the Hubbard formula (sqrt(d1² + d2²) scaled by artery/vein constants). Returns the
    median equivalent diameter across radii.
    """

    def __init__(
        self,
        CREMode: CREMode = CREMode.Temporal,
        max_vessels: int = 6,
        hemifield: HemifieldField | None = None,
        min_circles: int = 6,
    ):
        self.CREMode = CREMode
        self.max_vessels = max_vessels
        self.hemifield: HemifieldField | None = hemifield
        self.min_circles: int = int(min_circles)
        
    def __repr__(self) -> str:
        return "CRE()"

    def get_circle(self, layer: VesselTreeLayer, od_multiple=0.5):
        disc = layer.retina.disc
        assert disc is not None

        disc_center = disc.center_of_mass
        radius = 2 * disc.circle.r * (0.5 + od_multiple)

        circle = Circle(center=disc_center, r=radius)
        return circle

    def __get_binary_mask(self, layer: "VesselTreeLayer", circle: Circle) -> np.ndarray:
        """Boolean mask of circle ∧ FOD-angle constraint ∧ hemifield (if set)."""
        retina = layer.retina
        h, w = retina.resolution

        cy, cx = circle.center.tuple
        yy = np.arange(h)[:, None]
        xx = np.arange(w)[None, :]
        circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= circle.r ** 2

        # FOD per-pixel angle mask
        if getattr(retina, "fovea_location", None) is None or getattr(retina, "disc", None) is None:
            fod_mask = np.ones_like(circle_mask, dtype=bool)
        else:
            od = retina.disc.center_of_mass
            fy, fx = retina.fovea_location.tuple
            vy, vx = fy - od.y, fx - od.x
            norm_v = np.hypot(vx, vy) + 1e-6
            dy = yy - od.y
            dx = xx - od.x
            norm_p = np.hypot(dx, dy) + 1e-6
            cosang = (dx * vx + dy * vy) / (norm_p * norm_v)
            cosang = np.clip(cosang, -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))

            if self.CREMode == CREMode.Temporal:
                fod_mask = ang < 100.0
            elif self.CREMode == CREMode.Nasal:
                fod_mask = ang > 80.0
            else:
                fod_mask = np.ones_like(circle_mask, dtype=bool)

        mask = circle_mask & fod_mask

        # Optional hemifield filtering
        if self.hemifield is not None:
            try:
                base_mask = retina.mask
            except Exception:
                base_mask = None
            grid = HemifieldGrid(retina, resolution=retina.resolution, mask=base_mask)
            hemi_mask = grid.field(self.hemifield).mask.astype(bool)
            mask &= hemi_mask

        return mask

    def get_filtered_segments(self, layer: VesselTreeLayer, circle: Circle):
        # to speed it up only at the segments with one endpoint inside and one outside of the circle
        filtered_segments = [
            seg
            for seg in layer.segments
            if (circle.contains(seg.start) and not circle.contains(seg.end))
            or (not circle.contains(seg.start) and circle.contains(seg.end))
        ]

        # Orientation filtering kept as-is
        if self.CREMode == CREMode.Temporal:
            filtered_segments = [
                seg
                for seg in filtered_segments
                if seg.fod_angle() is not None and seg.fod_angle() < 100
                if seg.orientation() is not None and seg.orientation() < 90
            ]
        elif self.CREMode == CREMode.Nasal:
            filtered_segments = [
                seg
                for seg in filtered_segments
                if seg.fod_angle() is not None and seg.fod_angle() > 80
                if seg.orientation() is not None and seg.orientation() > 90
            ]
        else:
            pass

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
        if layer.name == 'arteries':
            cte = 0.88
        elif layer.name == 'veins':
            cte = 0.95
        else:
            raise ValueError("Unrecognized layer type for CRE computation")
        # Build mask and enforce full containment within retina bounds
        mask = self.__get_binary_mask(layer, circle.resize(layer.retina.disc.circle.r / 5.0))
        total = int(np.count_nonzero(mask))
        if total == 0:
            return None, []
        try:
            retina_mask = layer.retina.mask.astype(bool)
        except Exception:
            retina_mask = None
        if retina_mask is not None:
            in_bounds = np.count_nonzero(mask & retina_mask)
            if in_bounds < total:
                # Discard this circle if any part is out of bounds
                return None, []

        intersections = self.get_intersections(layer, circle)
        # Filter intersections by on-mask (True) values
        h, w = mask.shape
        filtered_intersections: List[Tuple[Segment, float]] = []
        for seg, t in intersections:
            y, x = seg.spline.get_point(t)
            yi, xi = int(round(y)), int(round(x))
            if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
                filtered_intersections.append((seg, t))
        intersections = filtered_intersections
        if len(intersections) == 0:
            return None, []
        # Deduplicate segments that may intersect circle multiple times
        segments = list(set([p[0] for p in intersections]))
        if len(segments) == 0:
            warnings.warn("Could not find intersecting segments for CRE circle.")
            return None, []

        # Keep up to the largest N segments by median diameter
        segments.sort(key=lambda s: s.median_diameter, reverse=True)
        segments = segments[: self.max_vessels]

        calibers = [s.median_diameter for s in segments]
        return self.recursive_cre(calibers, cte), intersections

    

    def compute(self, layer: VesselTreeLayer):
        cres = []
        for i in range(0, 6):
            circle = self.get_circle(layer, 0.5 + 0.1 * i)

            cre, _ = self.compute_cre_for_circle(layer, circle)
            if cre is not None:
                cres.append(cre)

        if len(cres) < self.min_circles:
            return None
        else:
            return float(np.mean(cres))

    def _plot(self, ax, layer: VesselTreeLayer, **kwargs):
        """This plot shows the circles used to compute CRE,
        the segments used in the CRE computation and the CRE next to each circle
        """
        segments, circles, cres, points = [], [], [], []
        # Optionally overlay hemifield axis
        if self.hemifield is not None:
            retina = layer.retina
            try:
                mask = retina.mask
            except Exception:
                mask = None
            grid = HemifieldGrid(retina, resolution=retina.resolution, mask=mask)
            grid.plot(ax)
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
        layer.retina.plot(ax=ax, image=True, bounds=True)
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

        # for i, cre in enumerate(reversed(cres)):
        #     ax.text(10, 20 + 30 * i, f"{cre:.2f}", color="white", fontsize=6)

        for p in points:
            ax.scatter(x=p[1], y=p[0], c="white", s=2, marker="x")

        # Overlay largest circle's mask as a transparent layer
        try:
            largest_circle = self.get_circle(layer, 0.5 + 0.1 * 5)
            mask = self.__get_binary_mask(layer, largest_circle.resize(layer.retina.disc.circle.r / 5.0))
            h, w = layer.retina.resolution
            overlay = np.zeros((h, w, 4), dtype=float)
            overlay[mask] = [0.0, 1.0, 0.0, 0.25]
            ax.imshow(overlay)
        except Exception:
            pass

        return ax
