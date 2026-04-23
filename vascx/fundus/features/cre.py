from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from rtnls_enface.base import Circle
from rtnls_enface.grids.hemifields import HemifieldField
from rtnls_enface.grids.specifications import (
    GridFieldSpecification,
    HemifieldGridSpecification,
)

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

    Representation: uses circle–segment intersections around the optic disc and each segment's
    `median_diameter`.

    Computation: across concentric radii around the disc, identifies intersecting segments, optionally
    filters by a superior/inferior hemifield, retains up to the largest `max_vessels` by `median_diameter`,
    and recursively combines diameters using the Hubbard reduction (√(d₁² + d₂²)) scaled by artery/vein
    constants (c=0.88 for arteries, c=0.95 for veins). Returns the median equivalent diameter across radii.

    Args (constructor):
    - CREMode: `CREMode` selection for temporal, nasal, or full orientation constraint.
    - max_vessels: keep up to this many largest-caliber intersecting segments per circle.
    - hemifield: optional `HemifieldField` to restrict to superior or inferior hemifield.
    - min_circles: minimum number of valid circles required; else returns None.
    - inner_circle: inner CRE circle radius in optic-disc-diameter multiples.
    - outer_circle: outer CRE circle radius in optic-disc-diameter multiples.
    - num_circles: total number of circles sampled between inner and outer radii, inclusive.

    Notes: each circle must be fully within the retinal mask; if any part is out-of-bounds, that circle
    is discarded from the aggregation for robustness.
    """

    def __init__(
        self,
        CREMode: CREMode = CREMode.Temporal,
        max_vessels: int = 6,
        hemifield: Optional[HemifieldField] = None,
        min_circles: int = 6,
        inner_circle: float = 1.0,
        outer_circle: float = 1.5,
        num_circles: int = 6,
        spline_error_fraction: float = 0.05,
    ):
        self.CREMode = CREMode
        self.max_vessels = max_vessels
        self.inner_circle = float(inner_circle)
        self.outer_circle = float(outer_circle)
        self.num_circles = int(num_circles)
        self.spline_error_fraction = float(spline_error_fraction)
        if self.num_circles < 1:
            raise ValueError("num_circles must be at least 1")
        if self.outer_circle < self.inner_circle:
            raise ValueError("outer_circle must be greater than or equal to inner_circle")
        self.min_intersections = 4
        if self.CREMode in [CREMode.Nasal, CREMode.Temporal]:
            self.min_intersections = 2
        if hemifield is None:
            self.hemifield_spec = None
        else:
            self.hemifield_spec = GridFieldSpecification(
                HemifieldGridSpecification(), hemifield
            )
            self.min_intersections //= 2
        self.min_circles: int = int(min_circles)

    def get_circle(self, layer: VesselTreeLayer, od_multiple: float = 1.0):
        disc = layer.retina.disc
        assert disc is not None

        disc_center = disc.center_of_mass
        radius = 2 * disc.circle.r * od_multiple

        circle = Circle(center=disc_center, r=radius)
        return circle

    def get_circle_multiples(self) -> list[float]:
        return np.linspace(
            self.inner_circle, self.outer_circle, num=self.num_circles
        ).tolist()

    def _binary_mask_cache_key(self, circle: Circle) -> tuple:
        """Build a stable cache key for a CRE binary mask."""
        cy, cx = circle.center.tuple
        hemifield = (
            None
            if self.hemifield_spec is None
            else getattr(
                self.hemifield_spec.field, "name", str(self.hemifield_spec.field)
            )
        )
        return (
            round(float(cy), 6),
            round(float(cx), 6),
            round(float(circle.r), 6),
            self.CREMode.value,
            hemifield,
        )

    def __get_binary_mask(self, layer: "VesselTreeLayer", circle: Circle) -> np.ndarray:
        """Boolean mask of circle ∧ FOD-angle constraint ∧ hemifield (if set)."""
        cache_key = self._binary_mask_cache_key(circle)
        cached_mask = layer._cre_binary_mask_cache.get(cache_key)
        if cached_mask is not None:
            return cached_mask

        retina = layer.retina
        cy, cx = circle.center.tuple
        disc_center = None if retina.disc is None else retina.disc.center_of_mass

        # Retina caches store reusable per-image geometry shared by CRE variants.
        if (
            disc_center is not None
            and np.isclose(cy, disc_center.y)
            and np.isclose(cx, disc_center.x)
            and retina.disc_center_dist_sq is not None
        ):
            circle_mask = retina.disc_center_dist_sq <= circle.r**2
        else:
            yy, xx = retina.yy_xx
            circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= circle.r**2

        angle_deg = retina.disc_fovea_angle_deg
        if angle_deg is None:
            fod_mask = np.ones_like(circle_mask, dtype=bool)
        else:
            if self.CREMode == CREMode.Temporal:
                fod_mask = angle_deg < 100.0
            elif self.CREMode == CREMode.Nasal:
                fod_mask = angle_deg > 80.0
            else:
                fod_mask = np.ones_like(circle_mask, dtype=bool)

        mask = circle_mask & fod_mask

        # The per-layer mask cache remains mode- and hemifield-specific.
        if self.hemifield_spec is not None:
            hemi_field = retina.get_grid_field(self.hemifield_spec)
            mask &= hemi_field.mask.astype(bool)

        layer._cre_binary_mask_cache[cache_key] = mask
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
        filtered_segments = set(self.get_filtered_segments(layer, circle))
        return [
            (segment, t)
            for segment, t in layer.get_circle_intersections(
                circle, spline_error_fraction=self.spline_error_fraction
            )
            if segment in filtered_segments
        ]

    def plot_filtered_segments(self, layer: VesselTreeLayer, **kwargs):
        circle = self.get_circle(layer, 7 / 6)
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
        if layer.name == "arteries":
            cte = 0.88
        elif layer.name == "veins":
            cte = 0.95
        else:
            raise ValueError("Unrecognized layer type for CRE computation")
        # Build mask and enforce full containment within retina bounds
        mask = self.__get_binary_mask(
            layer, circle.resize(layer.retina.disc.circle.r / 5.0)
        )
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
            y, x = seg.get_spline(error_fraction=self.spline_error_fraction).get_point(t)
            yi, xi = int(round(y)), int(round(x))
            if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
                filtered_intersections.append((seg, t))
        intersections = filtered_intersections
        if len(intersections) == 0:
            return None, []
        # Deduplicate segments that may intersect circle multiple times
        segments = list(set([p[0] for p in intersections]))
        if len(segments) < self.min_intersections:
            warnings.warn("Could not find enough intersecting segments for CRE circle.")
            return None, []

        # Keep up to the largest N segments by median diameter
        segments.sort(
            key=lambda s: s.get_median_diameter(self.spline_error_fraction),
            reverse=True,
        )
        segments = segments[: self.max_vessels]

        calibers = [s.get_median_diameter(self.spline_error_fraction) for s in segments]
        return self.recursive_cre(calibers, cte), intersections

    def compute(self, layer: VesselTreeLayer):
        cres = []
        for od_multiple in self.get_circle_multiples():
            circle = self.get_circle(layer, od_multiple)

            cre, _ = self.compute_cre_for_circle(layer, circle)
            if cre is not None:
                cres.append(cre)

        if len(cres) < self.min_circles:
            return None
        else:
            return float(np.mean(cres))

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_grid_field_suffix, get_layer_suffix

        field = get_grid_field_suffix(self.hemifield_spec)
        layer = get_layer_suffix(layer_name)
        mode = self.CREMode.name
        return f"{mode} CRE{field}{layer}"

    def name_prefix_tokens(self) -> list[str]:
        return [self.CREMode.value]

    def feature_name_tokens(self) -> list[str]:
        return ["cre"]

    def parameter_name_tokens(self) -> list[str]:
        from .base import format_name_value

        tokens: list[str] = []
        if self.max_vessels != 6:
            tokens.extend(["max_vessels", str(self.max_vessels)])
        if self.min_circles != 6:
            tokens.extend(["min_circles", str(self.min_circles)])
        if self.inner_circle != 1.0:
            tokens.extend(["inner_circle", str(self.inner_circle)])
        if self.outer_circle != 1.5:
            tokens.extend(["outer_circle", str(self.outer_circle)])
        if self.num_circles != 6:
            tokens.extend(["num_circles", str(self.num_circles)])
        if self.spline_error_fraction != 0.05:
            tokens.extend(
                [
                    "spline_error_fraction",
                    format_name_value(self.spline_error_fraction),
                ]
            )
        return tokens

    def name_tokens(self, layer_name: str, **kwargs) -> list[str]:
        from .base import get_grid_field_tokens, get_layer_tokens

        return [
            *self.name_prefix_tokens(),
            *self.feature_name_tokens(),
            *self.parameter_name_tokens(),
            *get_grid_field_tokens(self.hemifield_spec),
            *get_layer_tokens(layer_name),
        ]

    def _plot(self, ax, layer: VesselTreeLayer, **kwargs):
        """This plot shows the circles used to compute CRE,
        the segments used in the CRE computation and the CRE next to each circle
        """
        segments, circles, cres, points = [], [], [], []
        # Optionally overlay hemifield axis
        if self.hemifield_spec is not None:
            retina = layer.retina
            field = retina.get_grid_field(self.hemifield_spec)
            field.plot(ax)
        for od_multiple in self.get_circle_multiples():
            circle = self.get_circle(layer, od_multiple)

            cre, intersections = self.compute_cre_for_circle(layer, circle)
            if cre is None:
                continue
            segments += [p[0] for p in intersections]
            points += [
                p[0].get_spline(error_fraction=self.spline_error_fraction).get_point(
                    p[1]
                )
                for p in intersections
            ]
            circles.append(circle)
            cres.append(cre)

        segments = list(set(segments))
        layer.retina.plot(ax=ax, image=True, bounds=True, av=False)
        Vessels(layer, segments).plot(
            **{
                "show_index": True,
                "cmap": "tab20",
                "ax": ax,
                "segments": True,
                "image": False,
                **kwargs,
            },
        )

        for circle in circles:
            ax.add_patch(
                plt.Circle(
                    circle.center.tuple_xy, circle.r, color="w", fill=False, lw=0.5
                )
            )

        for p in points:
            ax.scatter(x=p[1], y=p[0], c="white", s=2, marker="x")

        ax.text(
            0.05,
            0.95,
            f"min_intersections={self.min_intersections}",
            transform=ax.transAxes,
            fontsize=6,
            color="white",
            ha="left",
            va="top",
        )

        # Overlay largest circle's mask as a transparent layer
        try:
            largest_circle = self.get_circle(layer, self.outer_circle)
            mask = self.__get_binary_mask(
                layer, largest_circle.resize(layer.retina.disc.circle.r / 5.0)
            )
            h, w = layer.retina.resolution
            overlay = np.zeros((h, w, 4), dtype=float)
            overlay[mask] = [0.0, 1.0, 0.0, 0.25]
            ax.imshow(overlay)
        except Exception:
            pass

        return ax
