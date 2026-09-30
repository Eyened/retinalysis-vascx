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
from ._region_coverage import circle_region_in_bounds

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


def recursive_cre(lst, cte):
    """Combine smallest/largest calibers, re-sorting at every reduction round."""
    if len(lst) == 0:
        return None
    # Base case: if the list is reduced to a single element, return that element
    if len(lst) == 1:
        return lst[0]

    lst = sorted(lst)

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


class CREMeasure(str, Enum):
    Circles = "circles"
    ARIC = "aric"


class CRE(LayerFeature):
    """Central retinal equivalents for full, temporal, nasal, superior, or inferior regions.

    The default ``measure=CREMeasure.Circles`` retains the historical calculation.
    ``measure=CREMeasure.ARIC`` selects trunks in Zone B and combines their
    safeguarded whole-segment median calibers once. See ``get_aric_result`` for QC.

    Representation (Circles): uses circle–segment intersections around the optic disc and each segment's
    `median_diameter`.

    Computation: across concentric radii around the disc, identifies intersecting segments, optionally
    filters by a superior/inferior hemifield, retains the largest `max_vessels` by `median_diameter`
    (mode defaults: 6 full, 4 temporal/nasal), and recursively combines diameters using the Knudtson pairwise formula (√(d₁² + d₂²)) scaled by artery/vein
    constants (c=0.88 for arteries, c=0.95 for veins). Returns the median equivalent diameter across radii.

    Args (constructor):
    - CREMode: `CREMode` selection for temporal, nasal, or full orientation constraint.
    - max_vessels: keep up to this many largest-caliber intersecting segments per circle.
    - hemifield: superior/inferior restriction, supported only with `CREMode.Full`.
    - min_circles: minimum number of valid circles required; else returns None.
    - inner_circle: inner CRE circle radius in optic-disc-diameter multiples.
    - outer_circle: outer CRE circle radius in optic-disc-diameter multiples.
    - num_circles: total number of circles sampled between inner and outer radii, inclusive.
    - measure: Circles (default) or ARIC. Circle-count parameters apply only to Circles.
    - min_vessels: ARIC minimum count; defaults to 4 for full and 2 for regional
      measurements, capped by max_vessels. ARIC defaults to at most 6 full or 3
      regional trunks, flags fewer-than-target counts, and preserves usable short trunks.
    - min_area_within_bounds: ARIC minimum visible fraction of the requested annulus;
      defaults to 1.0. This conservative automatic QC rule is a VascX addition.

    Notes: each circle must be fully within the retinal mask; if any part is out-of-bounds, that circle
    is discarded from the aggregation for robustness.
    """

    general_description = (
        "Central retinal equivalent caliber summarizes the width of the major "
        "retinal vessels."
    )

    def implementation_description(self, layer_name: str = "vessels", **kwargs) -> str:
        from .base import get_layer_description

        layer = get_layer_description(layer_name)
        region = {
            CREMode.Temporal: "temporal",
            CREMode.Nasal: "nasal",
            CREMode.Full: "",
        }[self.CREMode]
        qualifier = f"{region} " if region else ""
        if self.measure == CREMeasure.ARIC:
            return (
                f"Calculated from up to {self.max_vessels} largest eligible {qualifier}{layer} "
                "trunks selected in the measurement annulus, using the standard VascX "
                "whole-segment median widths and the Knudtson pairwise formula. "
                "Calibers include measurements outside the annulus and assume minor tapering. "
                "Ungradable trunks may be "
                "replaced by their daughters with a quality flag; parents and daughters "
                "are never counted together. This is an automated ARIC-style measurement."
            )
        return (
            f"Calculated by combining the largest {qualifier}{layer} vessels crossing "
            "concentric circles centered on the optic disc using an established "
            "equivalent-caliber formula."
        )

    def aggregation_description(self, **kwargs) -> str:
        if self.measure == CREMeasure.ARIC:
            return (
                f"Reported as one equivalent caliber from {self.required_aric_vessels} "
                f"to {self.max_vessels} eligible vessels; no averaging across circles."
            )
        return "Reported as the median across valid measurement circles."

    def __init__(
        self,
        CREMode: CREMode = CREMode.Temporal,
        max_vessels: Optional[int] = None,
        hemifield: Optional[HemifieldField] = None,
        min_circles: int = 2,
        inner_circle: float = 1.0,
        outer_circle: float = 1.5,
        num_circles: int = 5,
        spline_error_fraction: float = 0.05,
        plot: bool = False,
        *,
        measure: CREMeasure = CREMeasure.Circles,
        min_vessels: Optional[int] = None,
        min_area_within_bounds: float = 1.0,
    ):
        super().__init__(grid_field_spec=None, plot=plot)
        self.CREMode = CREMode
        self.measure = CREMeasure(measure)
        self.min_vessels = min_vessels
        self.min_area_within_bounds = float(min_area_within_bounds)
        
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
        if (self.CREMode != "full"
                and hemifield in (HemifieldField.Superior, HemifieldField.Inferior)):
            raise ValueError(
                "CRE does not support temporal/nasal combined with superior/inferior; "
                "use CREMode.Full with hemifield for a standalone superior/inferior region"
            )
        if hemifield is None:
            self.hemifield_spec = None
        else:
            self.hemifield_spec = GridFieldSpecification(
                HemifieldGridSpecification(), hemifield
            )
        self.min_circles: int = int(min_circles)
        self.max_vessels = (
            self.default_max_vessels()
            if max_vessels is None
            else int(max_vessels)
        )

        if self.measure == CREMeasure.ARIC:
            if not (np.isfinite(self.inner_circle) and np.isfinite(self.outer_circle)
                    and 0 < self.inner_circle < self.outer_circle):
                raise ValueError("ARIC requires finite 0 < inner_circle < outer_circle")
            if self.max_vessels < 1:
                raise ValueError("max_vessels must be positive")
            if min_vessels is not None and (
                isinstance(min_vessels, bool) or int(min_vessels) != min_vessels
                or not 1 <= min_vessels <= self.max_vessels
            ):
                raise ValueError("min_vessels must be an integer between 1 and max_vessels")
            if not 0 <= self.min_area_within_bounds <= 1:
                raise ValueError("min_area_within_bounds must be between 0 and 1")

    @property
    def required_aric_vessels(self) -> int:
        if self.min_vessels is not None:
            return int(self.min_vessels)
        return min(2 if self.is_regional else 4, self.max_vessels)

    @property
    def is_regional(self) -> bool:
        """Whether a temporal, nasal, superior, or inferior restriction is active."""
        return self.CREMode != CREMode.Full or (
            self.hemifield_spec is not None
            and self.hemifield_spec.field in (HemifieldField.Superior, HemifieldField.Inferior)
        )

    def default_max_vessels(self) -> int:
        """Return the method- and region-specific default vessel count."""
        if self.measure == CREMeasure.ARIC:
            return 3 if self.is_regional else 6
        if self.CREMode == CREMode.Full:
            max_vessels = 6
        else:
            max_vessels = 4
        if self.hemifield_spec is not None and self.hemifield_spec.field in [HemifieldField.Superior, HemifieldField.Inferior]:
            max_vessels = max_vessels // 2
        return max_vessels
        

    def get_circle(self, layer: VesselTreeLayer, od_multiple: float = 1.0):
        disc = layer.retina.disc
        assert disc is not None

        disc_center = disc.center
        radius = 2 * disc.circle.r * od_multiple

        circle = Circle(center=disc_center, r=radius)
        return circle

    def get_circle_multiples(self) -> list[float]:
        return np.linspace(
            self.inner_circle, self.outer_circle, num=self.num_circles
        ).tolist()

    def _temporal_origin_and_vector(
        self, layer: "VesselTreeLayer"
    ) -> Optional[Tuple[float, float, float, float]]:
        """Return the shifted temporal origin and OD-to-fovea vector."""
        retina = layer.retina
        if retina.disc is None or retina.fovea_location is None:
            return None

        disc_center = retina.disc.center
        fovea = retina.fovea_location
        vy = fovea.y - disc_center.y
        vx = fovea.x - disc_center.x
        norm_v = np.hypot(vx, vy)
        if norm_v == 0:
            return None

        origin_y = disc_center.y - 0.5 * retina.disc.circle.r * vy / norm_v
        origin_x = disc_center.x - 0.5 * retina.disc.circle.r * vx / norm_v
        return origin_y, origin_x, vy, vx

    def _temporal_angle_deg(
        self,
        layer: "VesselTreeLayer",
        y: np.ndarray | float,
        x: np.ndarray | float,
    ) -> Optional[np.ndarray | float]:
        """Measure angle from the shifted temporal origin to image point(s)."""
        geometry = self._temporal_origin_and_vector(layer)
        if geometry is None:
            return None

        origin_y, origin_x, vy, vx = geometry
        dy = y - origin_y
        dx = x - origin_x
        norm_v = np.hypot(vx, vy)
        norm_p = np.hypot(dx, dy) + 1e-6
        cosang = (dx * vx + dy * vy) / (norm_p * norm_v)
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    def _temporal_fod_mask(self, layer: "VesselTreeLayer") -> Optional[np.ndarray]:
        """Return the shifted-origin temporal field mask."""
        yy, xx = layer.retina.yy_xx
        angle_deg = self._temporal_angle_deg(layer, yy, xx)
        if angle_deg is None:
            return None
        return angle_deg < 85.0

    def _segment_temporal_fod_angle(self, segment: Segment) -> Optional[float]:
        """Return the shifted-origin temporal angle for a segment midpoint."""
        point = segment.mean_position()
        angle_deg = self._temporal_angle_deg(segment.layer, point.y, point.x)
        if angle_deg is None:
            return None
        return float(angle_deg)

    def _is_temporal_segment(self, segment: Segment) -> bool:
        """Return whether a segment is eligible for temporal CRE."""
        temporal_angle = self._segment_temporal_fod_angle(segment)
        orientation = segment.orientation()
        return (
            temporal_angle is not None
            and temporal_angle < 85.0
            and orientation is not None
            and orientation < 90.0
        )

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

        yy, xx = layer.retina.yy_xx
        mask = self._region_mask_at(layer, circle, yy, xx, image_canvas=True)
        layer._cre_binary_mask_cache[cache_key] = mask
        return mask

    def _region_mask_at(self, layer, circle, yy, xx, *, image_canvas=False):
        """Unclipped requested geometry on any source-coordinate canvas."""
        retina = layer.retina
        center = retina.disc.center
        if image_canvas and circle.center == center:
            distance_sq = retina.disc_center_dist_sq
        else:
            distance_sq = (yy - circle.center.y)**2 + (xx - circle.center.x)**2
        mask = distance_sq <= circle.r**2
        if retina.fovea_location is not None:
            if self.CREMode == CREMode.Temporal:
                angles = self._temporal_angle_deg(layer, yy, xx)
                if angles is not None:
                    mask &= angles < 85.0
            elif self.CREMode == CREMode.Nasal:
                if image_canvas:
                    angles = retina.disc_fovea_angle_deg
                else:
                    vy = retina.fovea_location.y - center.y
                    vx = retina.fovea_location.x - center.x
                    dy, dx = yy - center.y, xx - center.x
                    cosine = (dx*vx + dy*vy) / ((np.hypot(dx, dy) + 1e-6)
                                               * (np.hypot(vx, vy) + 1e-6))
                    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
                mask &= angles > 80.0
        if self.hemifield_spec is not None:
            # Hemifield grid masks already include ROI clipping. Use their
            # geometric dividing line so missing area still counts against QC.
            grid = retina.get_grid(self.hemifield_spec.grid_spec)
            px, py = grid._perp_unit
            superior = (xx - grid._center.x)*px + (yy - grid._center.y)*py < 0
            if self.hemifield_spec.field == HemifieldField.Superior:
                mask &= superior
            elif self.hemifield_spec.field == HemifieldField.Inferior:
                mask &= ~superior
        return mask

    def circle_region_in_bounds(self, layer, circle) -> bool:
        return circle_region_in_bounds(
            circle, layer.retina.resolution, layer.retina.roi_mask,
            lambda y, x: self._region_mask_at(layer, circle, y, x),
        )

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
                seg for seg in filtered_segments if self._is_temporal_segment(seg)
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
        return recursive_cre(calibers, cte)

    def compute_cre_for_circle(self, layer: VesselTreeLayer, circle: Circle):
        if layer.name == "arteries":
            cte = 0.88
        elif layer.name == "veins":
            cte = 0.95
        else:
            raise ValueError("Unrecognized layer type for CRE computation")
        # Build mask and enforce full containment within retina bounds
        requested_circle = circle.resize(layer.retina.disc.circle.r / 5.0)
        if not self.circle_region_in_bounds(layer, requested_circle):
            return None, []
        mask = self.__get_binary_mask(layer, requested_circle)

        intersections = self.get_intersections(layer, circle)
        # Filter intersections by on-mask (True) values
        h, w = mask.shape
        filtered_intersections: List[Tuple[Segment, float]] = []
        for seg, t in intersections:
            y, x = seg.get_spline(error_fraction=self.spline_error_fraction).get_point(
                t
            )
            yi, xi = int(round(y)), int(round(x))
            if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
                filtered_intersections.append((seg, t))
        intersections = filtered_intersections
        if len(intersections) == 0:
            return None, []
        # Deduplicate segments that may intersect circle multiple times
        segments = list(set([p[0] for p in intersections]))
        segments.sort(
            key=lambda s: s.get_median_diameter(self.spline_error_fraction),
            reverse=True,
        )
        if len(segments) < self.max_vessels:
            return None, []
        segments = segments[: self.max_vessels]
        selected_segments = set(segments)
        selected_intersections: List[Tuple[Segment, float]] = []
        seen_segments: set[Segment] = set()
        for seg, t in intersections:
            if seg not in selected_segments or seg in seen_segments:
                continue
            selected_intersections.append((seg, t))
            seen_segments.add(seg)

        calibers = [s.get_median_diameter(self.spline_error_fraction) for s in segments]
        return self.recursive_cre(calibers, cte), selected_intersections

    def get_aric_result(self, layer: VesselTreeLayer):
        """Return selected trunks, segment median calibers, coverage, and quality flags."""
        from ._cre_aric import measure_aric

        return measure_aric(self, layer)

    def compute(self, layer: VesselTreeLayer):
        if self.measure == CREMeasure.ARIC:
            result = self.get_aric_result(layer)
            if result.flags:
                warnings.warn(f"ARIC CRE ({self.canonical_name(layer_name=layer.name)}): "
                              + "; ".join(result.flags), stacklevel=2)
            if not result.valid:
                return None
            if layer.name not in ("arteries", "veins"):
                raise ValueError("Unrecognized layer type for CRE computation")
            cte = 0.88 if layer.name == "arteries" else 0.95
            value = recursive_cre([m.median_diameter for m in result.selected], cte)
            return layer.retina.scale_length_measurement(value)
        cres = []
        for od_multiple in self.get_circle_multiples():
            circle = self.get_circle(layer, od_multiple)

            cre, _ = self.compute_cre_for_circle(layer, circle)
            if cre is not None:
                cres.append(cre)

        if len(cres) < self.min_circles:
            return None

        return layer.retina.scale_length_measurement(float(np.median(cres)))

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_grid_field_suffix, get_layer_suffix

        field = get_grid_field_suffix(self.hemifield_spec)
        layer = get_layer_suffix(layer_name)
        mode = self.CREMode.name
        measure = " ARIC" if self.measure == CREMeasure.ARIC else ""
        return f"{mode} CRE{measure}{field}{layer}"

    def name_prefix_tokens(self) -> list[str]:
        return [self.CREMode.value]

    def feature_name_tokens(self) -> list[str]:
        # A separate naming family keeps existing resolved CRE names stable when
        # full/nasal ARIC variants are added to a temporal-only feature set.
        return ["cre", "aric"] if self.measure == CREMeasure.ARIC else ["cre"]

    def parameter_name_tokens(self) -> list[str]:
        from .base import format_name_value

        tokens: list[str] = []
        if self.measure == CREMeasure.ARIC:
            if self.min_vessels is not None:
                tokens.extend(["min_vessels", str(self.min_vessels)])
            if self.min_area_within_bounds != 1.0:
                tokens.extend(["min_area_within_bounds", format_name_value(self.min_area_within_bounds)])
        if self.max_vessels != self.default_max_vessels():
            tokens.extend(["max_vessels", str(self.max_vessels)])
        if self.measure == CREMeasure.Circles and self.min_circles != 2:
            tokens.extend(["min_circles", str(self.min_circles)])
        if self.inner_circle != 1.0:
            tokens.extend(["inner_circle", str(self.inner_circle)])
        if self.outer_circle != 1.5:
            tokens.extend(["outer_circle", str(self.outer_circle)])
        if self.measure == CREMeasure.Circles and self.num_circles != 5:
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

    def region_description(self, **kwargs) -> str:
        if self.measure != CREMeasure.ARIC:
            return super().region_description(**kwargs)
        label = "Zone B" if (self.inner_circle, self.outer_circle) == (1.0, 1.5) else "the configured annulus"
        text = (f"Trunks selected in {label}, {self.inner_circle:g}–{self.outer_circle:g} optic-disc "
                "diameters from the disc center.")
        if self.CREMode != CREMode.Full or self.hemifield_spec is not None:
            text += " Regional restrictions are VascX adaptations of the full-field ARIC protocol."
        if self.hemifield_spec is not None:
            text += f" Restricted to the {self.hemifield_spec.field.name.lower()} hemifield."
        text += f" Requires at least {100 * self.min_area_within_bounds:g}% visibility of the requested annulus region."
        return text

    def plot_description(self) -> str:
        if self.measure == CREMeasure.ARIC:
            return (
                "Green marks the measurement region. Colored paths show "
                "selected trunk sections in the region; labels give safeguarded whole-segment "
                "median widths. Dashed paths "
                "mark daughter substitutions; gray paths are unselected candidates. "
                "The panel reports coverage, vessel count, and quality flags."
            )
        return super().plot_description()

    def _plot(self, ax, layer: VesselTreeLayer, **kwargs):
        """This plot shows the circles used to compute CRE,
        the segments used in the CRE computation and the CRE next to each circle
        """
        if self.measure == CREMeasure.ARIC:
            from ._cre_aric import plot_aric
            return plot_aric(self, ax, layer, **kwargs)
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
                p[0]
                .get_spline(error_fraction=self.spline_error_fraction)
                .get_point(p[1])
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
        # reset axes limits to the image resolution
        # they might get extended if the circles go out of bounds
        h, w = layer.retina.resolution
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        for p in points:
            ax.scatter(x=p[1], y=p[0], c="white", s=2, marker="x")

        ax.text(
            0.05,
            0.95,
            f"max_vessels={self.max_vessels}",
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
