from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
from matplotlib import pyplot as plt

from vascx.shared.segment import Segment
from vascx.shared.vessels import Vessels

from .base import LayerFeature
from .cre import CREMode

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


def recursive_knudtson_cre(calibers: list[float], cte: float) -> Optional[float]:
    """Recursively reduce vessel calibers using the Knudtson formula."""
    if len(calibers) == 0:
        return None
    if len(calibers) == 1:
        return float(calibers[0])

    sorted_calibers = sorted(float(c) for c in calibers)
    mid = len(sorted_calibers) // 2
    reduced: list[float] = []
    for i in range(mid + len(sorted_calibers) % 2):
        if len(sorted_calibers) % 2 != 0 and i == mid:
            reduced.append(sorted_calibers[i])
        else:
            reduced.append(
                cte * np.sqrt(sorted_calibers[i] ** 2 + sorted_calibers[-i - 1] ** 2)
            )

    return recursive_knudtson_cre(reduced, cte)


class CREKnudtson(LayerFeature):
    """Central retinal equivalent using the Knudtson big-vessel Zone B protocol.

    Full mode uses the 6 largest Zone B vessel segments. Temporal and nasal modes use
    the 4 largest eligible Zone B vessel segments. Diameters are segment medians.
    """

    def __init__(
        self,
        CREMode: CREMode = CREMode.Temporal,
        full_vessels: int = 6,
        temporal_nasal_vessels: int = 4,
        zone_inner_circle: float = 1.0,
        zone_outer_circle: float = 1.5,
        spline_error_fraction: float = 0.05,
    ):
        self.CREMode = CREMode
        self.full_vessels = int(full_vessels)
        self.temporal_nasal_vessels = int(temporal_nasal_vessels)
        self.zone_inner_circle = float(zone_inner_circle)
        self.zone_outer_circle = float(zone_outer_circle)
        self.spline_error_fraction = float(spline_error_fraction)

        if self.full_vessels < 1 or self.temporal_nasal_vessels < 1:
            raise ValueError("Number of vessels must be at least 1")
        if self.zone_outer_circle <= self.zone_inner_circle:
            raise ValueError("zone_outer_circle must be greater than zone_inner_circle")

    @property
    def target_vessels(self) -> int:
        """Return the number of vessels required for this CRE mode."""
        if self.CREMode in [CREMode.Temporal, CREMode.Nasal]:
            return self.temporal_nasal_vessels
        return self.full_vessels

    def _knudtson_constant(self, layer: "VesselTreeLayer") -> float:
        """Return the artery/vein Knudtson coefficient."""
        if layer.name == "arteries":
            return 0.88
        if layer.name == "veins":
            return 0.95
        raise ValueError("Unrecognized layer type for CRE computation")

    def _zone_b_mask(self, layer: "VesselTreeLayer") -> Optional[np.ndarray]:
        """Return the Zone B annulus mask centered on the optic disc."""
        retina = layer.retina
        if retina.disc is None:
            return None

        yy, xx = retina.yy_xx
        disc_center = retina.disc.center_of_mass
        disc_diameter = 2.0 * retina.disc.circle.r
        inner_r = self.zone_inner_circle * disc_diameter
        outer_r = self.zone_outer_circle * disc_diameter
        dist_sq = (yy - disc_center.y) ** 2 + (xx - disc_center.x) ** 2
        mask = (inner_r**2 <= dist_sq) & (dist_sq < outer_r**2)
        try:
            mask &= retina.mask.astype(bool)
        except Exception:
            pass
        return mask

    def _segment_intersects_mask(self, segment: Segment, mask: np.ndarray) -> bool:
        """Return whether any segment skeleton point lies inside the mask."""
        skeleton = np.asarray(segment.skeleton, dtype=np.int32)
        if skeleton.size == 0:
            return False

        h, w = mask.shape
        y = skeleton[:, 0]
        x = skeleton[:, 1]
        in_image = (0 <= y) & (y < h) & (0 <= x) & (x < w)
        if not np.any(in_image):
            return False
        return bool(np.any(mask[y[in_image], x[in_image]]))

    def _temporal_origin_and_vector(
        self, layer: "VesselTreeLayer"
    ) -> Optional[tuple[float, float, float, float]]:
        """Return the shifted temporal origin and OD-to-fovea vector."""
        retina = layer.retina
        if retina.disc is None or retina.fovea_location is None:
            return None

        disc_center = retina.disc.center_of_mass
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
        self, layer: "VesselTreeLayer", y: float, x: float
    ) -> Optional[float]:
        """Measure angle from the shifted temporal origin to an image point."""
        geometry = self._temporal_origin_and_vector(layer)
        if geometry is None:
            return None

        origin_y, origin_x, vy, vx = geometry
        dy = y - origin_y
        dx = x - origin_x
        norm_v = np.hypot(vx, vy)
        norm_p = np.hypot(dx, dy) + 1e-6
        cosang = (dx * vx + dy * vy) / (norm_p * norm_v)
        return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

    def _segment_orientation(self, segment: Segment) -> Optional[float]:
        """Return segment orientation, or None if required landmarks are missing."""
        try:
            orientation = segment.orientation()
        except Exception:
            return None
        if orientation is None:
            return None
        return float(orientation)

    def _is_temporal_segment(self, segment: Segment) -> bool:
        """Return whether a segment is eligible for temporal Knudtson CRE."""
        point = segment.mean_position()
        temporal_angle = self._temporal_angle_deg(segment.layer, point.y, point.x)
        if temporal_angle is None or temporal_angle >= 85.0:
            return False
        orientation = self._segment_orientation(segment)
        return orientation is not None and orientation < 90.0

    def _is_nasal_segment(self, segment: Segment) -> bool:
        """Return whether a segment is eligible for nasal Knudtson CRE."""
        try:
            fod_angle = segment.fod_angle()
        except Exception:
            return False
        if fod_angle is None or fod_angle <= 80.0:
            return False
        orientation = self._segment_orientation(segment)
        return orientation is not None and orientation > 90.0

    def _mode_filter(self, segment: Segment) -> bool:
        """Return whether a segment passes the mode-specific spatial filter."""
        if self.CREMode == CREMode.Temporal:
            return self._is_temporal_segment(segment)
        if self.CREMode == CREMode.Nasal:
            return self._is_nasal_segment(segment)
        return True

    def _candidate_segments(self, layer: "VesselTreeLayer") -> list[Segment]:
        """Return Zone B segments eligible for the requested mode."""
        mask = self._zone_b_mask(layer)
        if mask is None:
            return []

        return [
            segment
            for segment in layer.segments
            if self._segment_intersects_mask(segment, mask) and self._mode_filter(segment)
        ]

    def get_selected_segments(self, layer: "VesselTreeLayer") -> list[Segment]:
        """Return the largest Zone B segments used for Knudtson CRE."""
        segments = self._candidate_segments(layer)

        def sort_key(segment: Segment) -> tuple[float, int, float, float, int]:
            diameter = float(segment.get_median_diameter(self.spline_error_fraction))
            index = segment.index if segment.index is not None else 10**9
            mean_y, mean_x = segment.mean_xy
            return (-diameter, int(index), float(mean_y), float(mean_x), len(segment.skeleton))

        segments.sort(key=sort_key)
        return segments[: self.target_vessels]

    def compute(self, layer: "VesselTreeLayer") -> Optional[float]:
        selected_segments = self.get_selected_segments(layer)
        if len(selected_segments) < self.target_vessels:
            warnings.warn(
                f"Could not find {self.target_vessels} Zone B vessels for "
                f"{self.CREMode.value} Knudtson CRE."
            )
            return None

        calibers = [
            segment.get_median_diameter(self.spline_error_fraction)
            for segment in selected_segments
        ]
        cre = recursive_knudtson_cre(calibers, self._knudtson_constant(layer))
        if cre is None:
            return None
        return layer.retina.scale_length_measurement(float(cre))

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_layer_suffix

        return f"{self.CREMode.name} CRE Knudtson{get_layer_suffix(layer_name)}"

    def name_prefix_tokens(self) -> list[str]:
        return [self.CREMode.value]

    def feature_name_tokens(self) -> list[str]:
        return ["cre", "knudtson"]

    def parameter_name_tokens(self) -> list[str]:
        from .base import format_name_value

        tokens: list[str] = []
        if self.full_vessels != 6:
            tokens.extend(["full_vessels", str(self.full_vessels)])
        if self.temporal_nasal_vessels != 4:
            tokens.extend(["temporal_nasal_vessels", str(self.temporal_nasal_vessels)])
        if self.zone_inner_circle != 1.0:
            tokens.extend(["zone_inner_circle", str(self.zone_inner_circle)])
        if self.zone_outer_circle != 1.5:
            tokens.extend(["zone_outer_circle", str(self.zone_outer_circle)])
        if self.spline_error_fraction != 0.05:
            tokens.extend(
                [
                    "spline_error_fraction",
                    format_name_value(self.spline_error_fraction),
                ]
            )
        return tokens

    def name_tokens(self, layer_name: str, **kwargs) -> list[str]:
        from .base import get_layer_tokens

        return [
            *self.name_prefix_tokens(),
            *self.feature_name_tokens(),
            *self.parameter_name_tokens(),
            *get_layer_tokens(layer_name),
        ]

    def _plot(self, ax, layer: "VesselTreeLayer", **kwargs):
        selected_segments = self.get_selected_segments(layer)
        layer.retina.plot(ax=ax, image=True, bounds=True, av=False)

        Vessels(layer, selected_segments).plot(
            ax=ax,
            show_index=True,
            cmap="tab20",
            segments=True,
            image=False,
            text=lambda s: f"{s.get_median_diameter(self.spline_error_fraction):.2f}",
            **kwargs,
        )

        retina = layer.retina
        if retina.disc is not None:
            disc_diameter = 2.0 * retina.disc.circle.r
            for radius, linestyle in [
                (self.zone_inner_circle * disc_diameter, "-"),
                (self.zone_outer_circle * disc_diameter, "--"),
            ]:
                ax.add_patch(
                    plt.Circle(
                        retina.disc.center_of_mass.tuple_xy,
                        radius,
                        color="white",
                        fill=False,
                        lw=0.6,
                        linestyle=linestyle,
                    )
                )

        h, w = layer.retina.resolution
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.text(
            0.05,
            0.95,
            f"selected={len(selected_segments)}/{self.target_vessels}",
            transform=ax.transAxes,
            fontsize=6,
            color="white",
            ha="left",
            va="top",
        )
        return ax
