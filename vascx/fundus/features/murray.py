"""Murray equivalent diameters from vessels crossing Zone B circles."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field, replace
from numbers import Real
import warnings

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from rtnls_enface.base import Circle
from rtnls_enface.grids.disc_centered import DiscCenteredRing
from rtnls_enface.grids.specifications import (
    DiscCenteredGridSpecification, GridFieldSpecification,
)

from vascx.shared.outputs import FeatureOutput
from .base import LayerFeature, format_name_value, get_grid_field_suffix, get_layer_suffix


OD_FULL = GridFieldSpecification(DiscCenteredGridSpecification(), DiscCenteredRing.FullGrid)
DEFAULT_POWERS = {"arteries": 1.5, "veins": 1.25}
ZONE_B_INNER = 1.0
ZONE_B_OUTER = 1.5
NUM_CIRCLES = 5
MIN_CIRCLES = 2
SPLINE_ERROR_FRACTION = 0.05
# Sample geometry at half-pixel spacing, including the ends of Zone B.
GEOMETRY_STEP = 0.5


def _sample_mask(mask, y, x):
    """Nearest-pixel membership, treating points outside the canvas as absent."""
    yi, xi = np.rint(y).astype(int), np.rint(x).astype(int)
    inside = (yi >= 0) & (yi < mask.shape[0]) & (xi >= 0) & (xi < mask.shape[1])
    result = np.zeros(np.shape(inside), dtype=bool)
    result[inside] = mask[yi[inside], xi[inside]]
    return result


@dataclass
class _Region:
    mask: np.ndarray
    y0: int
    x0: int

    def contains(self, y, x):
        return _sample_mask(self.mask, np.asarray(y) - self.y0, np.asarray(x) - self.x0)


@dataclass
class _CircleMeasurement:
    arc: np.ndarray
    crossings: list = field(default_factory=list)
    power_sum: float | None = None


class MurrayEquivalentDiameter(LayerFeature):
    """Root of the median caliber-power sum across complete Zone B arcs.

    ``region`` selects crossings, not diameter samples: calibers are the existing
    whole-segment medians, which can include samples outside the selected region.
    All crossing segments contribute once per circle, without a largest-N cutoff.
    ``p_arteries`` and ``p_veins`` independently set the branching powers,
    defaulting to 1.5 and 1.25 respectively.
    """

    outputs = (FeatureOutput(units="px"),)
    general_description = (
        "Murray equivalent diameter is the caliber of one hypothetical vessel "
        "equivalent to the measured vessels under the chosen branching power."
    )

    def __init__(self, p_arteries: float = 1.5, p_veins: float = 1.25,
                 region: GridFieldSpecification = OD_FULL,
                 *, plot: bool = False):
        for name, power in (("p_arteries", p_arteries), ("p_veins", p_veins)):
            if isinstance(power, bool) or not isinstance(power, Real):
                raise ValueError(f"{name} must be a finite positive number")
            try:
                power = float(power)
            except OverflowError as error:
                raise ValueError(f"{name} must be a finite positive number") from error
            if not np.isfinite(power) or power <= 0:
                raise ValueError(f"{name} must be a finite positive number")
            setattr(self, name, power)
        if not isinstance(region, GridFieldSpecification):
            raise TypeError("region must be a GridFieldSpecification")
        super().__init__(grid_field_spec=region, plot=plot)

    def get_power(self, layer_name: str) -> float:
        """Return the configured branching exponent for the vessel type."""
        return {"arteries": self.p_arteries, "veins": self.p_veins}[layer_name]

    @property
    def region(self):
        return self.grid_field_spec

    def _region_geometry(self, layer, diameter, angles):
        """Resolve geometry beyond image edges using the existing grid padding protocol."""
        retina = layer.retina
        original = retina.get_grid(self.region.grid_spec)
        center = retina.disc.center
        # The extra pixels let erosion/dilation work at the outer Zone B circle.
        radius = ZONE_B_OUTER * diameter + 3
        x0, y0, x1, y1 = original._circle_bounds(center.x, center.y, radius)
        grid = copy(original)
        grid._clear_cached_properties(grid)
        grid.h, grid.w = y1 - y0, x1 - x0
        grid.mask = None
        grid._shift_canvas_origin(x0, y0)
        mask = grid.field(self.region.field).mask

        radii = np.linspace(ZONE_B_INNER * diameter, ZONE_B_OUTER * diameter,
                            max(2, int(np.ceil(0.5 * diameter / GEOMETRY_STEP)) + 1))
        y = center.y - y0 + radii[:, None] * np.sin(angles)
        x = center.x - x0 + radii[:, None] * np.cos(angles)
        # Ignore one-pixel angular boundary jitter when checking radial coverage.
        # This tolerance is only for validation; crossings use the exact mask.
        core = _sample_mask(binary_erosion(mask), y, x).any(axis=0)
        covered = _sample_mask(binary_dilation(mask), y, x).all(axis=0)
        if not core.any() or np.any(core & ~covered):
            warnings.warn("MED region must radially contain the complete Zone B annulus "
                          "within its angular sector", stacklevel=3)
            return None
        return _Region(mask, y0, x0)

    def _measurements(self, layer):
        retina = layer.retina
        power = self.get_power(layer.name)
        if retina.disc is None:
            return []
        if self.region.grid_spec.requires_fovea and retina.fovea_location is None:
            return []
        diameter = 2 * retina.disc.circle.r
        center = retina.disc.center
        angles = np.linspace(0, 2 * np.pi,
                             max(8, int(np.ceil(2 * np.pi * ZONE_B_OUTER * diameter / GEOMETRY_STEP))),
                             endpoint=False)
        region = self._region_geometry(layer, diameter, angles)
        if region is None:
            return []

        measurements = []
        for multiple in np.linspace(ZONE_B_INNER, ZONE_B_OUTER, NUM_CIRCLES):
            circle = Circle(center=center, r=multiple * diameter)
            y = center.y + circle.r * np.sin(angles)
            x = center.x + circle.r * np.cos(angles)
            requested = region.contains(y, x)
            arc = np.column_stack((y, x))
            arc[~requested] = np.nan
            record = _CircleMeasurement(arc)
            measurements.append(record)
            if not requested.any() or not _sample_mask(retina.mask, y[requested], x[requested]).all():
                continue

            seen = set()
            for segment, t in layer.get_circle_intersections(circle, SPLINE_ERROR_FRACTION):
                point = segment.get_spline(SPLINE_ERROR_FRACTION).get_point(t)
                if segment not in seen and region.contains(*point):
                    record.crossings.append((segment, point))
                    seen.add(segment)
            if not record.crossings:
                continue
            try:
                diameters = np.array([segment.get_median_diameter(SPLINE_ERROR_FRACTION)
                                      for segment, _ in record.crossings], dtype=float)
            except (ValueError, RuntimeError, FloatingPointError):
                continue
            if not np.isfinite(diameters).all() or np.any(diameters <= 0):
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                total = np.sum(diameters ** power)
            if np.isfinite(total) and total > 0:
                record.power_sum = float(total)
        return measurements

    def compute(self, layer):
        sums = [m.power_sum for m in self._measurements(layer) if m.power_sum is not None]
        if len(sums) < MIN_CIRCLES:
            return None
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            result = np.power(np.median(sums), 1 / self.get_power(layer.name))
        return float(result) if np.isfinite(result) and result > 0 else None

    def feature_name_tokens(self):
        return ["med"]

    def name_parts(self, layer_name="retina", **kwargs):
        # Enum values retain 'quadrant_', unlike the shared name-based convention.
        token = "full" if self.region.field.value == "grid" else self.region.field.value
        parts = super().name_parts(layer_name=layer_name, **kwargs)
        result = []
        for part in parts:
            if part.key == "field":
                part = replace(part, tokens=(token,), display=self.region.field.friendly_name.title())
            elif part.key == "parameters" and layer_name in DEFAULT_POWERS:
                power = self.get_power(layer_name)
                tokens = () if power == DEFAULT_POWERS[layer_name] else ("power", format_name_value(power))
                part = replace(part, tokens=tokens, display=f"power={power:g}" if tokens else "")
            result.append(part)
        return result

    def name_tokens(self, layer_name, **kwargs):
        return [token for part in self.name_parts(layer_name=layer_name) for token in part.tokens]

    def display_name(self, layer_name, **kwargs):
        return (f"Murray Equivalent Diameter{get_grid_field_suffix(self.region)}"
                f"{get_layer_suffix(layer_name)}")

    def implementation_description(self, layer_name=None, **kwargs):
        if layer_name in DEFAULT_POWERS:
            power = f"power={self.get_power(layer_name):g}"
        else:
            power = f"p_arteries={self.p_arteries:g} and p_veins={self.p_veins:g}"
        return (f"Uses {power} and whole-segment median diameters of all unique "
                "segments crossing the requested region at five OD-centered radii from "
                "1 to 1.5 optic-disc diameters (Zone B). Diameter samples can lie outside "
                "the region. Each requested arc must be fully visible; no vessel-count "
                "cutoff or orientation filter is applied.")

    def aggregation_description(self, **kwargs):
        return (f"Requires at least {MIN_CIRCLES} valid arcs. Takes the median of the "
                "per-arc diameter-power sums, then the reciprocal-power root.")

    def _plot(self, ax, layer, **kwargs):
        """White arcs contribute to MED; dashed gray arcs are unavailable. Colored
        paths show contributing segments, whose whole-segment median widths are
        used. Dots mark the selected crossings inside the requested region.
        """
        layer.retina.plot(ax=ax, disc=True, fovea=False, av=False, bounds=False)
        drawn = set()
        for measurement in self._measurements(layer):
            valid = measurement.power_sum is not None
            y, x = measurement.arc.T
            ax.plot(x, y, color="white" if valid else "gray",
                    linestyle="-" if valid else "--", linewidth=0.8)
            if not valid:
                continue
            for segment, point in measurement.crossings:
                if segment not in drawn:
                    skeleton = np.asarray(segment.skeleton)
                    ax.plot(skeleton[:, 1], skeleton[:, 0], linewidth=1.5)
                    drawn.add(segment)
                ax.plot(point[1], point[0], "o", color="white", markersize=3)
        return ax
