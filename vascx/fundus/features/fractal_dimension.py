from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional

import numpy as np
from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from .base import (
    LayerFeature,
    format_name_value,
    get_grid_field_suffix,
    get_grid_field_tokens,
    get_layer_suffix,
    grid_field_passes_qc,
    resolve_min_area_within_bounds,
    validate_min_area_within_bounds,
)


@dataclass(frozen=True)
class FractalResult:
    """Shared D0/D1/D2 estimates and the exact scale curves used in their fits.

    Rows in ``entropies`` and ``coefficients`` correspond to q=0,1,2;
    coefficient columns are slope and intercept against -log(box size).
    Arrays are read-only. Missing estimates have a reason and no dimensions.
    """

    sizes: np.ndarray
    counts: np.ndarray
    entropies: np.ndarray
    coefficients: np.ndarray
    dimensions: Optional[tuple[float, float, float]]
    r_squared: np.ndarray
    reason: Optional[str] = None

    def __post_init__(self):
        for array in (self.sizes, self.counts, self.entropies,
                      self.coefficients, self.r_squared):
            array.setflags(write=False)


def _missing_result(reason):
    return FractalResult(np.array([], dtype=int), np.array([], dtype=int),
                         np.empty((3, 0)), np.full((3, 2), np.nan),
                         None, np.full(3, np.nan), reason)


def _block_sums(mask, size):
    """Sum image-origin-aligned boxes, including partial boxes at image edges."""
    return np.add.reduceat(
        np.add.reduceat(mask, np.arange(0, mask.shape[0], size), axis=0),
        np.arange(0, mask.shape[1], size), axis=1,
    )


def _measure(mask, sizes):
    # Each coarser dyadic grid reuses the preceding sums. Work is proportional
    # to image area, rather than rescanning the full image for every scale/q.
    sums = _block_sums(mask, int(sizes[0]))
    counts = []
    entropies = []
    for index, size in enumerate(sizes):
        if index:
            sums = _block_sums(sums, 2)
        mass = sums[sums > 0].astype(np.float64)
        if mass.size == 0:
            return _missing_result("No vessel pixels in the analysis region")
        p = mass / mass.sum()
        counts.append(mass.size)
        entropies.append((np.log(mass.size), -np.sum(p * np.log(p)),
                          -np.log(np.sum(p * p))))
    curves = np.asarray(entropies).T
    x = -np.log(sizes)
    coefficients = np.polyfit(x, curves.T, 1).T
    fitted = coefficients[:, :1] * x + coefficients[:, 1:]
    ss_total = np.sum((curves - curves.mean(axis=1, keepdims=True)) ** 2, axis=1)
    ss_error = np.sum((curves - fitted) ** 2, axis=1)
    r_squared = np.full(3, np.nan)
    np.divide(ss_error, ss_total, out=r_squared, where=ss_total > 0)
    r_squared = 1 - r_squared
    return FractalResult(sizes, np.asarray(counts), curves, coefficients,
                         tuple(float(v) for v in coefficients[:, 0]), r_squared)


class FractalDimension(LayerFeature):
    """Generalized box-counting dimension Dq for q=0,1,2.

    Defaults use the filled binary vessel mask, all nonempty boxes, a grid
    anchored at image pixel (0,0), and dyadic sizes from 4 pixels through half
    the shorter bounding-box extent of the visible analysis region. At least
    four scales are required. Region boundaries clip vessel mass; boundary
    boxes remain included without area normalization. No rotations, occupancy
    filtering, scale selection based on fit, or post-fit clamping are applied.

    ``representation='skeleton'`` skeletonizes the full layer before masking
    the region. ``get_result`` shares all three dimensions and diagnostics
    across q selectors on the same layer. As with other cached layer products,
    the layer masks and geometry must remain unchanged during its lifetime.
    """

    general_description = (
        "Generalized fractal dimensions describe how the retinal vessel pattern "
        "and its distribution of vessel pixels change with observation scale."
    )
    default_min_area_within_bounds = 0.80

    def __init__(self, q: int = 0,
                 grid_field: Optional[BaseGridFieldSpecification] = None,
                 *, min_box_size: int = 4, max_box_fraction: float = 0.5,
                 representation: str = "vessels",
                 min_area_within_bounds: Optional[float] = None,
                 plot: bool = False):
        super().__init__(grid_field_spec=grid_field, plot=plot)
        if isinstance(q, bool) or not isinstance(q, Integral) or q not in (0, 1, 2):
            raise ValueError("q must be 0, 1, or 2")
        if (isinstance(min_box_size, bool) or not isinstance(min_box_size, Integral)
                or min_box_size < 1 or min_box_size & (min_box_size - 1)):
            raise ValueError("min_box_size must be a positive power of two")
        if (isinstance(max_box_fraction, bool) or not isinstance(max_box_fraction, Real)
                or not np.isfinite(max_box_fraction) or not 0 < max_box_fraction <= 1):
            raise ValueError("max_box_fraction must be finite and in (0, 1]")
        if representation not in ("vessels", "skeleton"):
            raise ValueError("representation must be 'vessels' or 'skeleton'")
        self.q = int(q)
        self.min_box_size = int(min_box_size)
        self.max_box_fraction = float(max_box_fraction)
        self.representation = representation
        self.min_area_within_bounds = validate_min_area_within_bounds(min_area_within_bounds)

    def _region_mask(self, layer):
        region = layer.retina.mask.astype(bool)
        field = self._get_grid_field(layer)
        if field is not None:
            region = region & field.mask.astype(bool)
        return region

    def _vessel_mask(self, layer, region):
        binary = layer.binary.astype(bool)
        if self.representation == "skeleton":
            from skimage.morphology import skeletonize
            binary = skeletonize(binary)
        return binary & region

    def get_result(self, layer) -> FractalResult:
        """Return cached dimensions, scale curves and fit quality for this layer."""
        threshold = resolve_min_area_within_bounds(
            self.grid_field_spec, self.min_area_within_bounds,
            self.default_min_area_within_bounds,
        )
        key = (tuple(get_grid_field_tokens(self.grid_field_spec)), threshold,
               self.min_box_size, self.max_box_fraction, self.representation)
        cache = getattr(layer, "_fractal_dimension_cache", None)
        if cache is None:
            cache = layer._fractal_dimension_cache = {}
        if key not in cache:
            if self.grid_field_spec is not None and not grid_field_passes_qc(
                layer.retina, self.grid_field_spec, min_fraction_in_bounds=threshold
            ):
                result = _missing_result("Insufficient analysis-region coverage")
            else:
                region = self._region_mask(layer)
                rows = np.flatnonzero(region.any(axis=1))
                cols = np.flatnonzero(region.any(axis=0))
                if not len(rows) or not len(cols):
                    result = _missing_result("Empty analysis region")
                else:
                    extent = min(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1)
                    limit = extent * self.max_box_fraction
                    sizes = []
                    size = self.min_box_size
                    while size <= limit:
                        sizes.append(size)
                        size *= 2
                    if len(sizes) < 4:
                        result = _missing_result("Fewer than four box sizes fit the analysis region")
                    else:
                        result = _measure(self._vessel_mask(layer, region), np.asarray(sizes))
            cache[key] = result
        return cache[key]

    def compute(self, layer):
        result = self.get_result(layer)
        return None if result.dimensions is None else result.dimensions[self.q]

    def display_name(self, layer_name: str, key: str = None) -> str:
        return (f"Fractal Dimension D{self.q}"
                f"{get_grid_field_suffix(self.grid_field_spec)}{get_layer_suffix(layer_name)}")

    def feature_name_tokens(self):
        return ["fd", f"d{self.q}"]

    def parameter_name_tokens(self):
        tokens = super().parameter_name_tokens()
        if self.representation != "vessels":
            tokens.append("skl")
        if self.min_box_size != 4:
            tokens.extend(["min_box_size", str(self.min_box_size)])
        if self.max_box_fraction != 0.5:
            tokens.extend(["max_box_fraction", format_name_value(self.max_box_fraction)])
        return tokens

    def implementation_description(self, **kwargs):
        representation = "vessel centerline pixels" if self.representation == "skeleton" else "vessel pixels"
        statistic = ("the logarithm of the number of occupied boxes",
                     "the entropy of normalized vessel-pixel counts",
                     "the negative logarithm of the sum of squared vessel-pixel probabilities")[self.q]
        return (
            f"D{self.q} is the slope of {statistic} against negative log box size. "
            f"Counts use {representation} inside the visible analysis region and all nonempty "
            f"boxes, including full and boundary boxes. An image-origin-aligned grid uses "
            f"powers-of-two sizes from {self.min_box_size} pixels through "
            f"{self.max_box_fraction:g} times the region's shorter bounding-box extent, "
            "with at least four scales and no rotations. These finite-scale estimates "
            "are dimensionless and are not exact AutoMorph or PVBM reproductions."
        )

    def _plot(self, ax, layer, **kwargs):
        """Cyan shows the measured vessel mask; gold outlines counted boxes at one
        scale. The inset shows all three scale curves and fits, with the displayed
        box size highlighted. D0, D1 and D2 share the same mask and scales.
        """
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        from vascx.utils.plotting import plot_mask

        layer.retina.plot(ax=ax, image=True, av=False, disc=False, fovea=False, bounds=True)
        field = self._get_grid_field(layer)
        if field is not None:
            field.plot(ax)
        result = self.get_result(layer)
        if result.dimensions is None:
            ax.text(.02, .90, result.reason, transform=ax.transAxes, color="white",
                    fontsize=8, bbox=dict(facecolor="black", alpha=.7, edgecolor="none"))
            return ax
        region = self._region_mask(layer)
        mask = self._vessel_mask(layer, region)
        # Pick one intermediate scale for legibility, without changing fitted scales.
        index = len(result.sizes) // 2
        size = int(result.sizes[index])
        sums = _block_sums(mask, size)
        h, w = mask.shape
        patches = [Rectangle((c * size - .5, r * size - .5),
                             min(size, w - c * size), min(size, h - r * size))
                   for r, c in np.argwhere(sums > 0)]
        boxes = PatchCollection(patches, facecolor=(1, .75, .2, .08),
                               edgecolor=(1, .75, .2, .7), linewidth=.45)
        boxes.set_gid("fractal-boxes")
        ax.add_collection(boxes)
        plot_mask(ax, mask, color=(.3, .95, 1, .85))
        label = "  ".join(f"D{q}={d:.3f}" for q, d in enumerate(result.dimensions))
        ax.text(.02, .91, f"{label}\n{size} px boxes: N={result.counts[index]}",
                transform=ax.transAxes, color="white", fontsize=8, va="top",
                bbox=dict(facecolor="black", alpha=.75, edgecolor="none"))
        # Include the tick labels and axis labels in the inset's white panel.
        panel = Rectangle((.49, .015), .49, .36, transform=ax.transAxes,
                          facecolor=(1, 1, 1, .94), edgecolor="none", zorder=4)
        ax.add_patch(panel)
        inset = ax.inset_axes([.60, .11, .36, .23], zorder=5)
        inset.set_gid("fractal-fit")
        inset.set_facecolor((1, 1, 1, .93))
        x = -np.log(result.sizes)
        for q, color in enumerate(("#0868ac", "#d95f02", "#7a0177")):
            inset.plot(x, result.entropies[q], "o", ms=2, color=color)
            inset.plot(x, np.polyval(result.coefficients[q], x), lw=.8,
                       color=color, label=f"D{q}  R²={result.r_squared[q]:.3f}")
        inset.axvline(x[index], color="#b8860b", lw=.8, alpha=.8)
        inset.tick_params(labelsize=7.2, length=2)
        inset.set_xlabel("−log(box size / px)", fontsize=7.2, labelpad=1)
        inset.set_ylabel("Scale statistic", fontsize=7.2, labelpad=1)
        inset.legend(fontsize=7.2, loc="best", frameon=False, handlelength=1)
        return ax
