from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from rtnls_enface.grids.specifications import BaseGridFieldSpecification
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.special import erf

from vascx.shared.segment import Segment

from .base import (
    VesselsLayerFeature,
    format_name_value,
    grid_field_passes_qc,
    resolve_min_area_within_bounds,
    validate_min_area_within_bounds,
)

if TYPE_CHECKING:
    from vascx.fundus.vessels_layer import FundusVesselsLayer


def _pulse_profile(
    x: np.ndarray,
    i_bg: float,
    amp: float,
    x_left: float,
    x_right: float,
    sigma: float,
) -> np.ndarray:
    """Blurred dark-vessel pulse: background minus difference of two erfs."""
    scale = sigma * np.sqrt(2.0) + 1e-12
    left, right = (x_left, x_right) if x_left <= x_right else (x_right, x_left)
    return i_bg - 0.5 * amp * (erf((x - left) / scale) - erf((x - right) / scale))


def _pulse_profile_mu_w(
    x: np.ndarray,
    i_bg: float,
    amp: float,
    mu: float,
    width: float,
    sigma: float,
    bg_slope: float = 0.0,
) -> np.ndarray:
    """Blurred pulse on an affine background, parameterized by center μ and width W."""
    half = 0.5 * abs(width)
    x_ref = 0.5 * (float(x[0]) + float(x[-1]))
    profile = _pulse_profile(x, i_bg, amp, mu - half, mu + half, sigma)
    return profile + bg_slope * (x - x_ref)


def _edges_from_mu_w(mu: float, width: float) -> Tuple[float, float]:
    """Convert (μ, W) to (x_left, x_right)."""
    half = 0.5 * abs(width)
    return mu - half, mu + half


def _gaussian_dip(
    x: np.ndarray,
    i_bg: float,
    amp: float,
    mu: float,
    sigma: float,
) -> np.ndarray:
    """Dark Gaussian trough: I_bg - A exp(-(x-μ)²/(2σ²))."""
    return i_bg - amp * np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)


# (σ, I_bg, A, x_left_or_mu, x_right_or_mu, kind) with kind in {"pulse", "gaussian"}
BilateralFit = Tuple[float, float, float, float, float, str]


def bilateral_fit_is_valid(
    fit: BilateralFit,
    *,
    profile_length: int,
    diameter: float,
) -> bool:
    """True if σ≤2×diameter, W≤3×diameter, and their spans lie inside the profile."""
    sigma, _, _, a, b, kind = fit
    if not np.isfinite(sigma) or sigma <= 0 or diameter <= 0:
        return False
    if sigma > 2.0 * diameter:
        return False

    x_max = float(profile_length - 1)
    if kind == "gaussian":
        mu = a
        if mu - sigma < 0.0 or mu + sigma > x_max:
            return False
        return True

    x_left, x_right = (a, b) if a <= b else (b, a)
    width = x_right - x_left
    if width > 3.0 * diameter:
        return False
    if x_left < 0.0 or x_right > x_max:
        return False
    for x_edge in (x_left, x_right):
        if x_edge - sigma < 0.0 or x_edge + sigma > x_max:
            return False
    return True


def _init_pulse_params(
    profile: np.ndarray,
    *,
    smooth_sigma: float,
    min_contrast: float,
) -> Optional[Tuple[float, float, float, float, float]]:
    """Initialize (I_bg, background slope, A, μ, W) for a pulse fit."""
    p = np.asarray(profile, dtype=np.float64)
    if smooth_sigma > 0:
        p = gaussian_filter1d(p, sigma=smooth_sigma)

    n_samples = len(p)
    x = np.arange(n_samples, dtype=np.float64)
    x_ref = 0.5 * float(n_samples - 1)
    n_background = max(2, n_samples // 4)
    left_level = float(np.median(p[:n_background]))
    right_level = float(np.median(p[-n_background:]))
    left_x = float(np.median(x[:n_background]))
    right_x = float(np.median(x[-n_background:]))
    bg_slope = (right_level - left_level) / (right_x - left_x)
    i_bg = left_level + bg_slope * (x_ref - left_x)

    background = i_bg + bg_slope * (x - x_ref)
    amp = float(np.max(background - p))
    if amp < min_contrast:
        return None

    detrended = p - bg_slope * (x - x_ref)
    g = np.abs(np.gradient(detrended))
    peaks, _ = find_peaks(g, distance=max(2, len(p) // 5))
    if len(peaks) >= 2:
        peaks = np.sort(peaks[np.argsort(g[peaks])[-2:]])
        mu = 0.5 * (float(peaks[0]) + float(peaks[1]))
        width = float(peaks[1] - peaks[0])
    else:
        mu = float(np.argmin(detrended))
        half_level = i_bg - 0.5 * amp
        below = np.where(detrended <= half_level)[0]
        if below.size >= 2:
            width = float(below[-1] - below[0])
        else:
            width = max(2.0, len(p) / 6.0)

    width = max(width, 0.5)
    return i_bg, bg_slope, amp, mu, width


def fit_gaussian_profile_params(
    profile: np.ndarray,
    *,
    smooth_sigma: float = 0.7,
    min_contrast: float = 8.0,
) -> Optional[BilateralFit]:
    """Fit a dark Gaussian trough; return (σ, I_bg, A, μ, μ, 'gaussian')."""
    p = np.asarray(profile, dtype=np.float64)
    if p.ndim != 1 or p.size < 9:
        return None

    if smooth_sigma > 0:
        p = gaussian_filter1d(p, sigma=smooth_sigma)

    i_bg0 = float(np.mean([p[0], p[-1]]))
    amp0 = float(i_bg0 - np.min(p))
    if amp0 < min_contrast:
        return None

    mu0 = float(np.argmin(p))
    half_level = i_bg0 - 0.5 * amp0
    below = np.where(p <= half_level)[0]
    if below.size >= 2:
        # FWHM ≈ 2.355 σ for a Gaussian
        sigma0 = max(0.8, float(below[-1] - below[0]) / 2.355)
    else:
        sigma0 = max(0.8, len(p) / 8.0)

    n_samples = len(p)
    x = np.arange(n_samples, dtype=np.float64)
    x0 = np.array([sigma0, i_bg0, amp0, mu0], dtype=np.float64)
    lower = [0.25, -np.inf, min_contrast * 0.5, 0.0]
    upper = [
        float(max(n_samples / 2.0, 1.0)),
        np.inf,
        np.inf,
        float(n_samples - 1),
    ]

    def residual(params: np.ndarray) -> np.ndarray:
        sigma, i_bg, amp, mu = params
        return p - _gaussian_dip(x, i_bg, amp, mu, sigma)

    try:
        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            max_nfev=2000,
        )
    except (RuntimeError, ValueError):
        return None

    sigma = float(result.x[0])
    if not np.isfinite(sigma) or sigma < 0.25 or not result.success:
        return None
    i_bg, amp, mu = (float(v) for v in result.x[1:])
    return sigma, i_bg, amp, mu, mu, "gaussian"


def _fit_pulse_profile_params(
    p: np.ndarray,
    *,
    diameter: float,
    min_contrast: float,
    width_prior_fraction: float,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Fit an affine-background pulse; return (σ, I_bg, slope, A, μ, W)."""
    if (
        not np.isfinite(diameter)
        or diameter <= 0
        or not np.isfinite(width_prior_fraction)
        or width_prior_fraction <= 0
    ):
        return None

    init = _init_pulse_params(p, smooth_sigma=0.0, min_contrast=min_contrast)
    if init is None:
        return None

    i_bg0, bg_slope0, amp0, mu0, _ = init
    n_samples = len(p)
    x = np.arange(n_samples, dtype=np.float64)
    x_ref = 0.5 * float(n_samples - 1)
    width0 = float(np.clip(diameter, 0.5, n_samples))
    sigma0 = max(0.8, width0 / 6.0)
    x0 = np.array([sigma0, i_bg0, bg_slope0, amp0, mu0, width0], dtype=np.float64)
    lower = [0.25, -np.inf, -np.inf, min_contrast * 0.5, 0.0, 0.5]
    upper = [
        float(max(n_samples / 2.0, 1.0)),
        np.inf,
        np.inf,
        np.inf,
        float(n_samples - 1),
        float(n_samples),
    ]

    # Estimate noise from deviations around the robustly initialized background
    # line. The contrast-based floor keeps noiseless/smoothed profiles numerically
    # well scaled without tying the priors to [0, 1] versus [0, 255] intensities.
    n_background = max(2, n_samples // 4)
    background_indices = np.concatenate(
        [np.arange(n_background), np.arange(n_samples - n_background, n_samples)]
    )
    initial_background = i_bg0 + bg_slope0 * (x - x_ref)
    background_residuals = (
        p[background_indices] - initial_background[background_indices]
    )
    centered_residuals = background_residuals - np.median(background_residuals)
    noise_scale = float(1.482602218505602 * np.median(np.abs(centered_residuals)))
    noise_scale = max(noise_scale, 0.01 * amp0, np.finfo(float).eps)

    width_prior_sigma = width_prior_fraction * diameter
    background_x = x[background_indices]
    slope_standard_error = noise_scale / np.sqrt(
        np.sum((background_x - np.mean(background_x)) ** 2)
    )
    slope_prior_sigma = max(
        5.0 * slope_standard_error,
        0.10 * abs(bg_slope0),
        np.finfo(float).eps,
    )

    def residual(params: np.ndarray) -> np.ndarray:
        sigma, i_bg, bg_slope, amp, mu, width = params
        model = _pulse_profile_mu_w(x, i_bg, amp, mu, width, sigma, bg_slope=bg_slope)
        data_residual = (p - model) / noise_scale
        width_residual = (width - diameter) / width_prior_sigma
        slope_residual = (bg_slope - bg_slope0) / slope_prior_sigma
        return np.concatenate([data_residual, [width_residual, slope_residual]])

    try:
        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            max_nfev=2000,
        )
    except (RuntimeError, ValueError):
        return None

    sigma = float(result.x[0])
    if not np.isfinite(sigma) or sigma < 0.25 or not result.success:
        return None
    i_bg, bg_slope, amp, mu, width = (float(v) for v in result.x[1:])
    return sigma, i_bg, bg_slope, amp, mu, width


def fit_bilateral_profile_params(
    profile: np.ndarray,
    *,
    diameter: float,
    smooth_sigma: float = 0.7,
    min_contrast: float = 8.0,
    width_prior_fraction: float = 0.20,
) -> Optional[BilateralFit]:
    """Fit an affine-background pulse with a soft diameter prior on W."""
    p = np.asarray(profile, dtype=np.float64)
    if p.ndim != 1 or p.size < 9:
        return None

    if smooth_sigma > 0:
        p = gaussian_filter1d(p, sigma=smooth_sigma)

    pulse = _fit_pulse_profile_params(
        p,
        diameter=diameter,
        min_contrast=min_contrast,
        width_prior_fraction=width_prior_fraction,
    )
    if pulse is None:
        return None
    sigma, i_bg, _, amp, mu, width = pulse
    x_left, x_right = _edges_from_mu_w(mu, width)
    return sigma, i_bg, amp, x_left, x_right, "pulse"


def fit_bilateral_profile_sigma(
    profile: np.ndarray,
    *,
    diameter: float,
    smooth_sigma: float = 0.7,
    min_contrast: float = 8.0,
    width_prior_fraction: float = 0.20,
) -> float:
    """Fit one edge-blur σ for a profile using a diameter-constrained pulse."""
    params = fit_bilateral_profile_params(
        profile,
        diameter=diameter,
        smooth_sigma=smooth_sigma,
        min_contrast=min_contrast,
        width_prior_fraction=width_prior_fraction,
    )
    if params is None:
        return np.nan
    if not bilateral_fit_is_valid(
        params, profile_length=len(profile), diameter=diameter
    ):
        return np.nan
    return float(params[0])


def fit_bilateral_segment_sigma(
    profiles: np.ndarray,
    *,
    diameter: float,
    smooth_sigma: float = 0.7,
    min_contrast: float = 8.0,
    width_prior_fraction: float = 0.20,
) -> float:
    """Fit each profile separately, then return their median pulse edge σ."""
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim != 2 or profiles.shape[0] < 1:
        return np.nan

    sigmas = [
        fit_bilateral_profile_sigma(
            profile,
            diameter=diameter,
            smooth_sigma=smooth_sigma,
            min_contrast=min_contrast,
            width_prior_fraction=width_prior_fraction,
        )
        for profile in profiles
    ]
    finite = [s for s in sigmas if np.isfinite(s)]
    if not finite:
        return np.nan
    return float(np.median(finite))


class Sharpness(VesselsLayerFeature):
    """Vessel-edge blur width from diameter-constrained pulse fits.

    Representation: Uses undirected vessel segments from FundusVesselsLayer (optionally
    clipped to a grid field) and green-channel intensity profiles perpendicular to each
    segment centerline.

    Computation: Separately fits an affine-background two-edge pulse to every profile,
    using the segment diameter as a soft prior on pulse width; takes the median σ over
    clean profiles, then a length-weighted mean over segments.
    Profiles that leave the fundus bounds or intersect other vessels are discarded.
    Lower values indicate sharper edges.
    """

    default_min_area_within_bounds = 0.80
    default_min_numpoints = 20
    default_n_profiles = 12
    default_profile_smooth_sigma = 0.7
    default_min_contrast = 4.0
    default_min_diameter = 2.0
    default_max_diameter = 25.0
    default_width_prior_fraction = 0.05

    def __init__(
        self,
        grid_field: Optional[BaseGridFieldSpecification] = None,
        min_numpoints: int = default_min_numpoints,
        n_profiles: int = default_n_profiles,
        profile_smooth_sigma: float = default_profile_smooth_sigma,
        min_contrast: float = default_min_contrast,
        min_diameter: float = default_min_diameter,
        max_diameter: float = default_max_diameter,
        width_prior_fraction: float = default_width_prior_fraction,
        min_area_within_bounds: Optional[float] = None,
    ):
        """Bilateral ESF sharpness (blur σ) from vessel cross-sections."""
        if min_numpoints < 2:
            raise ValueError("min_numpoints must be >= 2")
        if n_profiles < 1:
            raise ValueError("n_profiles must be >= 1")
        if profile_smooth_sigma < 0:
            raise ValueError("profile_smooth_sigma must be >= 0")
        if min_contrast < 0:
            raise ValueError("min_contrast must be >= 0")
        if min_diameter <= 0 or max_diameter <= min_diameter:
            raise ValueError("require 0 < min_diameter < max_diameter")
        if not np.isfinite(width_prior_fraction) or width_prior_fraction <= 0:
            raise ValueError("width_prior_fraction must be > 0")
        self.min_numpoints = int(min_numpoints)
        self.n_profiles = int(n_profiles)
        self.profile_smooth_sigma = float(profile_smooth_sigma)
        self.min_contrast = float(min_contrast)
        self.min_diameter = float(min_diameter)
        self.max_diameter = float(max_diameter)
        self.width_prior_fraction = float(width_prior_fraction)
        self.min_area_within_bounds = validate_min_area_within_bounds(
            min_area_within_bounds
        )
        super().__init__(grid_field_spec=grid_field)

    def _green_channel(self, layer: FundusVesselsLayer) -> Optional[np.ndarray]:
        image = layer.retina.image
        if image is None:
            return None
        if image.ndim == 2:
            return image.astype(np.float64)
        return image[..., 1].astype(np.float64)

    def _profile_validity_mask(self, layer: FundusVesselsLayer) -> np.ndarray:
        """Fundus pixels valid for profiles, excluding the optic disc when present."""
        valid = np.asarray(layer.retina.mask, dtype=bool).copy()
        if layer.retina.disc is not None:
            valid &= ~np.asarray(layer.retina.disc.mask, dtype=bool)
        return valid

    def _get_segments(self, layer: FundusVesselsLayer) -> List[Segment]:
        """Region-clipped (or full) undirected segments passing length/diameter filters."""
        segments = layer.get_region_segments(self.grid_field_spec)
        eligible: List[Segment] = []
        for segment in segments:
            if len(segment.skeleton) < self.min_numpoints:
                continue
            try:
                diameter = float(segment.get_median_diameter())
            except Exception:
                continue
            if not (self.min_diameter <= diameter <= self.max_diameter):
                continue
            if segment.get_spline() is None:
                continue
            eligible.append(segment)
        return eligible

    def _segment_profile_half_len(self, segment: Segment) -> Optional[int]:
        """Half-length L of perpendicular profiles for a segment (pixels)."""
        try:
            diameter = float(segment.get_median_diameter())
        except Exception:
            return None
        return max(6, int(ceil(2.5 * diameter)))

    def _other_vessels_mask(
        self, segment: Segment, vessels_mask: np.ndarray
    ) -> np.ndarray:
        """Full vessel mask with the target segment's mask pixels removed."""
        vessels = np.asarray(vessels_mask, dtype=bool)
        other = vessels.copy()
        try:
            pixels = segment.pixels
        except (ValueError, KeyError, AttributeError):
            return other
        if not pixels:
            return other

        h, w = other.shape
        ys = np.fromiter((int(p[0]) for p in pixels), dtype=np.intp, count=len(pixels))
        xs = np.fromiter((int(p[1]) for p in pixels), dtype=np.intp, count=len(pixels))
        valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        other[ys[valid], xs[valid]] = False
        return other

    def _profile_hits_other_vessels(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        half_len: int,
        other_vessels: np.ndarray,
    ) -> bool:
        """True if any sample along the profile line lands in other vessels."""
        h, w = other_vessels.shape
        for i in range(-half_len, half_len + 1):
            sample = origin + i * direction
            y = int(round(float(sample[0])))
            x = int(round(float(sample[1])))
            if 0 <= y < h and 0 <= x < w and other_vessels[y, x]:
                return True
        return False

    def _profile_outside_bounds(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        half_len: int,
        bounds_mask: np.ndarray,
    ) -> bool:
        """True if any sample is outside the valid fundus mask or inside the disc."""
        h, w = bounds_mask.shape
        for i in range(-half_len, half_len + 1):
            sample = origin + i * direction
            y = int(round(float(sample[0])))
            x = int(round(float(sample[1])))
            if not (0 <= y < h and 0 <= x < w) or not bounds_mask[y, x]:
                return True
        return False

    def _clean_profile_indices(
        self,
        segment: Segment,
        vessels_mask: np.ndarray,
        bounds_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Profiles inside valid retina, outside the disc, and clear of other vessels."""
        half_len = self._segment_profile_half_len(segment)
        if half_len is None:
            return None
        spline = segment.get_spline()
        if spline is None:
            return None

        other = self._other_vessels_mask(segment, vessels_mask)
        keep: List[int] = []
        for i, t in enumerate(np.linspace(0.0, 1.0, self.n_profiles)):
            origin = spline.get_point(t)
            direction = spline.get_perpendicular(t)
            if self._profile_outside_bounds(origin, direction, half_len, bounds_mask):
                continue
            if self._profile_hits_other_vessels(origin, direction, half_len, other):
                continue
            keep.append(i)
        if not keep:
            return None
        return np.asarray(keep, dtype=np.int32)

    def _segment_profiles(
        self,
        segment: Segment,
        green: np.ndarray,
        vessels_mask: np.ndarray,
        bounds_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return clean (n_profiles_kept, profile_length) green-channel cross-sections."""
        half_len = self._segment_profile_half_len(segment)
        if half_len is None:
            return None
        spline = segment.get_spline()
        if spline is None:
            return None
        keep = self._clean_profile_indices(segment, vessels_mask, bounds_mask)
        if keep is None:
            return None
        try:
            profiles = spline.profile(green, N=self.n_profiles, L=half_len)
        except Exception:
            return None
        profiles = np.asarray(profiles, dtype=np.float64)
        return profiles[keep]

    def _segment_profile_lines(
        self,
        segment: Segment,
        vessels_mask: np.ndarray,
        bounds_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return (n_clean, 2, 2) endpoints of clean profile sample lines as (x, y)."""
        half_len = self._segment_profile_half_len(segment)
        if half_len is None:
            return None
        spline = segment.get_spline()
        if spline is None:
            return None
        keep = self._clean_profile_indices(segment, vessels_mask, bounds_mask)
        if keep is None:
            return None

        lines = []
        ts = np.linspace(0.0, 1.0, self.n_profiles)
        for i in keep:
            t = float(ts[int(i)])
            origin = spline.get_point(t)  # (y, x)
            u = spline.get_perpendicular(t)
            start = origin - half_len * u
            end = origin + half_len * u
            lines.append(
                [
                    [float(start[1]), float(start[0])],
                    [float(end[1]), float(end[0])],
                ]
            )
        return np.asarray(lines, dtype=np.float64)

    def _segment_sigma(
        self,
        segment: Segment,
        green: np.ndarray,
        vessels_mask: np.ndarray,
        bounds_mask: np.ndarray,
    ) -> float:
        """Return one σ for a segment, or NaN if fitting fails."""
        profiles = self._segment_profiles(segment, green, vessels_mask, bounds_mask)
        if profiles is None or profiles.size == 0:
            return np.nan

        try:
            diameter = float(segment.get_median_diameter())
        except Exception:
            return np.nan

        return fit_bilateral_segment_sigma(
            profiles,
            diameter=diameter,
            smooth_sigma=self.profile_smooth_sigma,
            min_contrast=self.min_contrast,
            width_prior_fraction=self.width_prior_fraction,
        )

    def segment_scores(
        self, layer: FundusVesselsLayer
    ) -> List[Tuple[Segment, float]]:
        """Return (segment, σ) pairs for eligible segments with valid fits."""
        if self.grid_field_spec is not None:
            if not grid_field_passes_qc(
                layer.retina,
                self.grid_field_spec,
                min_fraction_in_bounds=resolve_min_area_within_bounds(
                    self.grid_field_spec,
                    self.min_area_within_bounds,
                    self.default_min_area_within_bounds,
                ),
            ):
                return []

        green = self._green_channel(layer)
        if green is None:
            return []

        # Full vessels + fundus bounds for contamination (includes outside grid field).
        vessels_mask = np.asarray(layer.mask, dtype=bool)
        bounds_mask = self._profile_validity_mask(layer)
        scores: List[Tuple[Segment, float]] = []
        for segment in self._get_segments(layer):
            sigma = self._segment_sigma(segment, green, vessels_mask, bounds_mask)
            if not np.isfinite(sigma):
                continue
            scores.append((segment, float(sigma)))
        return scores

    def compute(self, layer: FundusVesselsLayer) -> Optional[float]:
        """Length-weighted mean of per-segment ESF σ."""
        scores = self.segment_scores(layer)
        if not scores:
            return None

        weighted_sum = 0.0
        total_length = 0.0
        for segment, sigma in scores:
            length = float(segment.length)
            if not np.isfinite(length) or length <= 0:
                continue
            weighted_sum += sigma * length
            total_length += length

        if total_length <= 0:
            return None
        return float(weighted_sum / total_length)

    def _sample_eligible_segments(
        self,
        layer: FundusVesselsLayer,
        *,
        n_segments: Optional[int] = None,
        seed: int = 0,
    ) -> List[Segment]:
        """Return eligible segments, optionally a random sample of size n_segments."""
        if self.grid_field_spec is not None:
            if not grid_field_passes_qc(
                layer.retina,
                self.grid_field_spec,
                min_fraction_in_bounds=resolve_min_area_within_bounds(
                    self.grid_field_spec,
                    self.min_area_within_bounds,
                    self.default_min_area_within_bounds,
                ),
            ):
                raise ValueError("Grid field failed QC for profile plotting")

        segments = self._get_segments(layer)
        if not segments:
            raise ValueError("No eligible segments available for profile plotting")

        if n_segments is None or n_segments >= len(segments):
            return list(segments)

        rng = np.random.default_rng(seed)
        chosen_idx = rng.choice(len(segments), size=int(n_segments), replace=False)
        return [segments[int(i)] for i in chosen_idx]

    def _overlay_bilateral_sigma_ranges(
        self,
        axis,
        profiles: np.ndarray,
        *,
        diameter: float,
    ) -> None:
        """Draw per-row σ spans; valid fits cyan/lime, discarded fits red."""
        for row, profile in enumerate(profiles):
            params = fit_bilateral_profile_params(
                profile,
                diameter=diameter,
                smooth_sigma=self.profile_smooth_sigma,
                min_contrast=self.min_contrast,
                width_prior_fraction=self.width_prior_fraction,
            )
            if params is None:
                continue
            valid = bilateral_fit_is_valid(
                params,
                profile_length=len(profile),
                diameter=diameter,
            )
            sigma, _, _, a, b, kind = params
            if kind == "gaussian":
                color = "lime" if valid else "red"
                mu = a
                axis.plot(
                    [mu - sigma, mu + sigma],
                    [row, row],
                    color=color,
                    linewidth=1.6,
                    solid_capstyle="butt",
                    zorder=3,
                )
                axis.plot(
                    mu,
                    row,
                    marker="|",
                    color="white",
                    markersize=7,
                    markeredgewidth=1.2,
                    zorder=4,
                )
                continue
            color = "cyan" if valid else "red"
            for x_edge in (a, b):
                axis.plot(
                    [x_edge - sigma, x_edge + sigma],
                    [row, row],
                    color=color,
                    linewidth=1.6,
                    solid_capstyle="butt",
                    zorder=3,
                )
                axis.plot(
                    x_edge,
                    row,
                    marker="|",
                    color="white",
                    markersize=7,
                    markeredgewidth=1.2,
                    zorder=4,
                )

    def plot_segment_profiles(
        self,
        layer: FundusVesselsLayer,
        *,
        n_segments: Optional[int] = None,
        seed: int = 0,
        ax=None,
    ):
        """Plot 2D profile stacks for eligible segments (all if n_segments is None)."""
        green = self._green_channel(layer)
        if green is None:
            raise ValueError("Retina image is required to plot segment profiles")

        vessels_mask = np.asarray(layer.mask, dtype=bool)
        bounds_mask = self._profile_validity_mask(layer)
        chosen = self._sample_eligible_segments(
            layer, n_segments=n_segments, seed=seed
        )
        n_show = len(chosen)

        ncols = min(5, n_show)
        nrows = int(ceil(n_show / ncols))
        fig = None
        if ax is None:
            fig, ax = plt.subplots(
                nrows,
                ncols,
                figsize=(3.2 * ncols, 3.0 * nrows),
                constrained_layout=True,
            )
        axes = np.atleast_1d(ax).ravel()

        for axis, segment in zip(axes, chosen):
            profiles = self._segment_profiles(
                segment, green, vessels_mask, bounds_mask
            )
            if profiles is None:
                axis.set_axis_off()
                continue
            sigma = self._segment_sigma(
                segment, green, vessels_mask, bounds_mask
            )
            sigma_txt = f"{sigma:.2f}" if np.isfinite(sigma) else "nan"
            im = axis.imshow(
                profiles,
                aspect="auto",
                cmap="magma",
                interpolation="nearest",
            )
            try:
                diameter = float(segment.get_median_diameter())
            except Exception:
                diameter = np.nan
            if np.isfinite(diameter):
                self._overlay_bilateral_sigma_ranges(
                    axis, profiles, diameter=diameter
                )
            axis.set_title(
                f"seg={segment.index}  σ={sigma_txt}\n"
                f"L={segment.length:.0f}px  n={profiles.shape[0]}",
                fontsize=8,
            )
            axis.set_xlabel("perp. px", fontsize=7)
            axis.set_ylabel("profile #", fontsize=7)
            axis.tick_params(labelsize=6)
            plt.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

        for axis in axes[n_show:]:
            axis.set_axis_off()

        if fig is not None:
            fig.suptitle(
                f"Segment profiles (bilateral, n={n_show})",
                y=1.02,
            )
        return fig if fig is not None else ax

    def plot_sampled_segments(
        self,
        layer: FundusVesselsLayer,
        *,
        n_segments: Optional[int] = None,
        seed: int = 0,
        ax=None,
    ):
        """Plot fundus image with eligible segments overlaid (all if n_segments is None)."""
        chosen = self._sample_eligible_segments(
            layer, n_segments=n_segments, seed=seed
        )
        vessels_mask = np.asarray(layer.mask, dtype=bool)
        bounds_mask = self._profile_validity_mask(layer)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        layer.plot(ax=ax, image=True, bounds=True, skeleton=False, mask=False)
        field = self._get_grid_field(layer)
        if field is not None:
            field.plot(ax)
        for segment in chosen:
            skel = np.asarray(segment.skeleton)
            if skel.size == 0:
                continue
            lines = self._segment_profile_lines(
                segment, vessels_mask, bounds_mask
            )
            if lines is not None:
                for line in lines:
                    ax.plot(
                        line[:, 0],
                        line[:, 1],
                        color="cyan",
                        linewidth=0.8,
                        alpha=0.85,
                    )
            mid = skel[len(skel) // 2]
            ax.text(
                float(mid[1]),
                float(mid[0]),
                str(segment.index),
                color="yellow",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="black",
                    edgecolor="none",
                    alpha=0.55,
                ),
            )

        ax.set_title(f"Sampled segments (bilateral, n={len(chosen)}, seed={seed})")
        return fig if fig is not None else ax

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_grid_field_suffix, get_layer_suffix

        field = get_grid_field_suffix(self.grid_field_spec)
        layer = get_layer_suffix(layer_name)
        return f"Sharpness (ESF σ){field}{layer}"

    def feature_name_tokens(self) -> list[str]:
        return ["sharpness"]

    def parameter_name_tokens(self) -> list[str]:
        tokens: list[str] = []
        if self.min_area_within_bounds is not None:
            tokens.extend(
                [
                    "min_area_within_bounds",
                    format_name_value(self.min_area_within_bounds),
                ]
            )
        if self.min_numpoints != self.default_min_numpoints:
            tokens.extend(["min_numpoints", str(self.min_numpoints)])
        if self.n_profiles != self.default_n_profiles:
            tokens.extend(["n_profiles", str(self.n_profiles)])
        if self.profile_smooth_sigma != self.default_profile_smooth_sigma:
            tokens.extend(
                [
                    "profile_smooth_sigma",
                    format_name_value(self.profile_smooth_sigma),
                ]
            )
        if self.min_contrast != self.default_min_contrast:
            tokens.extend(["min_contrast", format_name_value(self.min_contrast)])
        if self.min_diameter != self.default_min_diameter:
            tokens.extend(["min_diameter", format_name_value(self.min_diameter)])
        if self.max_diameter != self.default_max_diameter:
            tokens.extend(["max_diameter", format_name_value(self.max_diameter)])
        if self.width_prior_fraction != self.default_width_prior_fraction:
            tokens.extend(
                [
                    "width_prior_fraction",
                    format_name_value(self.width_prior_fraction),
                ]
            )
        return tokens

    def _plot(self, ax, layer: FundusVesselsLayer, **kwargs):
        layer.plot(ax=ax, image=True, bounds=True, skeleton=False, mask=False)
        field = self._get_grid_field(layer)
        if field is not None:
            field.plot(ax)

        scores = self.segment_scores(layer)
        if not scores:
            return ax

        values = np.asarray([sigma for _, sigma in scores], dtype=float)
        vmin = float(np.nanpercentile(values, 5))
        vmax = float(np.nanpercentile(values, 95))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.5, 3.0

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis")
        for segment, sigma in scores:
            skel = np.asarray(segment.skeleton)
            if skel.size == 0:
                continue
            ax.plot(
                skel[:, 1],
                skel[:, 0],
                color=cmap(norm(sigma)),
                linewidth=1.2,
                alpha=0.95,
            )

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="ESF σ (px)")
        return ax
