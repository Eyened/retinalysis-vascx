from __future__ import annotations

import warnings
from functools import cached_property, lru_cache
from math import ceil
from typing import TYPE_CHECKING, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from rtnls_enface.base import Line, Point, TuplePoint
from scipy.spatial.distance import euclidean as distance_2p
from sklearn.linear_model import TheilSenRegressor

from vascx.shared.diameters import DiameterMeasurement, retipy_vessel_diameters
from vascx.shared.splines import SplineInterpolation
from vascx.utils.plotting import find_bounding_box

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


class Segment:
    def __init__(self, skeleton: List[TuplePoint], edge=None):
        self.id = None
        self.index = None

        self.skeleton = skeleton
        self.edge = edge
        self._pixels = None
        self.layer: VesselTreeLayer = None

        self._spline = None
        self._diameter_measurements: List[DiameterMeasurement] = None
        self._mean_diameter: float = None
        self._median_diameter: float = None

        self._agg_mean_diameter: float = None
        self._agg_length: float = None
        self._mean_xy: Tuple[float, float] = None
        self._thickest_diam: float = 0

        # depth of the edge in the tree structure
        self._depth = None

        self.original_segments = None
        self._get_spline_cached = lru_cache(maxsize=None)(self._build_spline)
        self._get_diameter_measurements_cached = lru_cache(maxsize=None)(
            self._build_diameter_measurements
        )

    @cached_property
    def start(self) -> Point:
        return Point(*self.skeleton[0])

    @cached_property
    def end(self) -> Point:
        return Point(*self.skeleton[-1])

    @cached_property
    def pixels(self) -> List[Tuple[int, int]]:
        if self._pixels is not None:
            return self._pixels
        if self.layer is None:
            raise ValueError("Segment.layer is not set")
        return self.layer.get_segment_pixels(self)

    def _build_spline(self, error_fraction: float = 0.05) -> SplineInterpolation | None:
        """Build a spline for the segment using the requested smoothing."""
        if len(self.skeleton) <= 4:
            return None
        return SplineInterpolation(self, error_fraction=error_fraction)

    def get_spline(self, error_fraction: float = 0.05) -> SplineInterpolation | None:
        """Return a cached spline for the requested smoothing."""
        return self._get_spline_cached(float(error_fraction))

    @property
    def spline(self) -> SplineInterpolation | None:
        return self.get_spline()

    def _build_diameter_measurements(
        self, error_fraction: float = 0.05
    ) -> List[DiameterMeasurement]:
        assert self.layer is not None
        if len(self.skeleton) <= 4:
            # segment is too short for cubic spline, use retipy method.
            measurements = self.calc_diameters_retipy()
        else:
            try:
                measurements = self.calc_diameters_using_splines(
                    error_fraction=error_fraction
                )
            except Exception:
                warnings.warn(
                    "Exception when using splines for diameter calculation, "
                    "falling back to retipy."
                )
                # default to using retipy if there are exceptions when using splines
                measurements = self.calc_diameters_retipy()

            if len(measurements) == 0:
                measurements = self.calc_diameters_retipy()

        return measurements

    def get_diameter_measurements(
        self, error_fraction: float = 0.05
    ) -> List[DiameterMeasurement]:
        """Return cached diameter measurements for the requested smoothing."""
        return self._get_diameter_measurements_cached(float(error_fraction))

    def get_diameters(self, error_fraction: float = 0.05) -> List[float]:
        """Return per-sample diameters for the requested smoothing."""
        return [m.diameter for m in self.get_diameter_measurements(error_fraction)]

    def get_mean_diameter(self, error_fraction: float = 0.05) -> float:
        """Return mean diameter for the requested smoothing."""
        return np.mean(self.get_diameters(error_fraction))

    def get_median_diameter(self, error_fraction: float = 0.05) -> float:
        """Return median diameter for the requested smoothing."""
        return np.median(self.get_diameters(error_fraction))

    @cached_property
    def length(self) -> float:
        distance = 0
        x, y = zip(*self.skeleton)
        for i in range(0, len(x) - 1):
            distance += distance_2p([x[i], y[i]], [x[i + 1], y[i + 1]])
        return distance

    @cached_property
    def chord_length(self) -> float:
        return distance_2p(self.skeleton[0], self.skeleton[-1])

    @cached_property
    def diameter_measurements(self) -> List[DiameterMeasurement]:
        return self.get_diameter_measurements()

    @cached_property
    def diameters(self) -> List[float]:
        return self.get_diameters()

    @cached_property
    def mean_diameter(self) -> float:
        return self.get_mean_diameter()

    @cached_property
    def median_diameter(self) -> float:
        return self.get_median_diameter()

    @cached_property
    def mean_xy(self) -> float:
        return self.get_mean_xy()

    @cached_property
    def profile(self) -> np.ndarray:
        """Return the intensity profile of the segment"""
        if self.layer.retina.image is None:
            raise ValueError("Image not set in retina")

        return (
            self.spline.profile(
                self.layer.retina.grayscale, L=2 * ceil(self.median_diameter)
            )
            if self.spline is not None
            else None
        )

    def _invalidate_geometry_caches(self) -> None:
        """Clear cached geometry derived from the segment skeleton."""
        for attr in (
            "start",
            "end",
            "pixels",
            "length",
            "chord_length",
            "diameter_measurements",
            "diameters",
            "mean_diameter",
            "median_diameter",
            "mean_xy",
            "profile",
        ):
            self.__dict__.pop(attr, None)
        self._get_spline_cached.cache_clear()
        self._get_diameter_measurements_cached.cache_clear()

    def reverse(self):
        self.skeleton = np.flip(self.skeleton, axis=0)
        self._invalidate_geometry_caches()

    def filter_outliers(self, measurements: List[DiameterMeasurement]):
        diams = np.array([r.diameter for r in measurements])

        regressor = TheilSenRegressor()
        X = np.arange(0, len(diams))[:, None]
        regressor.fit(X, diams)
        y_pred = regressor.predict(X)
        residuals = np.abs(diams - y_pred)
        MAD = np.median(np.abs(residuals - np.median(residuals)))
        threshold = 9 * MAD
        inliers = np.where(residuals < threshold)[0].tolist()
        return [measurements[i] for i in inliers]

    def calc_diameters_using_splines(self, n_points=None, error_fraction: float = 0.05):
        if n_points is None:
            n_points = max(10, round(self.length / 10))

        spline = self.get_spline(error_fraction=error_fraction)
        if spline is None:
            raise RuntimeError("Segment is too short for spline diameter calculation")

        measurements = spline.diameters(n_points)

        # 30 pixels is a safe maximum
        measurements = [d for d in measurements if d.diameter < 30]
        if len(measurements) <= 5:
            raise RuntimeError("Not enough valid measurements for diameter")
        # we use theilsenregressor to find outliers

        measurements = self.filter_outliers(measurements)
        return measurements

    def calc_diameters_retipy(self):
        measurements = retipy_vessel_diameters(self, self.layer.mask)
        measurements = [d for d in measurements if d.diameter < 30]
        return measurements

    def get_mean_xy(self):
        return np.mean(self.skeleton, axis=0)

    def mean_position(self) -> Point:
        return Point(*np.mean(self.skeleton, axis=0))

    def orientation(self):
        d1 = self.start.distance_to(self.layer.retina.disc.center_of_mass)
        d2 = self.end.distance_to(self.layer.retina.disc.center_of_mass)

        line1 = Line(
            self.layer.retina.disc.center_of_mass, self.layer.retina.fovea_location
        )

        if d1 < d2:
            line2 = Line(self.start, self.end)
        else:
            line2 = Line(self.end, self.start)

        return line1.angle_to(line2)

    def fod_angle(self) -> Union[float, None]:
        if self.layer.retina.fovea_location is None or self.layer.retina.disc is None:
            return None
        line1 = Line(
            self.layer.retina.disc.center_of_mass, self.layer.retina.fovea_location
        )
        line2 = Line(self.layer.retina.disc.center_of_mass, self.mean_position())
        return line1.angle_to(line2)

    def __eq__(self, other):
        # segments are equal if their ends are equal.
        if isinstance(other, Segment):
            return self.start == other.start and self.end == other.end
        return False

    def __hash__(self):
        return hash((self.start, self.end))

    def __repr__(self):
        return str(self.edge)

    def intersections_with_circle(
        self, circle, error_fraction: float = 0.05
    ) -> Tuple[List[Point], List[float]]:
        spline = self.get_spline(error_fraction=error_fraction)
        if spline is None:
            return None, None
        return spline.intersections_with_circle(circle)

    def plot(
        self,
        ax=None,
    ):
        self.layer.plot_segments(ax, segments=[self])

    def plot_skeleton(
        self,
        ax=None,
        fig=None,
    ):
        top_left, bottom_right = find_bounding_box(self.skeleton)

        imsize = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])

        if ax is None:
            fig, ax = plt.subplots(
                1, 1, dpi=300, figsize=(4, 4 * imsize[1] / imsize[0])
            )
            ax.set_axis_off()

        ax.imshow(np.zeros(imsize, dtype=np.uint8))

        def plot_seg_skeleton(skl, color="white"):
            for i in range(len(skl) - 1):
                ax.plot(
                    [
                        skl[i][1] - top_left[1],
                        skl[i + 1][1] - top_left[1],
                    ],
                    [
                        skl[i][0] - top_left[0],
                        skl[i + 1][0] - top_left[0],
                    ],
                    marker=None,
                    linewidth=0.2,
                    color=color,
                )

                ax.plot(
                    skl[0][1] - top_left[1],
                    skl[0][0] - top_left[0],
                    marker="o",
                    markersize=1,
                    color="green",
                )
                ax.plot(
                    skl[-1][1] - top_left[1],
                    skl[-1][0] - top_left[0],
                    marker="o",
                    markersize=1,
                    color="red",
                )

        plot_seg_skeleton(self.skeleton, color="white")
        if self.original_segments is not None:
            colors = ["red", "green", "blue"]
            for i, seg in enumerate(self.original_segments):
                plot_seg_skeleton(seg.skeleton, color=colors[i % len(colors)])

    def plot_mask(self):
        top_left, bottom_right = find_bounding_box(self.skeleton)
        return self.layer.plot(
            xlim=(top_left[0], bottom_right[0]), ylim=(top_left[1], bottom_right[1])
        )

    def plot_graph(self):
        top_left, bottom_right = find_bounding_box(self.skeleton)
        return self.layer.plot_graph(
            xlim=(top_left[1], bottom_right[1]), ylim=(top_left[0], bottom_right[0])
        )

    def split_equal_length(self, max_length: float) -> List["Segment"]:
        """Split the segment into equal-length contiguous subsegments."""
        max_length = float(max_length)
        if not np.isfinite(max_length) or max_length <= 0:
            raise ValueError("max_length must be positive and finite")

        if len(self.skeleton) < 2 or self.length <= max_length:
            return [self]

        points = np.asarray(self.skeleton, dtype=float)
        seg_lengths = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        n_segments = ceil(self.length / max_length)
        split_distances = np.linspace(0.0, self.length, n_segments + 1)

        def point_at_distance(distance: float) -> np.ndarray:
            if distance <= 0:
                return points[0].copy()
            if distance >= self.length:
                return points[-1].copy()

            idx = np.searchsorted(cumulative, distance, side="right") - 1
            idx = min(max(idx, 0), len(seg_lengths) - 1)
            segment_length = seg_lengths[idx]
            if segment_length <= 0:
                return points[idx].copy()

            alpha = (distance - cumulative[idx]) / segment_length
            return points[idx] + alpha * (points[idx + 1] - points[idx])

        split_segments: List[Segment] = []
        for start_distance, end_distance in zip(
            split_distances[:-1], split_distances[1:]
        ):
            piece_points = [point_at_distance(start_distance)]
            interior_mask = (cumulative > start_distance) & (cumulative < end_distance)
            piece_points.extend(points[interior_mask])
            piece_points.append(point_at_distance(end_distance))

            deduplicated_points = [piece_points[0]]
            for point in piece_points[1:]:
                if not np.allclose(point, deduplicated_points[-1]):
                    deduplicated_points.append(point)

            if len(deduplicated_points) < 2:
                continue

            split_segment = Segment(
                skeleton=[tuple(point) for point in deduplicated_points],
                edge=self.edge,
            )
            split_segment.layer = self.layer
            split_segment.original_segments = (
                self.original_segments if self.original_segments is not None else [self]
            )
            split_segments.append(split_segment)

        return split_segments or [self]


def concatenate(*lists):
    """Join an arbitrary number of lists together."""
    result = []
    for lst in lists:
        result.extend(lst)
    return result


def merge_segments(*segments) -> Segment:
    assert len(segments) > 0, "Must provide at least one segment to merge"

    skeleton = np.concatenate([s.skeleton for s in segments], axis=0)
    seg = Segment(skeleton, edge=(segments[0].edge[0], segments[-1].edge[1]))
    seg.layer = segments[0].layer

    return seg
