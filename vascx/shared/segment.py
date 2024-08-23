from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean as distance_2p
from sklearn.linear_model import TheilSenRegressor

from rtnls_enface.base import Line, Point, TuplePoint
from vascx.shared.diameters import (
    DiameterMeasurement,
    retipy_vessel_diameters,
    segment_interpolate,
)
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

        self.layer: VesselTreeLayer = None

        self._spline = None
        self._diameter_measurements: List[DiameterMeasurement] = None
        self._mean_diameter: float = None
        self._median_diameter: float = None

        self._agg_mean_diameter: float = None
        self._agg_length: float = None
        self._mean_xy: Tuple[float, float] = None
        self._thickest_diam: float = 0

        self.rank: int = None
        self.visited = False
        self.original_segments = None

    @property
    def start(self) -> Point:
        return Point(*self.skeleton[0])

    @property
    def end(self) -> Point:
        return Point(*self.skeleton[-1])

    @property
    def pixels(self) -> List[Tuple[int, int]]:
        return self.layer.get_segment_pixels(self)

    @property
    def spline(self) -> SplineInterpolation:
        if self._spline is None:
            self.set_diameters()
        return self._spline

    @property
    def length(self) -> float:
        distance = 0
        x, y = zip(*self.skeleton)
        for i in range(0, len(x) - 1):
            distance += distance_2p([x[i], y[i]], [x[i + 1], y[i + 1]])
        return distance

    @property
    def chord_length(self) -> float:
        return distance_2p(self.skeleton[0], self.skeleton[-1])

    @property
    def mean_diameter(self) -> float:
        if self._mean_diameter is None:
            self.set_diameters()
        return self._mean_diameter

    @property
    def median_diameter(self) -> float:
        if self._median_diameter is None:
            self.set_diameters()
        return self._median_diameter

    @property
    def diameters(self) -> float:
        if self._diameter_measurements is None:
            self.set_diameters()
        return self._diameter_measurements

    @property
    def mean_xy(self) -> float:
        if self._mean_xy is None:
            self._mean_xy = self.get_mean_xy()
        return self._mean_xy

    def reverse(self):
        self.skeleton = np.flip(self.skeleton, axis=0)

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

    def calc_diameters_using_splines(self, n_points):
        if n_points is None:
            n_points = max(10, round(100 * self.length / 1024))
        self._spline, measurements = segment_interpolate(
            self.layer.binary, self, n_points
        )

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

    def set_diameters(self, n_points=None):
        assert self.layer is not None
        if len(self.skeleton) <= 4:
            # segment is too short for cubic spline, use retipy method.
            measurements = self.calc_diameters_retipy()
        else:
            try:
                measurements = self.calc_diameters_using_splines(n_points)
            except Exception:
                warnings.warn(
                    "Exception when using splines for diameter calculation, "
                    "falling back to retipy."
                )
                # default to using retipy if there are exceptions when using splines
                measurements = self.calc_diameters_retipy()

            if len(measurements) == 0:
                measurements = self.calc_diameters_retipy()

        segment_diameters = [r.diameter for r in measurements]
        self._diameter_measurements = measurements
        self._median_diameter = np.median(segment_diameters)
        self._mean_diameter = np.mean(segment_diameters)

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

    def intersections_with_circle(self, circle) -> Tuple[List[Point], List[float]]:
        if self.spline is None:
            return None, None
        return self.spline.intersections_with_circle(circle)

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
