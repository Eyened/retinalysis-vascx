from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from rtnls_enface.types import Line, Point, TuplePoint
from scipy.spatial.distance import euclidean as distance_2p
from sklearn.linear_model import TheilSenRegressor

from vascx.diameters import (
    DiameterMeasurement,
    retipy_vessel_diameters,
    segment_interpolate,
)
from vascx.splines import SplineInterpolation
from vascx.utils.plotting import find_bounding_box
from vascx.splines import SplineInterpolation
from vascx.utils.plotting import find_bounding_box

if TYPE_CHECKING:
    from vascx.layer import Layer


class Segment:
    def __init__(
        self, skeleton: List[TuplePoint] = None, start: Point = None, end: Point = None
    ):
        self.id = None
        self.index = None
        self.start = start
        self.end = end

        if distance_2p(self.start.tuple, skeleton[0]) > distance_2p(
            self.end.tuple, skeleton[0]
        ):
            skeleton = np.flip(skeleton, axis=0)
        self.skeleton = skeleton

        self.connectors = frozenset([start, end])

        self.pixels = []
        self.neighbors: List[Segment] = []
        self.mask: List[TuplePoint] = []

        self.layer: Layer = None

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

    def reverse(self):
        self.start, self.end = self.end, self.start
        self.skeleton = np.flip(self.skeleton, axis=0)

    def set_diameters(self, n_points=None):
        assert self.layer is not None
        if len(self.skeleton) <= 4:
            # segment is too short for cubic spline, use retipy method.
            segment_diameters = retipy_vessel_diameters(self, self.layer.mask)
        else:
            if n_points is None:
                n_points = max(10, round(100 * self.length / 1024))
            self._spline, measurements = segment_interpolate(
                self.layer.binary, self, n_points
            )

            # 30 pixels is a safe maximum
            measurements = [d for d in measurements if d.diameter < 30]
            if len(measurements) <= 5:
                segment_diameters = retipy_vessel_diameters(self, self.layer.mask)
            else:
                # we use theilsenregressor to find outliers
                diams = np.array([r.diameter for r in measurements])

                regressor = TheilSenRegressor()
                X = np.arange(0, len(diams))[:, None]
                regressor.fit(X, diams)
                y_pred = regressor.predict(X)
                residuals = np.abs(diams - y_pred)
                MAD = np.median(np.abs(residuals - np.median(residuals)))
                threshold = 9 * MAD
                inliers = np.where(residuals < threshold)[0].tolist()

                measurements = [measurements[i] for i in inliers]
                self._diameter_measurements = measurements

                segment_diameters = [
                    r.diameter * self.layer.retina.scaling_factor for r in measurements
                ]

            if len(segment_diameters) == 0:
                segment_diameters = retipy_vessel_diameters(self, self.layer.mask)

        self._median_diameter = np.median(segment_diameters)
        self._mean_diameter = np.mean(segment_diameters)

    def get_mean_xy(self):
        return np.mean(self.pixels, axis=0)

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
        return distance * self.layer.retina.scaling_factor

    @property
    def chord_length(self) -> float:
        return (
            distance_2p(self.start.tuple, self.end.tuple)
            * self.layer.retina.scaling_factor
        )

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

    def fod_angle(self) -> float | None:
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
            return self.connectors == other.connectors
        return False

    def __hash__(self):
        return hash(self.connectors)

    def __repr__(self):
        return f"ID:{self.id}"

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
    seg = Segment(skeleton, start=segments[0].start, end=segments[-1].end)

    # seg.id = segments[0].id
    seg.pixels = concatenate(*[s.pixels for s in segments])
    seg.neighbors = concatenate(*[s.neighbors for s in segments])
    seg.layer = segments[0].layer
    seg.original_segments = segments

    return seg
