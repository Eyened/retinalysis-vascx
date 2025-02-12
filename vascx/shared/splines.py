from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from scipy.spatial.distance import euclidean as distance_2p

from rtnls_enface.base import Circle, Point
from vascx.shared.diameters import DiameterMeasurement, find_vessel_edge
from vascx.utils import linear_interpolate

if TYPE_CHECKING:
    from vascx.shared.segment import Segment


class SplineInterpolation:
    """
    Class for spline interpolation of a segment.
    """

    def __init__(self, segment, error_fraction=0.05):
        """
        Args:
            segment: The segment to interpolate.
            error_fraction: The maximum allowed error between the spline and original points
                            as a fraction of the segment length.
        """

        self.segment: Segment = segment
        self.error_fraction: float = error_fraction

        self.points = np.array(segment.skeleton, dtype=float)
        self.points += 0.5  # center of pixels
        # print(len(self.points))

        # distance from start
        distance = np.cumsum(np.sqrt(np.sum(np.diff(self.points, axis=0) ** 2, axis=1)))
        self.total_distance = distance[-1]
        self.smoothing = error_fraction * self.total_distance

        # pixel locations in interval 0-1
        distance_normalized = np.insert(distance, 0, 0) / self.total_distance

        self.y_spline = UnivariateSpline(
            distance_normalized, self.points[:, 0], k=3, s=self.smoothing
        )
        self.x_spline = UnivariateSpline(
            distance_normalized, self.points[:, 1], k=3, s=self.smoothing
        )

        if np.isnan(self.y_spline(distance_normalized)).any():
            raise RuntimeError("Spline was not fit correctly")
        if np.isnan(self.x_spline(distance_normalized)).any():
            raise RuntimeError("Spline was not fit correctly")

        self.dy = self.y_spline.derivative()
        self.dx = self.x_spline.derivative()

        self.dyy = self.dy.derivative()
        self.dxx = self.dx.derivative()

    def get_point(self, t: float) -> np.ndarray:
        """
        Args:
            t: The parameter value along the spline (0 - 1).

        Returns:
            Returns the interpolated point on the spline at parameter t.
        """
        return np.array([self.y_spline(t), self.x_spline(t)])

    def get_point_pixels(self, t_pixels: float) -> np.ndarray:
        """
        Args:
            t: The parameter value along the spline (0 - 1).

        Returns:
            Returns the interpolated point on the spline at parameter t.
        """
        return np.array(
            [
                self.y_spline(t_pixels / self.total_distance),
                self.x_spline(t_pixels / self.total_distance),
            ],
            dtype=int,
        )

    def get_perpendicular(self, t: float) -> np.ndarray:
        """
        Args:
            t: The parameter value along the spline (0 - 1).

        Returns:
            The unit vector representing the perpendicular direction to the spline at parameter t.
        """
        dx_dt = self.dx(t)
        dy_dt = self.dy(t)
        p = np.array([-dx_dt, dy_dt])
        length = np.sqrt(p.dot(p))
        return p / length  # yx

    def length(self, every=1, min_values=10):
        n_points = max(round(self.total_distance / every), min_values)
        t = np.linspace(0, 1, n_points)
        x, y = self.x_spline(t), self.y_spline(t)
        points = np.stack([x, y], axis=1)
        length = np.sum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        return length

    def diameters(
        self, n_points: int
    ) -> Tuple[SplineInterpolation, List[DiameterMeasurement]]:
        """
        Interpolates a segment using a spline and evaluates segment properties at multiple points.

        Args:
            vessels: A numpy array representing the vessels.
            segment: The segment object to interpolate.
            n_points: The number of points to evaluate along the segment.

        Returns:
            A tuple containing the spline interpolation object and a list of interpolated segment properties.
        """
        mask = self.segment.layer.mask.copy()  # vessels.copy()

        def evaluate(t):
            origin = self.get_point(t)
            direction = self.get_perpendicular(t)
            edge_0 = find_vessel_edge(mask, origin, direction)
            edge_1 = find_vessel_edge(mask, origin, -direction)
            diameter = np.sqrt(np.sum((edge_0 - edge_1) ** 2))
            return DiameterMeasurement(origin, edge_0, edge_1, diameter)

        return [evaluate(t) for t in np.linspace(0, 1, n_points)]

    def profile(self, image: np.ndarray, N=50, L=15):
        def evaluate(t):
            origin = self.get_point(t)
            u = self.get_perpendicular(t)  # unit vector
            # print(u)
            profile = np.zeros(2 * L + 1)

            for i in range(-L, L + 1):
                sample_point = (
                    origin + i * u
                )  # Calculate the sample point along the direction
                # print(sample_point)
                profile[i + L] = linear_interpolate(
                    image, sample_point[1], sample_point[0]
                )  # Interpolate the value at the sample point

            return profile

        return np.stack([evaluate(t) for t in np.linspace(0, 1, N)])

    def curvatures(self, every=5, min_values=10):
        n_points = max(round(self.total_distance / every), min_values)

        def evaluate(t):
            dx = self.dx(t)
            dy = self.dy(t)
            dxx = self.dxx(t)
            dyy = self.dyy(t)

            return np.abs(dx * dyy - dy * dxx) / ((dx**2 + dy**2) ** (3 / 2))

        return [evaluate(t) for t in np.linspace(0, 1, n_points + 2)[1:-1]]

    def inflection_points(self, every=50, min_values=0):
        n_points = max(round(self.total_distance / every), min_values)

        t_vals = np.linspace(0, 1, n_points + 2)

        return np.where(np.diff(np.sign(self.dxx(t_vals) * self.dyy(t_vals))))[0]

    def count_inflection_points(self):
        pass

    def intersections_with_circle(self, circle: Circle):
        """return all intersections of the spline with the given circle"""

        def distance_to_circle_center(t):
            x, y = self.x_spline(t).item(), self.y_spline(t).item()
            point_on_curve = np.array([x, y])  # Point on the curve
            dist = distance_2p(point_on_curve, circle.center.tuple_xy)

            return dist - circle.r

        initial_t_guesses = np.linspace(0, 1, 10)

        # solve for each initial guess

        t_intersection = []
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            for t_guess in initial_t_guesses:
                try:
                    t_intersection.append(fsolve(distance_to_circle_center, t_guess)[0])
                except RuntimeWarning:
                    continue  # skip warnings when fsolve gets stuck

        # check that the solutions are roots of distance_to_circle_center
        # fsolve may return non-roots if it gets stuck in minima, maxima
        t_intersection = np.array(
            [t for t in t_intersection if np.isclose(distance_to_circle_center(t), 0)]
        )

        # remove t-values outside of [0,1]
        t_intersection = t_intersection[(t_intersection >= 0) & (t_intersection <= 1)]

        if len(t_intersection) == 0:
            return None, None

        t_intersection = np.unique(np.round(t_intersection, 5))

        intersection_points = [
            Point(self.y_spline(t), self.x_spline(t)) for t in t_intersection
        ]
        return intersection_points, t_intersection
