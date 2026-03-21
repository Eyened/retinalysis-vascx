from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from rtnls_enface.base import Circle, Point
from scipy.interpolate import UnivariateSpline

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

        diffs = np.diff(self.points[:, 0])
        sigma_sq = np.mean(diffs**2) / 2
        s = sigma_sq * len(self.points)
        self.y_spline = UnivariateSpline(
            distance_normalized, self.points[:, 0], k=3, s=s
        )

        diffs = np.diff(self.points[:, 1])
        sigma_sq = np.mean(diffs**2) / 2
        s = sigma_sq * len(self.points)
        self.x_spline = UnivariateSpline(
            distance_normalized, self.points[:, 1], k=3, s=s
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
        """Approximate circle intersections using the segment polyline."""
        points = self.points
        if len(points) < 2:
            return None, None

        center = np.array(circle.center.tuple, dtype=float)
        radius_sq = float(circle.r) ** 2
        distances = np.sum((points - center) ** 2, axis=1) - radius_sq

        seg_lengths = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        total_length = cumulative[-1]
        if total_length <= 0:
            return None, None

        t_intersection = []
        intersection_points = []
        tol = 1e-9

        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            d0 = distances[i]
            d1 = distances[i + 1]
            segment_length = seg_lengths[i]

            if abs(d0) <= tol:
                t_intersection.append(cumulative[i] / total_length)
                intersection_points.append(Point(*p0))

            if abs(d1) <= tol:
                t_intersection.append(cumulative[i + 1] / total_length)
                intersection_points.append(Point(*p1))
                continue

            if d0 * d1 > 0 or segment_length <= 0:
                continue

            alpha = abs(d0) / (abs(d0) + abs(d1))
            point = p0 + alpha * (p1 - p0)
            t = (cumulative[i] + alpha * segment_length) / total_length
            t_intersection.append(t)
            intersection_points.append(Point(*point))

        if len(t_intersection) == 0:
            return None, None

        rounded_t = np.round(np.asarray(t_intersection), 5)
        _, unique_indices = np.unique(rounded_t, return_index=True)
        unique_indices = np.sort(unique_indices)
        t_intersection = rounded_t[unique_indices]
        intersection_points = [intersection_points[i] for i in unique_indices]

        return intersection_points, t_intersection
