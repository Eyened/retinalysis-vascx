from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from rtnls_enface.base import TuplePoint
from vascx.splines import SplineInterpolation

if TYPE_CHECKING:
    from vascx.layer import Segment


@dataclass
class DiameterMeasurement:
    origin: TuplePoint
    edge_0: TuplePoint
    edge_1: TuplePoint
    diameter: float


def find_vessel_edge(
    grid: np.ndarray,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    eps: float = 0.01,
) -> np.ndarray:
    """
    Finds the edge of a vessel in a grid using a ray casting method.

    Args:
        grid: A numpy array representing the grid of the vessel.
        ray_origin: The origin of the ray as a numpy array (2D point).
        ray_direction: The direction of the ray as a numpy array (2D vector).
        eps: A small value used to ensure termination.

    Returns:
        The coordinate of the vessel edge as a numpy array (2D point).
    """
    h, w = grid.shape
    ry, rx = ray_direction
    oy, ox = ray_origin
    cy, cx = ray_origin
    iy, ix = int(cy), int(cx)

    while iy >= 0 and iy < h and ix >= 0 and ix < w and grid[iy, ix]:
        fract_y = cy % 1
        fract_x = cx % 1

        if ry == 0:
            fy = 1
        elif ry > 0:
            fy = (1 - fract_y) / ry
        else:
            fy = -fract_y / ry

        if rx == 0:
            fx = 1
        elif rx > 0:
            fx = (1 - fract_x) / rx
        else:
            fx = -fract_x / rx

        step = max(min(fy, fx), eps)
        cy += step * ry
        cx += step * rx

        iy, ix = int(cy), int(cx)

    if ry == 0:
        fy = 1
    elif ry > 0:
        fy = (np.floor(cy) - oy) / ry
    else:
        fy = (np.ceil(cy) - oy) / ry
    if rx == 0:
        fx = 1
    elif rx > 0:
        fx = (np.floor(cx) - ox) / rx
    else:
        fx = (np.ceil(cx) - ox) / rx

    return ray_origin + max(fy, fx) * ray_direction


def segment_interpolate(
    vessels: np.ndarray, segment: Segment, n_points: int
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
    spline = SplineInterpolation(segment)

    vessels = vessels.copy()

    def evaluate(t):
        origin = spline.get_point(t)
        direction = spline.get_perpendicular(t)
        edge_0 = find_vessel_edge(vessels, origin, direction)
        edge_1 = find_vessel_edge(vessels, origin, -direction)
        diameter = np.sqrt(np.sum((edge_0 - edge_1) ** 2))
        return DiameterMeasurement(origin, edge_0, edge_1, diameter)

    return spline, [evaluate(t) for t in np.linspace(0, 1, n_points)]


def retipy_vessel_diameters(
    segment: Segment, mask: np.ndarray
) -> List[DiameterMeasurement]:
    mask[0, :] = False
    mask[:, 0] = False
    mask[-1, :] = False
    mask[:, -1] = False

    widths = []
    for i, j in segment.skeleton:
        if not mask[i, j]:
            continue
        w0 = 0
        w45 = 0
        w90 = 0
        w135 = 0
        w180 = 0
        w225 = 0
        w270 = 0
        w315 = 0
        while True:
            if mask[i, j + w0 + 1]:
                w0 += 1
            if mask[i, j - w180 - 1]:
                w180 += 1
            if mask[i - w90 - 1, j]:
                w90 += 1
            if mask[i + w270 + 1, j]:
                w270 += 1
            if mask[i - w45 - 1, j + w45 + 1]:
                w45 += 1
            if mask[i + w225 + 1, j - w225 - 1]:
                w225 += 1
            if mask[i - w135 - 1, j - w135 - 1]:
                w135 += 1
            if mask[i + w315 + 1, j + w315 + 1]:
                w315 += 1

            if mask[i, j + w0 + 1] == False and mask[i, j - w180 - 1] == False:
                widths.append(
                    DiameterMeasurement(
                        (i, j), (i, j + w0 + 1), (i, j - w180 - 1), w0 + w180 + 1
                    )
                )
                break
            elif mask[i - w90 - 1, j] == False and mask[i + w270 + 1, j] == False:
                widths.append(
                    DiameterMeasurement(
                        (i, j), (i - w90 - 1, j), (i + w270 + 1, j), w90 + w270 + 1
                    )
                )
                break
            elif (
                mask[i - w45 - 1, j + w45 + 1] == False
                and mask[i + w225 + 1, j - w225 - 1] == False
            ):
                widths.append(
                    DiameterMeasurement(
                        (i, j),
                        (i - w45 - 1, j + w45 + 1),
                        (i + w225 + 1, j - w225 - 1),
                        w45 + w225 + 1,
                    )
                )
                break
            elif (
                mask[i - w135 - 1, j - w135 - 1] == False
                and mask[i + w315 + 1, j + w315 + 1] == False
            ):
                widths.append(
                    DiameterMeasurement(
                        (i, j),
                        (i - w135 - 1, j - w135 - 1),
                        (i + w315 + 1, j + w315 + 1),
                        w135 + w315 + 1,
                    )
                )
                break
    return widths
