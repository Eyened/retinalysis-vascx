from typing import List, Set

from rtnls_enface.base import PixelGraph, TuplePoint


# import vascx.segment

# 8 neigbors of a pixel (including diagonal)
deltas_8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def make_adjacency_graph_8(vertices: List[TuplePoint]) -> PixelGraph:
    """
    Creates an adjacency graph for 8-connected vertices.

    Args:
        vertices: A list of Points representing the pixels of the skeleton.

    Returns:
        A PixelGraph representing the adjacency graph of the pixels.
    """
    return {
        (x, y): {
            (x + dx, y + dy) for dx, dy in deltas_8 if (x + dx, y + dy) in vertices
        }
        for x, y in vertices
    }


def connect_segment(points: Set[TuplePoint], ends: Set[TuplePoint]) -> List[TuplePoint]:
    """
    Connects the pixels in a segment and returns a list of connected pixels.

    Args:
        segment: A segment object containing pixels and ends.

    Returns:
        An ordered list of points representing the connected pixels.
    """

    remaining = set(points)
    start = next(iter(ends))
    result = [start]
    current = start
    remaining.remove(start)
    while remaining:
        x, y = current
        for dx, dy in deltas_8:
            next_pixel = x + dx, y + dy
            if next_pixel in remaining:
                remaining.remove(next_pixel)
                current = next_pixel
                result.append(next_pixel)
                break

    return result


def plot_all_nodes(ax, nodes):
    endpoints_x = [n[1] for n in nodes if n[2] == 1]
    endpoints_y = [n[0] for n in nodes if n[2] == 1]

    bifurcations_x = [n[1] for n in nodes if n[2] == 3]
    bifurcations_y = [n[0] for n in nodes if n[2] == 3]

    crossings_x = [n[1] for n in nodes if n[2] == 4]
    crossings_y = [n[0] for n in nodes if n[2] == 4]

    ax.scatter(x=endpoints_x, y=endpoints_y, marker="o", c="b", s=3)
    ax.scatter(x=bifurcations_x, y=bifurcations_y, marker="^", c="r", s=3)
    ax.scatter(x=crossings_x, y=crossings_y, marker="^", c="r", s=3)
