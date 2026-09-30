"""ARIC-style trunk selection with standard VascX segment median calibers for ``CRE``.

ARIC NCS vessel measurement protocol, pp. 17–18, 24–25:
https://aric.cscc.unc.edu/aric9/sites/default/files/public/visitdocuments/v5/14b1%20ARIC%20NCS%20VM%20Protocol%2011.23.10.pdf

The graph supplies anatomical hypotheses, not grader-confirmed vessel identities.
Zone B determines trunk eligibility; caliber uses the original segment's safeguarded
median, including measurements outside Zone B. We assume minor within-segment tapering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from vascx.shared.segment import Segment

if TYPE_CHECKING:
    from .cre import CRE
    from vascx.fundus.layer import VesselTreeLayer


@dataclass
class ARICTrunk:
    """One anatomical candidate with in-zone display pieces and whole-segment caliber."""

    segment: Segment
    pieces: list[Segment]
    median_diameter: float
    replacement_for: tuple | None = None


@dataclass
class ARICResult:
    candidates: list[ARICTrunk] = field(default_factory=list)
    selected: list[ARICTrunk] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    coverage: float = 0.0
    valid: bool = False


def _radial_mask(feature: CRE, retina, y, x):
    center = retina.disc.center
    diameter = 2 * retina.disc.circle.r
    distance = (y - center.y) ** 2 + (x - center.x) ** 2
    return ((feature.inner_circle * diameter) ** 2 <= distance) & (
        distance < (feature.outer_circle * diameter) ** 2
    )


def _regional_mask(feature: CRE, retina, y, x):
    """Same angular regions and OD–fovea hemifields as the circle CRE variants."""
    mask = np.ones(np.broadcast_shapes(np.shape(y), np.shape(x)), dtype=bool)
    if feature.CREMode.value == "temporal":
        center, fovea = retina.disc.center, retina.fovea_location
        vy, vx = fovea.y - center.y, fovea.x - center.x
        norm = np.hypot(vx, vy)
        origin_y = center.y - 0.5 * retina.disc.circle.r * vy / norm
        origin_x = center.x - 0.5 * retina.disc.circle.r * vx / norm
        dy, dx = y - origin_y, x - origin_x
        cosine = (dx * vx + dy * vy) / ((np.hypot(dx, dy) + 1e-6) * norm)
        mask &= np.degrees(np.arccos(np.clip(cosine, -1, 1))) < 85
    elif feature.CREMode.value == "nasal":
        center = retina.disc.center
        fovea = retina.fovea_location
        vy, vx = fovea.y - center.y, fovea.x - center.x
        dy, dx = y - center.y, x - center.x
        cosine = (dx * vx + dy * vy) / ((np.hypot(dx, dy) + 1e-6) * np.hypot(vx, vy))
        mask &= np.degrees(np.arccos(np.clip(cosine, -1, 1))) > 80
    if feature.hemifield_spec is not None:
        hemi = feature.hemifield_spec.field.name
        if hemi in ("Superior", "Inferior"):
            center, fovea = retina.disc.center, retina.fovea_location
            vy, vx = fovea.y - center.y, fovea.x - center.x
            if vx < 0:
                vy, vx = -vy, -vx
            superior = (y - center.y) * vx - (x - center.x) * vy < 0
            mask &= superior if hemi == "Superior" else ~superior
    return mask


def _visible(retina, points):
    points = np.asarray(points)
    finite = np.isfinite(points).all(axis=1)
    indices = np.floor(np.where(np.isfinite(points), points, -1)).astype(int)
    y, x = indices.T
    h, w = retina.resolution
    inside = finite & (y >= 0) & (y < h) & (x >= 0) & (x < w)
    inside[inside] &= retina.mask[y[inside], x[inside]].astype(bool)
    return inside


def _coverage(feature: CRE, retina):
    """Include off-image pixels in the denominator, including regional variants."""
    center = retina.disc.center
    radius = feature.outer_circle * 2 * retina.disc.circle.r
    y, x = np.mgrid[
        int(np.floor(center.y - radius)):int(np.ceil(center.y + radius)) + 1,
        int(np.floor(center.x - radius)):int(np.ceil(center.x + radius)) + 1,
    ]
    requested = _radial_mask(feature, retina, y, x) & _regional_mask(feature, retina, y, x)
    points = np.column_stack((y[requested], x[requested]))
    return float(np.mean(_visible(retina, points))) if len(points) else 0.0


def _pieces(segment, mask):
    """Clip skeleton sections for display only; never fit splines to these pieces."""
    skeleton = np.asarray(segment.skeleton)
    indices = np.floor(skeleton).astype(int)
    y, x = indices.T
    h, w = mask.shape
    keep = (y >= 0) & (y < h) & (x >= 0) & (x < w)
    keep[keep] &= mask[y[keep], x[keep]]
    changes = np.diff(np.r_[False, keep, False].astype(int))
    result = []
    for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        piece = Segment(skeleton[start:stop].copy(), edge=segment.edge)
        piece.layer, piece.index, piece.id = segment.layer, segment.index, segment.id
        piece.original_segments = [segment]
        result.append(piece)
    return result


def _trunks(feature: CRE, layer: VesselTreeLayer):
    key = (feature.inner_circle, feature.outer_circle, feature.spline_error_fraction)
    cache = getattr(layer, "_aric_trunk_cache", None)
    if cache is None:
        cache = layer._aric_trunk_cache = {}
    if key in cache:
        return cache[key]
    retina = layer.retina
    yy, xx = retina.yy_xx
    annulus = _radial_mask(feature, retina, yy, xx) & retina.mask.astype(bool)
    inner = feature.inner_circle * 2 * retina.disc.circle.r
    outer = feature.outer_circle * 2 * retina.disc.circle.r
    center = np.asarray(retina.disc.center.tuple)
    graph = layer.digraph
    handled = set()
    candidates, flags = [], []

    def visit(segment, replacement_for=None):
        edge = segment.edge
        if edge in handled:
            return
        handled.add(edge)
        children = [graph.edges[e]["segment"] for e in graph.out_edges(edge[1])]
        end_radius = np.linalg.norm(np.asarray(graph.nodes[edge[1]]["o"]) - center)
        # Skeleton nodes occupy pixels, so a one-pixel boundary tolerance avoids
        # counting a tiny artificial parent at the inner gridline.
        if len(children) > 1 and end_radius <= inner + 1:
            for child in children:
                visit(child, replacement_for)
            return
        pieces = _pieces(segment, annulus)
        for piece in pieces:
            # The displayed Zone B section stops before the branch endpoint.
            if len(children) > 1 and np.array_equal(piece.skeleton[-1], segment.skeleton[-1]):
                piece.skeleton = piece.skeleton[:-1]
        pieces = [p for p in pieces if len(p.skeleton)]
        # Delegate caliber estimation, including safeguards and fallback, to the
        # original Segment. Zone B clipping only determines trunk eligibility.
        diameter = segment.get_median_diameter(feature.spline_error_fraction) if pieces else np.nan
        if np.isfinite(diameter) and diameter > 0:
            candidates.append(ARICTrunk(segment, pieces, float(diameter), replacement_for))
            # All downstream pieces belong to this measured trunk's branches.
            handled.update(graph.out_edges(edge[1]))
            for node in nx.descendants(graph, edge[1]):
                handled.update(graph.out_edges(node))
        else:
            flags.append(segment)
            if children and inner <= end_radius < outer:
                for child in children:
                    visit(child, edge if replacement_for is None else replacement_for)

    # Work proximally to distally so re-entering daughters cannot become new trunks.
    for node in nx.topological_sort(graph):
        for edge in graph.out_edges(node):
            if edge in handled:
                continue
            segment = graph.edges[edge]["segment"]
            start_radius = np.linalg.norm(np.asarray(graph.nodes[edge[0]]["o"]) - center)
            radii = np.linalg.norm(np.asarray(segment.skeleton) - center, axis=1)
            if start_radius <= inner + 1 and np.any(radii >= inner):
                visit(segment)
    cache[key] = candidates, flags
    return candidates, flags


def measure_aric(feature: CRE, layer: VesselTreeLayer) -> ARICResult:
    result = ARICResult()
    retina = layer.retina
    if retina.disc is None:
        result.flags.append("missing_optic_disc")
        return result
    needs_fovea = feature.CREMode.value != "full" or (feature.hemifield_spec is not None
        and feature.hemifield_spec.field.name in ("Superior", "Inferior"))
    if needs_fovea and (retina.fovea_location is None or
                       retina.disc.center.distance_to(retina.fovea_location) == 0):
        result.flags.append("missing_disc_fovea_axis")
        return result
    result.coverage = _coverage(feature, retina)
    trunks, flags = _trunks(feature, layer)
    for segment in flags:
        points = np.asarray(segment.skeleton)
        affected = _radial_mask(feature, retina, points[:, 0], points[:, 1])
        affected &= _regional_mask(feature, retina, points[:, 0], points[:, 1])
        if np.any(affected):
            result.flags.append(f"ungradable_trunk:{segment.index}")
    yy, xx = retina.yy_xx
    region = _radial_mask(feature, retina, yy, xx) & _regional_mask(feature, retina, yy, xx)
    region &= retina.mask.astype(bool)
    for trunk in trunks:
        # Apply regional restrictions after anatomical selection. A daughter does
        # not replace a usable trunk just because that trunk is outside this field.
        pieces = [p for original in trunk.pieces for p in _pieces(original, region)]
        if not pieces:
            continue
        if feature.CREMode.value == "temporal" and not feature._is_temporal_segment(trunk.segment):
            continue
        if feature.CREMode.value == "nasal" and not (
            trunk.segment.fod_angle() > 80 and trunk.segment.orientation() > 90
        ):
            continue
        result.candidates.append(ARICTrunk(trunk.segment, pieces, trunk.median_diameter, trunk.replacement_for))
    result.candidates.sort(key=lambda m: (-m.median_diameter, m.segment.index if m.segment.index is not None else -1))
    result.selected = result.candidates[:feature.max_vessels]
    if len(result.selected) < feature.max_vessels:
        result.flags.append(f"fewer_vessels:{len(result.selected)}/{feature.max_vessels}")
    if any(m.replacement_for is not None for m in result.selected):
        result.flags.append("daughter_substitution")
    if any(sum(len(p.skeleton) for p in m.pieces) < 3 for m in result.selected):
        result.flags.append("short_zone_b_trunk")
    if result.coverage < 1:
        result.flags.append(f"partial_zone_b:{result.coverage:.1%}")
    result.valid = (len(result.selected) >= feature.required_aric_vessels
                    and result.coverage >= feature.min_area_within_bounds)
    return result


def plot_aric(feature: CRE, ax, layer: VesselTreeLayer, **kwargs):
    result = feature.get_aric_result(layer)
    retina = layer.retina
    retina.plot(ax=ax, image=True, bounds=True, av=False)
    if retina.disc is not None and "missing_disc_fovea_axis" not in result.flags:
        yy, xx = retina.yy_xx
        region = _radial_mask(feature, retina, yy, xx) & _regional_mask(feature, retina, yy, xx)
        overlay = np.zeros((*retina.resolution, 4))
        overlay[region] = (0.2, 1, 0.3, 0.15)
        ax.imshow(overlay)
        ax.contour(region, levels=[0.5], colors="white", linewidths=0.6)
    for trunk in result.candidates:
        if any(trunk is selected for selected in result.selected):
            continue
        for piece in trunk.pieces:
            y, x = (np.asarray(piece.skeleton) + 0.5).T
            ax.plot(x, y, color="0.65", lw=0.7, alpha=0.7, zorder=2)
    for i, trunk in enumerate(result.selected):
        color = plt.get_cmap("tab10")(i % 10)
        style = "--" if trunk.replacement_for is not None else "-"
        for piece in trunk.pieces:
            y, x = (np.asarray(piece.skeleton) + 0.5).T
            ax.plot(x, y, color=color, lw=1.5, linestyle=style, zorder=3,
                    gid="aric-segment-section")
        points = np.concatenate([p.skeleton for p in trunk.pieces]) + 0.5
        middle = points[len(points) // 2]
        width = retina.scale_length_measurement(trunk.median_diameter)
        units = "mm" if retina.mm_per_pixel is not None else "px"
        right_edge = middle[1] > 0.78 * retina.resolution[1]
        ax.annotate(f"{trunk.segment.index}: median {width:.3g} {units}",
                    (middle[1], middle[0]), xytext=(-5 if right_edge else 5, 8 if i % 2 == 0 else -12),
                    ha="right" if right_edge else "left", textcoords="offset points",
                    color="white", fontsize=7,
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=0.5),
                    bbox=dict(facecolor="black", edgecolor=color, linewidth=0.6, alpha=0.8, pad=1))
    status = "valid" if result.valid else "N/A"
    lines = [f"ARIC: {status}; vessels={len(result.selected)}/{feature.max_vessels}; "
             f"minimum={feature.required_aric_vessels}; coverage={result.coverage:.1%}", *result.flags]
    ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
            va="bottom", ha="left", color="white", fontsize=7,
            bbox=dict(facecolor="black", alpha=0.65, pad=3))
    h, w = retina.resolution
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    return ax
