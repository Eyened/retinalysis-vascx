from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from networkx import Graph
from rtnls_enface.disc import OpticDisc
from rtnls_enface.grids.specifications import BaseGridFieldSpecification
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize as skimage_skeletonize

from vascx.shared.base import JointVesselsLayer
from vascx.shared.graph import make_graph
from vascx.shared.masks import binarize_and_fill
from vascx.shared.segment import Segment
from vascx.utils.plotting import plot_mask

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina


def _true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return half-open index ranges for contiguous True values."""
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts, ends))


class FundusVesselsLayer(JointVesselsLayer):
    """Represents an artery or vein layer with a (probably imperfect) tree structure for the vessel graph."""

    def __init__(
        self,
        name: str,
        mask: np.ndarray,
        retina: Retina = None,
        color: Tuple = (1, 1, 1),
    ):
        self.name = name
        self.mask: np.ndarray = mask
        self.retina: Retina = retina
        self.color = color
        self._region_segments_cache: Dict[
            BaseGridFieldSpecification, List[Segment]
        ] = {}

    @property
    def disc(self) -> OpticDisc:
        return self.retina.disc

    @cached_property
    def binary(self) -> np.ndarray:
        return binarize_and_fill(self.mask)

    @cached_property
    def binary_nodisc(self) -> np.ndarray:
        return self.binary & ~self.disc.mask

    # STAGE 1 of processing, calc the skeleton
    @cached_property
    def skeleton(self) -> np.ndarray:
        skeleton = skimage_skeletonize(self.binary)[:, :]
        if self.disc is not None:
            # mask out the skeletonization using the disc
            skeleton = skeleton & ~self.disc.mask
        return skeleton

    # STAGE 2: graph and undirected segments
    @cached_property
    def graph(self) -> Graph:
        """Skeleton graph with a Segment attached to each edge."""
        graph = make_graph(self.skeleton)
        segments = []
        for s, e in graph.edges():
            skl = graph[s][e]["pts"]
            seg = Segment(skl, edge=(s, e))
            graph[s][e]["segment"] = seg
            seg.id = frozenset([s, e])
            segments.append(seg)

        for index, seg in enumerate(segments):
            seg.layer = self
            seg.index = index
        return graph

    @cached_property
    def undirected_segments(self) -> List[Segment]:
        """List of vessel segments, one per skeleton-graph edge."""
        return [self.graph.edges[e]["segment"] for e in self.graph.edges()]

    def _clip_segments_to_mask(
        self, segments: List[Segment], mask: np.ndarray
    ) -> List[Segment]:
        """Clip segments to contiguous runs inside a binary mask."""
        clipped_segments: List[Segment] = []
        mask = mask.astype(bool, copy=False)

        for segment in segments:
            skeleton = np.asarray(segment.skeleton, dtype=np.int32)
            keep = mask[skeleton[:, 0], skeleton[:, 1]]
            runs = [
                (start, end)
                for start, end in _true_runs(keep)
                if end - start >= 2
            ]

            if len(runs) == 0:
                continue

            if len(runs) == 1 and runs[0] == (0, len(skeleton)):
                clipped_segments.append(segment)
                continue

            for start, end in runs:
                piece = Segment(skeleton=skeleton[start:end].copy(), edge=segment.edge)
                piece.layer = self
                piece.id = segment.id
                piece.index = segment.index
                piece.original_segments = (
                    segment.original_segments
                    if segment.original_segments is not None
                    else [segment]
                )
                clipped_segments.append(piece)

        return clipped_segments

    def get_region_segments(
        self, field_spec: BaseGridFieldSpecification = None
    ) -> List[Segment]:
        """Return undirected segments clipped to a grid field."""
        if field_spec is None:
            return self.undirected_segments

        cached_segments = self._region_segments_cache.get(field_spec)
        if cached_segments is not None:
            return cached_segments

        field = self.retina.get_grid_field(field_spec)
        clipped_segments = self._clip_segments_to_mask(
            self.undirected_segments, field.mask
        )
        self._region_segments_cache[field_spec] = clipped_segments
        return clipped_segments

    @cached_property
    def segment_pixels(self) -> Dict[Segment, List[Tuple[int, int]]]:
        """Assign vessel-mask pixels to the nearest undirected segment skeleton."""
        segments = self.undirected_segments
        skeleton_pixel_to_segment = {
            (int(p[0]), int(p[1])): s for s in segments for p in s.skeleton
        }

        img = self.skeleton.astype(np.uint8) * 255
        x_closest, y_closest = distance_transform_edt(
            ~img, return_distances=False, return_indices=True
        )

        binary = self.binary_nodisc
        segment_to_pixels: Dict[Segment, List[Tuple[int, int]]] = {
            s: [] for s in segments
        }
        for x, y in zip(*np.where(binary)):
            closest_point = (int(x_closest[x, y]), int(y_closest[x, y]))
            segment = skeleton_pixel_to_segment.get(closest_point)
            if segment is not None:
                segment_to_pixels[segment].append((int(x), int(y)))
        return segment_to_pixels

    def get_segment_pixels(self, segment: Segment) -> List[Tuple[int, int]]:
        """Return binary-mask pixels associated with the given segment."""
        if segment in self.segment_pixels:
            return self.segment_pixels[segment]
        # Clipped pieces: aggregate pixels from the original undirected segment(s).
        originals = (
            segment.original_segments
            if segment.original_segments is not None
            else []
        )
        pixels: List[Tuple[int, int]] = []
        for original in originals:
            pixels.extend(self.segment_pixels.get(original, []))
        return pixels

    @cached_property
    def distance_transform(self) -> np.ndarray:
        skeleton = self.skeleton.astype(np.uint8) * 255
        bounds_mask = self.retina.mask.astype(np.uint8) * 255

        dt_skeleton = distance_transform_edt(~skeleton)
        dt_bounds = distance_transform_edt(bounds_mask)

        dt_skeleton[dt_bounds < dt_skeleton] = np.nan

        dt_skeleton[self.retina.disc.mask.astype(bool)] = np.nan
        dt_skeleton[self.binary] = np.nan

        return dt_skeleton

    @cached_property
    def mean_distance_to_vessel(self) -> float:
        return np.nanmean(self.distance_transform)

    @cached_property
    def invisibility_map(self) -> float:
        mask = np.nan_to_num(self.distance_transform, nan=0.0)
        return gaussian_filter(mask, sigma=50)

    def plot(
        self,
        ax=None,
        image=False,
        bounds=False,
        mask=False,
        color=None,
        skeleton=True,
    ):
        ax = self._get_base_axes(ax)
        if color is None:
            color = self.color

        if image:
            if self.retina.image is not None:
                self.retina.plot(ax=ax, image=True, bounds=bounds, av=False)

        if mask:
            self.plot_mask(ax, color=color)

        if skeleton:
            self.plot_skeleton(ax, color=(1, 1, 1))

        return ax

    def _get_base_axes(self, ax):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
            ax.imshow(np.zeros(self.binary.shape))
            ax.set_axis_off()

            if self.retina.image is not None:
                ax.imshow(self.retina.image)
        return ax

    def plot_mask(self, ax=None, **kwargs):
        ax = self._get_base_axes(ax)
        plot_mask(ax, self.binary, **kwargs)

    def plot_skeleton(self, ax=None, **kwargs):
        ax = self._get_base_axes(ax)
        plot_mask(ax, self.skeleton, **kwargs)
