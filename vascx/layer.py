from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sknw
from matplotlib.colors import LinearSegmentedColormap
from networkx import Graph, connected_components, get_node_attributes
from rtnls_enface.disc import OpticDisc
from rtnls_enface.types import BinaryImage, LayerType, Point
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import euclidean as distance_2p
from skimage.morphology import skeletonize as skimage_skeletonize

from vascx.analysis.network import plot_all_nodes
from vascx.analysis.vessel_resolve import RecursiveWeightedAverageResolver
from vascx.segment import Segment
from vascx.utils.plotting import resize_image_by_width
from vascx.vessels import Vessels

if TYPE_CHECKING:
    from vascx.retina import Retina


def seg_length(seg):
    x, y = zip(*seg.pixels)
    distance = 0
    for i in range(0, len(x) - 1):
        distance += distance_2p([x[i], y[i]], [x[i + 1], y[i + 1]])
    return distance


def default_seg_color(seg):
    return seg.index % 20


default_vessels_resolver = RecursiveWeightedAverageResolver("median_diameter")


class Layer:
    # location of the zones in optic disc multiples from the border of the optic disc
    zone_intervals = {"A": (0.0, 0.5), "B": (0.5, 1.0), "C": (1.0, 2.0)}

    def __init__(
        self,
        mask: np.ndarray,
        retina: Retina = None,
        name: Union[str, LayerType] = "vessels",
    ):
        self.mask: np.ndarray = mask
        self.retina: Retina = retina
        if not isinstance(type, LayerType):
            self.type = LayerType[name.upper()]
        else:
            self.type = type

        self._binary = None
        self._skeleton = None
        self._graph = None
        self._nodes = None
        self._segments = None
        self._trees = None
        self._id_to_segment = None
        self._bifurcations = None

    def get_binary(self, remove_disk=False):
        if remove_disk:
            assert self.retina.disc is not None
            mask = self.binary.copy()
            mask[self.retina.disc.mask != 0] = 0
            return mask
        else:
            return self.binary

    @property
    def binary(self) -> np.ndarray:
        if self._binary is None:
            self._binary = self.get_binarized()
        return self._binary

    @property
    def skeleton(self) -> np.ndarray:
        if self._skeleton is None:
            self._skeleton = skimage_skeletonize(self.binary)[:, :]
            if self.disc is not None:
                # mask out the skeletonization using the disc
                self._skeleton = self._skeleton & ~self.disc.mask
        return self._skeleton

    @property
    def graph(self) -> Graph:
        if self._graph is None:
            self._graph = sknw.build_sknw(self.skeleton)
        return self._graph

    @property
    def disc(self) -> OpticDisc:
        return self.retina.disc

    @property
    def segments(self) -> List[Segment]:
        """List of all vessel segments
        Each segment corresponts to an edge in the skeletonization graph
        """
        if self._segments is None:
            self.calc_segments()
        return self._segments

    @property
    def trees(self) -> List[Segment]:
        """Roots of the trees of the vasculature.
        The trees can be traversed using the neighbors property of the segments.
        """
        if self._trees is None:
            self.calc_segments()
        return self._trees

    @property
    def vessels(self):
        """Resolved vessels (segments) of the vasculature, after running a vessel resolving algorithm on the trees."""
        return self.get_vessels()

    @property
    def bifurcations(self):
        if self._bifurcations is None:
            self.calc_bifurcations()

        return self._bifurcations

    def calc_segments(self):
        """From self.graph, creates segment instances
        Each edge in the graph is mapped to a segment
        Each connected component is mapped to a tree (self.trees) defined by its starting segment (the one closest to the optic disc)
        Segments are linked to their neighbors (neighbors property)
        """
        edge_to_segment = {}
        segments = []
        bifurcations = []

        if self.disc is None:
            raise NotImplementedError("disc must be provided for graph analysis")

        # make one segment per graph edge
        for s, e in self.graph.edges():
            skl = self.graph[s][e]["pts"]

            seg = Segment(
                skl,
                start=Point(*self.graph.nodes[s]["o"]),
                end=Point(*self.graph.nodes[e]["o"]),
            )
            seg.id = frozenset([s, e])
            segments.append(seg)
            edge_to_segment[seg.id] = seg

        # _trees is a list of segments, each is the root of a separate tree
        # starting from the optic disc
        self._trees = []
        trees = []
        node_positions = get_node_attributes(self.graph, "o")
        for cc_nodes in connected_components(self.graph):  # one tree per graph CC
            # find the tree edge closest to the optic disc.
            # look only at end edges (connected to nodes of degree 1)
            cc_nodes = [n for n in cc_nodes if self.graph.degree[n] == 1]

            if len(cc_nodes) == 0:
                continue

            nodes_distances = [
                (n, self.disc.distance_to_center(node_positions[n])) for n in cc_nodes
            ]
            closest_node = sorted(nodes_distances, key=lambda x: x[1])[0][0]

            closest_edges = list(self.graph.edges(closest_node))
            if len(closest_edges) == 0:
                continue
            closest_segment = edge_to_segment[frozenset(closest_edges[0])]

            closest_node_pos = Point(*self.graph.nodes[closest_node]["o"])
            assert closest_node_pos in [closest_segment.start, closest_segment.end]

            if closest_segment.start != closest_node_pos:
                closest_segment.reverse()

            # closest_segment.start = closest_node
            trees.append(closest_edges[0])
            self._trees.append(closest_segment)

        # asign the neighbors to the segments to have a segment graph
        for seg in segments:
            edge = list(seg.id)  # contains the 1 or 2 end nodes of this edge
            seg.neighbors += [
                edge_to_segment[frozenset(edge)]
                for edge in self.graph.edges(edge[0])
                if frozenset(edge) != seg.id
            ]
            if len(edge) > 1:
                seg.neighbors += [
                    edge_to_segment[frozenset(edge)]
                    for edge in self.graph.edges(edge[1])
                    if frozenset(edge) != seg.id
                ]

        for index, s in enumerate(segments):
            s.layer = self
            s.index = index

        # assign pixels from segmentation to segments
        skeleton_pixel_to_segment = {
            (p[0], p[1]): i for i, s in enumerate(segments) for p in s.skeleton
        }

        img = self.skeleton.astype(np.uint8) * 255
        x_closest, y_closest = distance_transform_edt(
            ~img, return_distances=False, return_indices=True
        )

        binary = self.binary if self.disc is None else self.binary & ~self.disc.mask
        for x, y in zip(*np.where(binary)):
            closest_point = (x_closest[x, y], y_closest[x, y])
            if closest_point in skeleton_pixel_to_segment:
                segment_index = skeleton_pixel_to_segment[closest_point]
                segments[segment_index].pixels.append((x, y))

        self._segments = segments

    def calc_bifurcations(self):
        raise NotImplementedError()

    def get_segment(self, id: int):
        for seg in self.segments:
            if seg.id == id:
                return seg
        return None

    def get_vessels(self, resolver=default_vessels_resolver):
        vessels = resolver(self.trees)

        vessels = [v for v in vessels if len(v.pixels) > 5]
        return Vessels(self, vessels)

    def filter_segments(self, min_length_factor) -> List[Segment]:
        segments = self.segments
        minimum_length = min_length_factor * np.sum(self.mask.shape).item() / 2
        return [s for s in segments if seg_length(s) >= minimum_length]

    def filter_segments_by_numpoints(self, min_numpoints) -> List[Segment]:
        return [s for s in self.segments if len(s.skeleton) >= min_numpoints]

    def extract_zone(self, zone_name):
        assert (
            zone_name in Layer.zone_intervals
        ), f"Zone {zone_name} not recognized. Only valid values are {str(Layer.zone_intervals.keys())}"
        inner, outer = Layer.zone_intervals[zone_name]
        zone_mask = np.zeros(self.mask.shape, dtype=np.uint8)

        radius = round(self.retina.disc.radius + outer * self.retina.disc.diameter)
        zone_mask = cv2.circle(zone_mask, self.retina.disc.center, radius, 255, -1)

        radius = round(self.retina.disc.radius + inner * self.retina.disc.diameter)
        zone_mask = cv2.circle(zone_mask, self.retina.disc.center, radius, 0, -1)

        # return zone_mask
        return cv2.bitwise_and(self.mask, self.mask, mask=zone_mask)

    def annotation_to_data(self, annot):
        if annot == BinaryImage:
            return self.binary
        if annot == List[Segment]:
            return self.segments

    def get_binarized(self, threshold=0.5):
        # binarize
        bin = np.empty(self.mask.shape, dtype=np.uint8)
        bin[self.mask < threshold] = 0
        bin[self.mask >= threshold] = 1
        return bin

    def plot(
        self, ax=None, fig=None, color=(1, 1, 1), xlim=None, ylim=None, resize=None
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
            ax.imshow(np.zeros(self.binary.shape))
            ax.set_axis_off()

            if self.retina.fundus_image is not None:
                ax.imshow(self.retina.fundus_image)

        colors = [(0, 0, 0, 0), (*color, 1)]
        cmap = LinearSegmentedColormap.from_list("binary", colors, N=2)

        # alpha = np.zeros(self.binary.shape)
        im = self.binary
        if resize is not None:
            im = resize_image_by_width(im, resize)
        ax.imshow(im, cmap=cmap)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        return fig, ax

    def plot_voronoi(self, ax=None, fig=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(40, 40))
            ax.set_axis_off()

        return fig, ax

    def plot_graph(
        self, ax=None, fig=None, show_mask=True, plot_nodes=True, xlim=None, ylim=None
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()

        if show_mask:
            ax.imshow(self.binary, cmap="gray")
        else:
            ax.imshow(np.zeros_like(self.binary), cmap="gray")

        for s, e in self.graph.edges():
            ps = self.graph[s][e]["pts"]
            ax.plot(
                ps[:, 1], ps[:, 0], "green" if show_mask else "white", linewidth=0.5
            )

        if plot_nodes:
            nodes = self.graph.nodes()
            ps = np.array([nodes[i]["o"] for i in nodes])
            ax.plot(ps[:, 1], ps[:, 0], "r.", markersize=0.5)

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        return fig, ax

    def plot_segments(
        self,
        text: Callable | str = None,
        color: Callable | str = default_seg_color,
        filter_fn=None,
        min_length_factor=0,
        show_id=True,
        plot_endpoints=False,
        ax=None,
        fig=None,
        cmap="viridis",
        min_numpoints=4,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()

        segments = self.filter_segments_by_numpoints(min_numpoints)

        if filter_fn is not None:
            segments = [s for s in segments if filter_fn(s)]

        im = np.full_like(self.binary, np.nan)

        def get_value(accesor: Callable | str | None, segment, default=None):
            if isinstance(accesor, str):
                return getattr(segment, accesor)
            elif callable(accesor):
                return accesor(segment)
            else:
                return default

        color_values = [get_value(color, s) for s in segments]
        max_color_value = max([c for c in color_values if c is not None])

        for i, segment in enumerate(segments):
            w, h = im.shape
            for p in segment.pixels:
                if p[0] > w or p[1] > h:
                    continue
                im[p] = (
                    max_color_value - color_values[i] + 1
                    if color_values[i] is not None
                    else 0
                )

        ax.imshow(np.zeros_like(im))
        masked = np.ma.masked_where(im == 0, im)
        ax.imshow(masked, cmap=cmap)

        for i, segment in enumerate(segments):
            text_value = get_value(text, segment)
            if text_value is None:
                seg_text = f"{segment.index}"
            else:
                if show_id:
                    seg_text = f"{segment.index}: {text_value:s}"
                else:
                    seg_text = f"{text_value:s}"
            ax.text(
                segment.mean_xy[1],
                segment.mean_xy[0],
                seg_text,
                fontsize=3,
                color="white",
            )
            if plot_endpoints:
                ax.plot(
                    [segment.skeleton[0][1], segment.skeleton[-1][1]],
                    [segment.skeleton[0][0], segment.skeleton[-1][0]],
                    linewidth=0.1,
                    color="white",
                    markersize=2,
                )

        return fig, ax

    def plot_some_segments(self, ids, ax=None, fig=None):
        from .utils import bounding_box, concat, pad_bounding_box

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
            ax.set_axis_off()

        segments = [self.get_segment(id) for id in ids]
        all_pixels = concat(*[seg.pixels for seg in segments])

        bb = bounding_box(all_pixels)
        bb = pad_bounding_box(*bb, 5)

        skl = self.skeleton.copy().astype(np.uint8)
        for i, segment in enumerate(segments):
            for p in segment.skeleton:
                skl[p] = 2 + i * 2
            for p in segment.connectors:
                skl[p] = 2 + i * 2 + 1
        im = skl[bb[0][0] : bb[1][0], bb[0][1] : bb[1][1]]

        ax.imshow(im, cmap="viridis")
        return fig, ax

    def plot_nodes(self, ax=None, fig=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.zeros(self.binary.shape))
            ax.set_axis_off()

        ax.imshow(self.skeleton, cmap="binary")
        plot_all_nodes(ax, self.nodes)
        return fig, ax

    def dice(self, layer: Layer, remove_disk=True):
        mask1 = self.get_binary(remove_disk)
        mask2 = layer.get_binary(remove_disk)

        return dice_score(mask1.flatten(), mask2.flatten())
