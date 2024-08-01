from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from networkx import DiGraph, Graph, connected_components, get_node_attributes
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize as skimage_skeletonize
from vascx.shared.base import VesselLayer
from vascx.shared.graph import calc_digraph, correct_digraph, make_graph, make_nodes
from vascx.shared.masks import binarize_and_fill, min_distance_to_edge
from vascx.shared.nodes import Bifurcation, Endpoint, Node
from vascx.shared.segment import Segment

from rtnls_enface.base import Point

if TYPE_CHECKING:
    from vascx.faz.retina import Retina


class FazLayer(VesselLayer):
    """Represents an artery or vein layer around the faz
    which contains many trees starting at the edges and with leaves near the FAZ
    """

    def __init__(
        self,
        mask: np.ndarray,
        retina: Retina = None,
        color: Tuple = (1, 1, 1),
        name: str = None,
    ):
        self.mask = mask
        self.retina = retina
        self.color = color
        self.name = name

    @cached_property
    def binary(self) -> np.ndarray:
        return binarize_and_fill(self.mask)

    # STAGE 1 of processing, calc the skeleton
    @cached_property
    def skeleton(self) -> np.ndarray:
        return skimage_skeletonize(self.binary)[:, :]

    # STAGE 2: graph and undirected segments
    @cached_property
    def graph(self) -> Graph:
        graph = make_graph(self.skeleton)
        segments = []
        for s, e in graph.edges():
            skl = graph[s][e]["pts"]

            seg = Segment(skl, edge=(s, e))
            graph[s][e]["segment"] = seg
            seg.id = frozenset([s, e])
            segments.append(seg)

        for index, s in enumerate(segments):
            s.layer = self
            s.index = index
        return graph

    @cached_property
    def undirected_segments(self) -> List[Segment]:
        """List of all vessel segments
        Each segment corresponts to an edge in the skeletonization graph
        """
        return [self.graph.edges[e]["segment"] for e in self.graph.edges()]

    # STAGE 3: trees / digraph and bifurcations (everything that requires direction)
    @cached_property
    def trees(self) -> List[Segment]:
        """Roots of the trees of the vasculature.
        These are nodes of a networkx graph.
        Each edge has attached Segment instance in edge['segment']
        """
        return self.make_trees()

    @cached_property
    def digraph(self) -> DiGraph:
        """Digraph of the vasculature"""
        digraph = calc_digraph(self.graph, self.trees)
        digraph = correct_digraph(digraph, threshold=3)
        for index, (s, e) in enumerate(digraph.edges()):
            digraph[s][e]["segment"].layer = self
            digraph[s][e]["segment"].index = index
        return digraph

    @cached_property
    def segments(self) -> List[Segment]:
        return [self.digraph.edges[e]["segment"] for e in self.digraph.edges()]

    @cached_property
    def segment_pixels(self) -> Dict[Segment, List[Tuple[int, int]]]:
        # assign pixels from segmentation to segments
        skeleton_pixel_to_segment = {
            (p[0], p[1]): s for s in self.segments for p in s.skeleton
        }

        img = self.skeleton.astype(np.uint8) * 255
        x_closest, y_closest = distance_transform_edt(
            ~img, return_distances=False, return_indices=True
        )

        binary = self.binary

        segment_to_pixels = {s: [] for s in self.segments}
        for x, y in zip(*np.where(binary)):
            closest_point = (x_closest[x, y], y_closest[x, y])
            if closest_point in skeleton_pixel_to_segment:
                s = skeleton_pixel_to_segment[closest_point]
                segment_to_pixels[s].append((x, y))
        return segment_to_pixels

    def get_segment_pixels(self, segment: Segment) -> List[Tuple[int]]:
        return self.segment_pixels[segment]

    @cached_property
    def nodes(self) -> List[Node]:
        """
        Nodes of a networkx graph
        """
        return make_nodes(self.digraph)

    @cached_property
    def bifurcations(self) -> List[Bifurcation]:
        """
        Bifurcations of a networkx graph
        """
        return [n for n in self.nodes if isinstance(n, Bifurcation)]

    def min_distance_to_edge(self, point: Point):
        return min_distance_to_edge(self.mask, point)

    def make_trees(self):
        # _trees is a list of segments, each is the root of a separate tree
        # starting from the optic disc
        trees = []
        node_positions = get_node_attributes(self.graph, "o")
        for cc_nodes in connected_components(self.graph):  # one tree per graph CC
            # find the tree edge closest to the optic disc.
            # look only at end edges (connected to nodes of degree 1)
            cc_nodes = [n for n in cc_nodes if self.graph.degree[n] == 1]

            if len(cc_nodes) == 0:
                continue

            nodes_distances = [
                (n, self.min_distance_to_edge(Point(*node_positions[n])))
                for n in cc_nodes
            ]
            closest_node = sorted(nodes_distances, key=lambda x: x[1])[0][0]

            trees.append(closest_node)
        return trees

    def plot(
        self,
        ax=None,
        fig=None,
        mask=True,
        color=None,
        xlim=None,
        ylim=None,
        skeleton=False,
        skeleton_color=None,
        skeleton_dilate=None,
        nodes=False,
        digraph=False,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
            ax.imshow(np.zeros(self.binary.shape))
            ax.set_axis_off()

            if self.retina.image is not None:
                ax.imshow(self.retina.image)
        if color is None:
            color = self.color

        if mask:
            colors = [(0, 0, 0, 0), color]
            cmap = LinearSegmentedColormap.from_list("binary", colors, N=2)
            im = self.binary
            ax.imshow(im, cmap=cmap)

        if skeleton:
            self.plot_skeleton(
                ax,
                fig,
                color=skeleton_color if skeleton_color is not None else (1, 1, 1, 0.5),
                dilate=skeleton_dilate,
            )

        if nodes:
            self.plot_nodes(ax, fig)

        if digraph:
            self._plot_digraph(self.digraph, ax, fig)

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        return fig, ax

    def _get_base_fig(self, fig, ax):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
            ax.imshow(np.zeros(self.binary.shape))
            ax.set_axis_off()

            if self.retina.image is not None:
                ax.imshow(self.retina.image)
        return fig, ax

    def plot_skeleton(self, ax=None, fig=None, color=(1, 1, 1, 1), dilate=None):
        fig, ax = self._get_base_fig(fig, ax)
        mask = self.skeleton

        if dilate is not None:
            mask = cv2.dilate(mask, np.ones((3, 3)), iterations=dilate)
        colors = [(0, 0, 0, 0), color]
        cmap = LinearSegmentedColormap.from_list("binary", colors, N=2)
        ax.imshow(mask, cmap=cmap, interpolation="nearest")
        return fig, ax

    def plot_nodes(self, ax=None, fig=None):
        if ax is None:
            fig, ax = self._get_base_fig(None, None)
            self.plot_skeleton(ax, fig)
        for node in self.nodes:
            if isinstance(node, Bifurcation):
                ax.scatter(*node.position.tuple_xy, s=5, marker="^", color="g")
            elif isinstance(node, Endpoint):
                ax.scatter(*node.position.tuple_xy, s=5, marker="o", color="r")
            else:
                ax.scatter(*node.position.tuple_xy, s=5, marker="s", color="b")

        return fig, ax

    def _plot_digraph(self, g: Graph, ax=None, fig=None):
        for s, e in g.edges():
            start = self.graph.nodes[s]["o"].astype(np.int32)
            end = self.graph.nodes[e]["o"].astype(np.int32)

            dx = end[1] - start[1]
            dy = end[0] - start[0]
            ax.arrow(
                start[1], start[0], dx, dy, color="white", head_width=5, linewidth=0.4
            )

        return fig, ax
