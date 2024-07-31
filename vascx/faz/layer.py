from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from networkx import DiGraph, Graph, connected_components, get_node_attributes
from scipy.spatial.distance import euclidean as distance_2p
from skimage import measure
from skimage.morphology import skeletonize as skimage_skeletonize
from vascx.shared.graph import make_graph
from vascx.shared.nodes import Bifurcation, Endpoint
from vascx.shared.segment import Segment

from rtnls_enface.base import Layer

if TYPE_CHECKING:
    from vascx.faz.retina import Retina


class FazLayer(Layer):
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
        return self.get_binarized()

    # STAGE 1 of processing, calc the skeleton
    @cached_property
    def skeleton(self) -> np.ndarray:
        return skimage_skeletonize(self.binary)[:, :]

    # STAGE 2: graph and undirected segments
    @cached_property
    def graph(self) -> Graph:
        return make_graph(self.skeleton)

    @cached_property
    def segments(self) -> List[Segment]:
        """List of all vessel segments
        Each segment corresponts to an edge in the skeletonization graph
        """
        segments = []
        for s, e in self.graph.edges():
            skl = self.graph[s][e]["pts"]

            seg = Segment(skl, edge=(s, e))
            self.graph[s][e]["segment"] = seg
            seg.id = frozenset([s, e])
            segments.append(seg)

        for index, s in enumerate(segments):
            s.layer = self
            s.index = index

        return segments

    # STAGE 3: trees / digraph and bifurcations (everything that requires direction)
    @cached_property
    def trees(self) -> List[Segment]:
        """Roots of the trees of the vasculature.
        These are nodes of a networkx graph.
        Each edge has attached Segment instance in edge['segment]
        """
        if self._trees is None:
            self.calc_digraph()
            self.correct_digraph()
        return self._trees

    def get_binarized(self, threshold=0.5, area_threshold=25):
        # binarize
        bin = np.empty(self.mask.shape, dtype=np.uint8)
        bin[self.mask < threshold] = 0
        bin[self.mask >= threshold] = 255

        # fill in small holes in the segmentation mask
        inverted_image = np.invert(bin)
        labeled_image, num_features = measure.label(
            inverted_image, return_num=True, connectivity=1, background=0
        )
        properties = measure.regionprops(labeled_image)
        for prop in properties:
            if prop.area < area_threshold:
                labeled_image[labeled_image == prop.label] = 0

        # Invert the image back to original form where small holes are now filled
        return np.invert(labeled_image > 0)

    def calc_digraph(self):
        """From self.graph, creates segment instances
        Each edge in the graph is mapped to a segment
        Each connected component is mapped to a tree (self.trees) defined by its starting segment (the one closest to the optic disc)
        Segments are linked to their neighbors (neighbors property)
        """
        if self.disc is None:
            raise NotImplementedError("disc must be provided for graph analysis")

        # make one segment per graph edge

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
                (n, self.disc.distance_to_center(node_positions[n])) for n in cc_nodes
            ]
            closest_node = sorted(nodes_distances, key=lambda x: x[1])[0][0]

            trees.append(closest_node)

        def dfs_create_directed(graph, start, directed_graph, visited):
            """Create a directed graph from an undirected graph using depth-first search (DFS) traversal"""
            visited.add(start)
            # Iterate over each neighbor of the start node
            for neighbor in graph[start]:
                if neighbor not in visited:
                    segment = graph[start][neighbor]["segment"]
                    # make sure skeleton is in the right order
                    if distance_2p(
                        segment.start.tuple, graph.nodes[start]["o"]
                    ) > distance_2p(segment.end.tuple, graph.nodes[start]["o"]):
                        segment.reverse()

                    segment.edge = (start, neighbor)
                    directed_graph.add_edge(
                        start, neighbor, segment=segment
                    )  # Add edge in the direction of traversal
                    dfs_create_directed(graph, neighbor, directed_graph, visited)
            return directed_graph

        visited = set()
        digraph = DiGraph()
        for node, data in self.graph.nodes(data=True):
            digraph.add_node(node, **data)
        for node in trees:
            digraph = dfs_create_directed(self.graph, node, digraph, visited)

        self._trees = trees
        self._digraph = digraph

    def plot(
        self,
        ax=None,
        fig=None,
        mask=True,
        color=None,
        xlim=None,
        ylim=None,
        skeleton=True,
        skeleton_color=None,
        skeleton_dilate=None,
        nodes=False,
        digraph=False,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
            ax.imshow(np.zeros(self.binary.shape))
            ax.set_axis_off()

            if self.retina.fundus_image is not None:
                ax.imshow(self.retina.fundus_image)
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
