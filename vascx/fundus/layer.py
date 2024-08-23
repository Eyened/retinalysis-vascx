from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from networkx import DiGraph, Graph, connected_components, get_node_attributes
from rtnls_utils.eval import dice_score
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize as skimage_skeletonize

from rtnls_enface.base import LayerType
from rtnls_enface.disc import OpticDisc
from vascx.fundus.vessel_resolve import RecursiveWeightedAverageResolver
from vascx.shared.base import VesselLayer
from vascx.shared.graph import calc_digraph, correct_digraph, make_graph, make_nodes
from vascx.shared.masks import binarize_and_fill
from vascx.shared.nodes import Bifurcation, Endpoint, Node
from vascx.shared.segment import Segment
from vascx.shared.vessels import Vessels

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina


def default_seg_color(seg):
    return seg.index % 20


default_vessels_resolver = RecursiveWeightedAverageResolver("median_diameter")


class VesselTreeLayer(VesselLayer):
    """Represents an artery or vein layer with a (probably imperfect) tree structure for the vessel graph."""

    def __init__(
        self,
        mask: np.ndarray,
        retina: Retina = None,
        name: Union[str, LayerType] = "vessels",
        color: Tuple = (1, 1, 1),
    ):
        self.mask: np.ndarray = mask
        self.retina: Retina = retina
        if not isinstance(type, LayerType):
            self.type = LayerType[name.upper()]
        else:
            self.type = type
        self.color = color

    @property
    def disc(self) -> OpticDisc:
        return self.retina.disc

    @cached_property
    def binary(self) -> np.ndarray:
        return binarize_and_fill(self.mask)

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

    # STAGE 4: vessels
    @property
    def vessels(self):
        """Resolved vessels (segments) of the vasculature, after running a vessel resolving algorithm on the trees."""
        return self.get_vessels()

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
                (n, self.disc.distance_to_center(node_positions[n])) for n in cc_nodes
            ]
            closest_node = sorted(nodes_distances, key=lambda x: x[1])[0][0]

            trees.append(closest_node)

        return trees

    def make_graph_img(self):
        img = np.zeros((*self.binary.shape, 3), dtype=np.uint8)
        for s, e in self.graph.edges():
            pts = self.graph[s][e]["pts"]
            for p in pts:
                img[p[0], p[1], :] = 255

        for n in self.graph.nodes():
            pts = self.graph.nodes[n]["pts"]
            for p in pts:
                img[p[0], p[1], :] = [255, 0, 0]
        return img

    def get_segment(self, id: int):
        for seg in self.segments:
            if seg.id == id:
                return seg
        return None

    def filter_segments_by_numpoints(self, min_numpoints) -> List[Segment]:
        return [s for s in self.segments if len(s.skeleton) >= min_numpoints]

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
        segments=False,
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

        if segments:
            self.plot_segments(ax, fig)

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

            if self.retina.fundus_image is not None:
                ax.imshow(self.retina.fundus_image)
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

    def _plot_graph(self, g: Graph, ax=None, fig=None):
        fig, ax = self._get_base_fig(fig, ax)
        for s, e in g.edges():
            start = self.graph.nodes[s]["o"]
            end = self.graph.nodes[e]["o"]

            ax.plot([start[1], end[1]], [start[0], end[0]], color="white", markersize=2)

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

    def plot_tree_roots(self):
        g = Graph(self.trees)
        return self._plot_graph(g)

    def plot_digraph(self, ax=None, fig=None):
        if ax is None:
            fig, ax = self._get_base_fig(fig, ax)
        return self._plot_digraph(self.digraph, ax, fig)

    def plot_segments(self, ax=None, fig=None, **kwargs):
        if ax is None:
            fig, ax = self._get_base_fig(None, None)
        # self.calc_digraph()
        # fig, ax = self._get_base_fig(None, None)
        vessels = Vessels(self, self.segments)
        return vessels.plot(fig=fig, ax=ax, **kwargs)

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

    def get_binary(self, remove_disk=False):
        if remove_disk:
            assert self.retina.disc is not None
            mask = self.binary.copy()
            mask[self.retina.disc.mask != 0] = 0
            return mask
        else:
            return self.binary

    def dice(self, layer: VesselTreeLayer, remove_disk=True):
        mask1 = self.get_binary(remove_disk)
        mask2 = layer.get_binary(remove_disk)

        return dice_score(mask1.flatten(), mask2.flatten())
