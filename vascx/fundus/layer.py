from __future__ import annotations

import copy
import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx import DiGraph, Graph, connected_components, get_node_attributes
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize as skimage_skeletonize

from rtnls_enface.disc import OpticDisc
from rtnls_enface.grids.base import GridField
from vascx.fundus.vessel_resolve import RecursiveWeightedAverageResolver
from vascx.shared.base import VesselLayer
from vascx.shared.graph import calc_digraph, correct_digraph, make_graph, make_nodes
from vascx.shared.masks import binarize_and_fill
from vascx.shared.nodes import Bifurcation, Endpoint, Node
from vascx.shared.segment import Segment
from vascx.shared.vessels import Vessels
from vascx.utils.eval import dice_score
from vascx.utils.plotting import plot_digraph, plot_graph, plot_mask

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina


def default_seg_color(seg):
    return seg.index % 20


default_vessels_resolver = RecursiveWeightedAverageResolver("median_diameter")


class RegionOutOfBoundsError(RuntimeError):
    pass


class VesselTreeLayer(VesselLayer):
    """Represents an artery or vein layer with a (probably imperfect) tree structure for the vessel graph."""

    def __init__(
        self,
        name: str,
        mask: np.ndarray,
        retina: Retina = None,
        color: Tuple = (1, 1, 1),
    ):
        self.mask: np.ndarray = mask
        self.retina: Retina = retina
        self.name = name
        self.color = color

    @property
    def disc(self) -> OpticDisc:
        return self.retina.disc

    @cached_property
    def binary(self) -> np.ndarray:
        bin = binarize_and_fill(self.mask)
        # bin = remove_small_objects(bin, 30, connectivity=5)
        return bin

    @cached_property
    def binary_nodisc(self) -> np.ndarray:
        if self.disc is None:
            return None
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

    def filter_segments(self, field: GridField, field_threshold=0.75) -> List[Segment]:
        """Filters the segments of the graph based on some criteria."""
        if field is None:
            return self.segments
        grid = field.grid
        # if not grid.field_visible(field):
        #     raise RegionOutOfBoundsError(
        #         f"Field {field} not completely visible in this retina."
        #     )
        return [
            s
            for s in self.segments
            if grid.fraction_in_field(s.skeleton, field) > field_threshold
        ]

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

        binary = self.binary_nodisc

        segment_to_pixels = {s: [] for s in self.segments}
        for x, y in zip(*np.where(binary)):
            closest_point = (x_closest[x, y], y_closest[x, y])
            if closest_point in skeleton_pixel_to_segment:
                s = skeleton_pixel_to_segment[closest_point]  # get closest segment
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

    def filter_nodes(self, field: GridField = None) -> List[Node]:
        """Filters the nodes of the graph based on some criteria."""
        if field is None:
            return self.nodes
        grid = field.grid
        # if not grid.field_visible(field):
        #     raise RegionOutOfBoundsError(
        #         f"Field {field} not completely visible in this retina."
        #     )
        return [n for n in self.nodes if grid.point_in_field(n.position, field)]

    @cached_property
    def bifurcations(self) -> List[Bifurcation]:
        """
        Bifurcations of a networkx graph
        """
        return [n for n in self.nodes if isinstance(n, Bifurcation)]

    def filter_bifurcations(self, field: GridField = None) -> List[Bifurcation]:
        """Filters the bifurcations of the graph based on some criteria."""
        if field is None:
            return self.bifurcations
        grid = field.grid
        return [
            n
            for n in self.bifurcations
            if grid.point_in_field(n.position, field)
        ]

    # STAGE 4: vessels
    @cached_property
    def vessels(self):
        """Resolved vessels (segments) of the vasculature, after running a vessel resolving algorithm on the trees."""
        G = copy.copy(self.digraph)

        def recursive_set_prop_values(edge):
            u, v = edge
            length, median_diameter = (
                G[u][v]["segment"].length,
                G[u][v]["segment"].median_diameter,
            )

            outgoing_edges = G.out_edges(v)
            if len(outgoing_edges) == 0:
                nx.set_edge_attributes(G, {edge: median_diameter}, "agg_value")
                nx.set_edge_attributes(G, {edge: length}, "agg_length")
                return length, median_diameter
            else:
                tmp = [recursive_set_prop_values(e) for e in outgoing_edges]

                tmp = sorted(tmp, key=lambda x: x[0], reverse=True)  # sort by mean diam
                max_agg_diameter, agg_length = tmp[0]
                agg_value = (
                    max_agg_diameter * agg_length + median_diameter * length
                ) / (agg_length + length)
                nx.set_edge_attributes(G, {edge: agg_value}, "agg_value")
                nx.set_edge_attributes(G, {edge: agg_length + length}, "agg_length")
                return agg_value, agg_length + length

        def merge_edges(edges):
            # Ensure edge_list has consecutive edges
            for i in range(len(edges) - 1):
                if edges[i][1] != edges[i + 1][0]:
                    raise ValueError("The edges are not consecutive!")

            # Identify the start and end nodes of the merged edge
            edge = (edges[0][0], edges[-1][1])

            segments = [G[u][v]["segment"] for u, v in edges]
            skeleton = np.concatenate([s.skeleton for s in segments], axis=0)
            seg = Segment(skeleton, edge=edge)
            seg.layer = self
            pixels = [self.get_segment_pixels(s) for s in segments]
            seg._pixels = [item for sublist in pixels for item in sublist]

            # Remove intermediate nodes and edges
            for u, v in edges:
                G.remove_edge(u, v)

            # Add a new edge with combined properties
            G.add_edge(*edge, segment=seg)

        def recursive_merge(edge):
            u, v = edge
            outgoing_edges = G.out_edges(v)

            if len(outgoing_edges) == 0:
                return [edge]

            # get the edge with the max agg_value
            # print(outgoing_edges)
            outgoing_edges = sorted(
                outgoing_edges,
                key=lambda e: G[e[0]][e[1]]["agg_value"],
                reverse=True,
            )
            # get open and completed edges for the edge with max agg_value
            os = recursive_merge(outgoing_edges[0])

            # the rest are all completed segments
            for e in outgoing_edges[1:]:
                merge_edges(recursive_merge(e))

            return [edge] + os

        root_edges = [list(G.out_edges(n))[0] for n in self.trees]
        for edge in root_edges:
            recursive_set_prop_values(edge)
            open_edge = recursive_merge(edge)
            merge_edges(open_edge)

        # for i, (u, v) in enumerate(G.edges()):
        #     G[u][v]["segment"].index = i

        return G

    @cached_property
    def resolved_segments(self) -> List[Segment]:
        return [self.vessels.edges[e]["segment"] for e in self.vessels.edges()]

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
        image=False,
        mask=False,
        color=None,
        skeleton=False,
        segments=False,
        nodes=False,
        digraph=False,
        vessels=False,
        grid_field: GridField = None
    ):
        ax = self._get_base_axes(ax)
        if color is None:
            color = (1,1,1)

        if image:
            if self.retina.image is not None:
                ax.imshow(self.retina.image)

        if mask:
            self.plot_mask(ax, color=color, grid_field=grid_field)

        if skeleton:
            self.plot_skeleton(ax, color=(1, 1, 1), grid_field=grid_field)

        if segments:
            self.plot_segments(ax, grid_field=grid_field)

        if nodes:
            self.plot_nodes(ax, grid_field=grid_field)

        if digraph:
            self.plot_digraph(ax, grid_field=grid_field)

        if vessels:
            self.plot_vessels(ax, grid_field=grid_field)

        # plot ETDRS region
        if grid_field is not None:
            grid_field.plot(ax)

        return ax

    def _get_base_axes(self, ax):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
            ax.imshow(np.zeros(self.binary.shape))
            ax.set_axis_off()

            if self.retina.image is not None:
                ax.imshow(self.retina.image)
        return ax

    def plot_mask(self, ax=None, grid_field: GridField = None, **kwargs):
        ax = self._get_base_axes(ax)
        base_mask = (
            self.binary_nodisc if self.binary_nodisc is not None else self.binary
        )
        if grid_field is not None:
            mask_to_show = base_mask & grid_field.mask
        else:
            mask_to_show = base_mask
        plot_mask(ax, mask_to_show, **kwargs)

    def plot_skeleton(self, ax=None, grid_field: GridField = None, **kwargs):
        ax = self._get_base_axes(ax)
        skeleton = self.skeleton
        if grid_field is not None:
            skeleton = skeleton & grid_field.mask
        plot_mask(ax, skeleton, **kwargs)

    def plot_graph(self, ax=None, **kwargs):
        ax = self._get_base_axes(ax)
        plot_graph(ax, self.graph, **kwargs)

    def plot_digraph(self, ax=None, grid_field: GridField = None, **kwargs):
        ax = self._get_base_axes(ax)
        if grid_field is None:
            g = self.digraph
        else:
            segs = self.filter_segments(grid_field)
            g = self._edge_subgraph_from_segments(self.digraph, segs)
        plot_digraph(ax, g, **kwargs)

    def plot_vessels(self, ax=None, grid_field: GridField = None, **kwargs):
        ax = self._get_base_axes(ax)
        if grid_field is None:
            segs = self.resolved_segments
        else:
            segs = self._filter_resolved_segments(grid_field)
            # g = self._edge_subgraph_from_segments(self.vessels, segs)
        # plot_digraph(ax, g, **kwargs)
        vessels = Vessels(self, segs)
        return vessels.plot(ax=ax, show_index=False, **kwargs)

    def plot_tree_roots(self, ax=None, **kwargs):
        g = Graph(self.trees)
        return self.plot_graph(ax, g, **kwargs)

    def plot_segments(
        self,
        ax=None,
        segments: List[Segment] = None,
        grid_field: GridField = None,
        **kwargs,
    ):
        if segments is None:
            segments = (
                self.filter_segments(grid_field) if grid_field is not None else self.segments
            )
        ax = self._get_base_axes(ax)
        vessels = Vessels(self, segments)
        return vessels.plot(ax=ax, show_index=False, **kwargs)

    def plot_nodes(self, ax=None, fig=None, grid_field: GridField = None):
        if ax is None:
            fig, ax = self._get_base_axes(None, None)
            self.plot_skeleton(ax, fig)
        nodes = self.filter_nodes(grid_field) if grid_field is not None else self.nodes
        for node in nodes:
            if isinstance(node, Bifurcation):
                ax.scatter(*node.position.tuple_xy, s=5, marker="^", color="g")
            elif isinstance(node, Endpoint):
                ax.scatter(*node.position.tuple_xy, s=5, marker="o", color="r")
            else:
                ax.scatter(*node.position.tuple_xy, s=5, marker="s", color="b")

        return fig, ax

    def _edge_subgraph_from_segments(
        self, base: DiGraph, segments: List[Segment]
    ):
        seg_set = set(segments)
        kept_edges = [
            (u, v)
            for u, v, data in base.edges(data=True)
            if data.get("segment") in seg_set
        ]
        return base.edge_subgraph(kept_edges).copy()

    def _filter_resolved_segments(
        self, field: GridField, field_threshold: float = 0.75
    ) -> List[Segment]:
        if field is None:
            return self.resolved_segments
        grid = field.grid
        # if not grid.field_visible(field):
        #     raise RegionOutOfBoundsError(
        #         f"Field {field} not completely visible in this retina."
        #     )
        return [
            s
            for s in self.resolved_segments
            if grid.fraction_in_field(s.skeleton, field) > field_threshold
        ]

    def get_binary(self, remove_disk=False):
        if remove_disk:
            assert self.retina.disc is not None
            mask = self.binary.copy()
            mask[self.retina.disc.mask != 0] = 0
            return mask
        else:
            return self.binary

    def dice(self, layer: VesselTreeLayer, remove_disk=True):
        if remove_disk and self.retina.disc is None:
            warnings.warn(
                "called with remove_disk but disc not present. Proceeding with remove_disk = False"
            )
            remove_disk = False

        mask1 = self.get_binary(remove_disk)
        mask2 = layer.get_binary(remove_disk)

        return dice_score(mask1.flatten(), mask2.flatten())

    def connected_component_count(self):
        return nx.number_connected_components(self.graph)
