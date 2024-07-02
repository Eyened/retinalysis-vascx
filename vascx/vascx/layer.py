from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sknw
from matplotlib.colors import LinearSegmentedColormap
from networkx import DiGraph, Graph, connected_components, get_node_attributes
from rtnls_enface.base import BinaryImage, Layer, LayerType, Point
from rtnls_enface.disc import OpticDisc
from rtnls_utils.eval import dice_score
from scipy.spatial.distance import euclidean as distance_2p
from skimage import measure
from skimage.morphology import skeletonize as skimage_skeletonize
from sortedcontainers import SortedListWithKey

from vascx.analysis.vessel_resolve import RecursiveWeightedAverageResolver
from vascx.nodes import Bifurcation, Endpoint, Node
from vascx.segment import Segment, merge_segments
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


class VesselLayer(Layer):
    # location of the zones in optic disc multiples from the border of the optic disc
    zone_intervals = {"A": (0.0, 0.5), "B": (0.5, 1.0), "C": (1.0, 2.0)}

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

        self._binary = None
        self._skeleton = None
        self._graph = None
        self._segments = None
        self._trees = None
        self._digraph = None
        self._nodes = None
        self._id_to_segment = None

    def get_binary(self, remove_disk=False):
        if remove_disk:
            assert self.retina.disc is not None
            mask = self.binary.copy()
            mask[self.retina.disc.mask != 0] = 0
            return mask
        else:
            return self.binary

    @property
    def disc(self) -> OpticDisc:
        return self.retina.disc

    @property
    def binary(self) -> np.ndarray:
        if self._binary is None:
            self._binary = self.get_binarized()
        return self._binary

    # STAGE 1 of processing, calc the skeleton
    @property
    def skeleton(self) -> np.ndarray:
        if self._skeleton is None:
            self._skeleton = skimage_skeletonize(self.binary)[:, :]
            if self.disc is not None:
                # mask out the skeletonization using the disc
                self._skeleton = self._skeleton & ~self.disc.mask
        return self._skeleton

    # STAGE 2: graph and undirected segments
    @property
    def graph(self) -> Graph:
        if self._graph is None:
            self.calc_graph()
        return self._graph

    @property
    def segments(self) -> List[Segment]:
        """List of all vessel segments
        Each segment corresponts to an edge in the skeletonization graph
        """
        if self._segments is None:
            self.calc_graph()
        return self._segments

    # STAGE 3: trees / digraph and bifurcations (everything that requires direction)
    @property
    def trees(self) -> List[Segment]:
        """Roots of the trees of the vasculature.
        These are nodes of a networkx graph.
        Each edge has attached Segment instance in edge['segment]
        """
        if self._trees is None:
            self.calc_digraph()
            self.correct_digraph()
        return self._trees

    @property
    def digraph(self) -> DiGraph:
        """Digraph of the vasculature"""
        if self._digraph is None:
            self.calc_digraph()
            self.correct_digraph()
        return self._digraph

    @property
    def nodes(self) -> List[Node]:
        """
        Nodes of a networkx graph
        """
        if self._nodes is None:
            self.calc_nodes()

        return self._nodes

    @property
    def bifurcations(self) -> List[Bifurcation]:
        """
        Bifurcations of a networkx graph
        """
        if self._nodes is None:
            self.calc_nodes()

        return [n for n in self.nodes if isinstance(n, Bifurcation)]

    # STAGE 4: vessels
    @property
    def vessels(self):
        """Resolved vessels (segments) of the vasculature, after running a vessel resolving algorithm on the trees."""
        return self.get_vessels()

    def calc_graph(self):
        segments = []
        graph = sknw.build_sknw(self.skeleton)

        def edge_from(G, edge, node):
            node_pt = G.nodes[node]["o"]
            pts = G[edge[0]][edge[1]]["pts"]
            if tuple(pts[0, :]) != tuple(node_pt):
                G[edge[0]][edge[1]]["pts"] = G[edge[0]][edge[1]]["pts"][::-1]
            if edge[0] == node:
                return edge
            else:
                return (edge[1], edge[0])

        def edge_to(G, edge, node):
            node_pt = G.nodes[node]["o"]
            pts = G[edge[0]][edge[1]]["pts"]
            if tuple(pts[-1, :]) != tuple(node_pt):
                G[edge[0]][edge[1]]["pts"] = G[edge[0]][edge[1]]["pts"][::-1]

            if edge[1] == node:
                return edge
            else:
                return (edge[1], edge[0])

        # graph may contain a few self loops. we remove them.
        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)

        # now we remove nodes of degree 2
        # we substitute the their edges with a single edge
        deg_2_nodes = [n for n in graph.nodes() if graph.degree(n) == 2]
        while len(deg_2_nodes) > 0:
            n = deg_2_nodes.pop()
            edges = list(graph.edges(n))

            in_edge = edge_to(graph, edges[0], n)
            out_edge = edge_from(graph, edges[1], n)

            # print(in_edge, out_edge)
            # print(
            #     "in_edge",
            #     graph[in_edge[0]][in_edge[1]]["pts"][0, :],
            #     "-",
            #     graph[in_edge[0]][in_edge[1]]["pts"][-1, :],
            # )
            # print(
            #     "out_edge",
            #     graph[out_edge[0]][out_edge[1]]["pts"][0, :],
            #     "-",
            #     graph[out_edge[0]][out_edge[1]]["pts"][-1, :],
            # )

            graph.add_edge(
                in_edge[0],
                out_edge[1],
                pts=np.concatenate(
                    [
                        graph.get_edge_data(*in_edge)["pts"][:-1, :],
                        graph.get_edge_data(*out_edge)["pts"],
                    ],
                    axis=0,
                ),
            )
            graph.remove_edge(*in_edge)
            graph.remove_edge(*out_edge)
            graph.remove_node(n)
            deg_2_nodes = [n for n in graph.nodes() if graph.degree(n) == 2]

        for s, e in graph.edges():
            skl = graph[s][e]["pts"]

            seg = Segment(skl, edge=(s, e))
            graph[s][e]["segment"] = seg
            seg.id = frozenset([s, e])
            segments.append(seg)

        for index, s in enumerate(segments):
            s.layer = self
            s.index = index
        self._graph = graph

        self._segments = segments

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

        # # asign the neighbors to the segments to have a segment graph
        # for seg in segments:
        #     edge = list(seg.id)  # contains the 1 or 2 end nodes of this edge
        #     seg.neighbors += [
        #         edge_to_segment[frozenset(edge)]
        #         for edge in self.graph.edges(edge[0])
        #         if frozenset(edge) != seg.id
        #     ]
        #     if len(edge) > 1:
        #         seg.neighbors += [
        #             edge_to_segment[frozenset(edge)]
        #             for edge in self.graph.edges(edge[1])
        #             if frozenset(edge) != seg.id
        #         ]

        # # assign pixels from segmentation to segments
        # skeleton_pixel_to_segment = {
        #     (p[0], p[1]): i for i, s in enumerate(segments) for p in s.skeleton
        # }

        # img = self.skeleton.astype(np.uint8) * 255
        # x_closest, y_closest = distance_transform_edt(
        #     ~img, return_distances=False, return_indices=True
        # )

        # binary = self.binary if self.disc is None else self.binary & ~self.disc.mask
        # for x, y in zip(*np.where(binary)):
        #     closest_point = (x_closest[x, y], y_closest[x, y])
        #     if closest_point in skeleton_pixel_to_segment:
        #         segment_index = skeleton_pixel_to_segment[closest_point]
        #         segments[segment_index].pixels.append((x, y))

        # self._segments = segments

    def calc_nodes(self):
        nodes = []
        for node in self.digraph.nodes():
            incoming, outgoing = (
                list(self.digraph.in_edges(node)),
                list(self.digraph.out_edges(node)),
            )

            if len(incoming) != 1:
                nodes.append(Node(Point(*self.digraph.nodes[node]["o"]), node=node))
            else:
                if len(outgoing) == 0:
                    nodes.append(
                        Endpoint(Point(*self.digraph.nodes[node]["o"]), node=node)
                    )
                elif len(outgoing) == 2 or len(outgoing) == 3:
                    nodes.append(
                        Bifurcation(
                            Point(*self.digraph.nodes[node]["o"]),
                            incoming=self.digraph.edges[incoming[0]]["segment"],
                            outgoing=[
                                self.digraph.edges[e]["segment"] for e in outgoing
                            ],
                            node=node,
                        )
                    )
                else:
                    nodes.append(Node(Point(*self.digraph.nodes[node]["o"]), node=node))
        self._nodes = nodes

    def correct_digraph(self, threshold=10):
        """Here we remove edges / segments less than a threshold in length
        This avoid spurious small edges that will affect eg. bifurcation detection.
        """
        self._nodes = None
        segments = [
            self.digraph.edges[e]["segment"]
            for e in self.digraph.edges()
            if self.digraph.out_degree(e[1]) == 0
        ]
        if len(segments) == 0:
            print("Nothing to be done")
            return

        def get_value(obj: Segment):
            return obj.length

        queue = SortedListWithKey(segments, key=get_value)

        while queue[0].length < threshold:
            seg: Segment = queue.pop(0)
            s, e = seg.edge

            if self.digraph.in_degree(e) == 1:
                # analyze the s node
                in_edges = list(self.digraph.in_edges(s))
                out_edges = list(self.digraph.out_edges(s))

                out_edges = [e for e in out_edges if e != seg.edge]

                # this is an endpoint, may remove
                if len(in_edges) == 1 and len(out_edges) == 1:
                    seg1 = self.digraph.get_edge_data(*in_edges[0])["segment"]
                    seg2 = self.digraph.get_edge_data(*out_edges[0])["segment"]

                    new_segment = merge_segments(seg1, seg2)
                    self.digraph.add_edge(
                        seg1.edge[0], seg2.edge[1], segment=new_segment
                    )

                    # remove old edges
                    self.digraph.remove_edge(*in_edges[0])
                    self.digraph.remove_edge(*out_edges[0])
                    self.digraph.remove_node(s)
                    self.digraph.remove_node(e)

                    # update the queue
                    queue.discard(seg1)
                    queue.discard(seg2)
                    queue.add(new_segment)

        self._segments = [
            self.digraph.edges[e]["segment"] for e in self.digraph.edges()
        ]

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
            zone_name in VesselLayer.zone_intervals
        ), f"Zone {zone_name} not recognized. Only valid values are {str(VesselLayer.zone_intervals.keys())}"
        inner, outer = VesselLayer.zone_intervals[zone_name]
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

    def plot_segments(self, **kwargs):
        self.calc_digraph()
        fig, ax = self._get_base_fig(None, None)
        vessels = Vessels(self, self.segments)
        return vessels.plot(fig=fig, ax=ax, **kwargs)

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

    def dice(self, layer: VesselLayer, remove_disk=True):
        mask1 = self.get_binary(remove_disk)
        mask2 = layer.get_binary(remove_disk)

        return dice_score(mask1.flatten(), mask2.flatten())
