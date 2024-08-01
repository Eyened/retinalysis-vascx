from __future__ import annotations

import networkx as nx
import numpy as np
import sknw
from networkx import DiGraph
from scipy.spatial.distance import euclidean as distance_2p
from sortedcontainers import SortedListWithKey

from rtnls_enface.base import Point
from vascx.shared.nodes import Bifurcation, Endpoint, Node
from vascx.shared.segment import Segment, merge_segments


def make_graph(skeleton: np.array):
    graph = sknw.build_sknw(skeleton)

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
    return graph


def make_nodes(digraph: nx.DiGraph):
    nodes = []
    for node in digraph.nodes():
        incoming, outgoing = (
            list(digraph.in_edges(node)),
            list(digraph.out_edges(node)),
        )

        if len(incoming) != 1:
            nodes.append(Node(Point(*digraph.nodes[node]["o"]), node=node))
        else:
            if len(outgoing) == 0:
                nodes.append(Endpoint(Point(*digraph.nodes[node]["o"]), node=node))
            elif len(outgoing) == 2 or len(outgoing) == 3:
                nodes.append(
                    Bifurcation(
                        Point(*digraph.nodes[node]["o"]),
                        incoming=digraph.edges[incoming[0]]["segment"],
                        outgoing=[digraph.edges[e]["segment"] for e in outgoing],
                        node=node,
                    )
                )
            else:
                nodes.append(Node(Point(*digraph.nodes[node]["o"]), node=node))
    return nodes


def calc_digraph(graph, roots):
    """From self.graph, creates segment instances
    Each edge in the graph is mapped to a segment
    Each connected component is mapped to a tree (self.trees) defined by its starting segment (the one closest to the optic disc)
    Segments are linked to their neighbors (neighbors property)
    """
    # make one segment per graph edge

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
    for node, data in graph.nodes(data=True):
        digraph.add_node(node, **data)
    for node in roots:
        digraph = dfs_create_directed(graph, node, digraph, visited)

    return digraph


def correct_digraph(digraph: nx.DiGraph, threshold=10):
    """Here we remove edges / segments less than a threshold in length
    This avoid spurious small edges that will affect eg. bifurcation detection.
    """
    segments = [
        digraph.edges[e]["segment"]
        for e in digraph.edges()
        if digraph.out_degree(e[1]) == 0
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

        if digraph.in_degree(e) == 1:
            # analyze the s node
            in_edges = list(digraph.in_edges(s))
            out_edges = list(digraph.out_edges(s))

            out_edges = [e for e in out_edges if e != seg.edge]

            # this is an endpoint, may remove
            if len(in_edges) == 1 and len(out_edges) == 1:
                seg1 = digraph.get_edge_data(*in_edges[0])["segment"]
                seg2 = digraph.get_edge_data(*out_edges[0])["segment"]

                new_segment = merge_segments(seg1, seg2)
                digraph.add_edge(seg1.edge[0], seg2.edge[1], segment=new_segment)

                # remove old edges
                digraph.remove_edge(*in_edges[0])
                digraph.remove_edge(*out_edges[0])
                digraph.remove_node(s)
                digraph.remove_node(e)

                # update the queue
                queue.discard(seg1)
                queue.discard(seg2)
                queue.add(new_segment)

    return digraph
