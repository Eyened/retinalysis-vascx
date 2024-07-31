from __future__ import annotations

import networkx as nx
import numpy as np
import sknw


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
