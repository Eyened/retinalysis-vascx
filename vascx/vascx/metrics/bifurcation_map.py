from typing import Any, List

import matplotlib as mpl
import numpy as np
from vascx.layer import VesselLayer
from vascx.metrics.base import LayerMetric
from vascx.nodes import Node


def calculate_OKS(detections, gt_keypoint, s, k=0.1):
    """
    Calculate OKS for a single ground truth keypoint against all detections.

    :param detections: np.array of shape (m, 2), where m is the number of detected keypoints
    :param gt_keypoint: np.array of shape (2,), a single ground truth keypoint (x, y)
    :param s: float, scale of the target (e.g., the area of the bounding box)
    :param k: float, per-keypoint constant controlling the falloff
    :return: np.array, OKS scores for this ground truth keypoint against all detections
    """
    if detections.size == 0:
        return np.array([])

    d_squared = np.sum((detections - gt_keypoint) ** 2, axis=1)
    oks = np.exp(-d_squared / (2 * s**2 * k**2))
    return oks


def oks_to_distance(oks, s, k=0.1):
    return np.sqrt(-2 * (s**2) * (k**2) * np.log(oks))


def match_keypoints(detections: List[Node], gt_keypoints: List[Node], s, oks_threshold):
    """
    Evaluate detected keypoints against ground truth keypoints using an OKS threshold.

    :param detections: np.array of shape (m, 2), detected keypoints
    :param gt_keypoints: np.array of shape (n, 2), ground truth keypoints
    :param s: float, scale of the target
    :param k: float, per-keypoint constant controlling the falloff
    :param oks_threshold: float, threshold to determine true positives
    :return: dict, counts of TP, FP, FN
    """
    tp = 0
    matched_detections = set()
    matches = []

    detections_pos = np.array(
        [node.position.numpy for node in detections],
        dtype=float,
    )

    for gt_keypoint in gt_keypoints:
        oks_scores = calculate_OKS(detections_pos, gt_keypoint.position.numpy, s)
        if len(oks_scores) > 0 and max(oks_scores) >= oks_threshold:
            tp += 1
            # multiple ground truth keypoints can match the same detection
            det_idx = np.argmax(oks_scores)
            matched_detections.add(det_idx)
            matches.append((detections[det_idx], gt_keypoint))
            # matches.append((Point(*detections[det_idx]), Point(*gt_keypoint)))

    fp = len(detections) - len(matched_detections)
    fn = len(gt_keypoints) - tp

    return matches, {"TP": tp, "FP": fp, "FN": fn}


class BifurcationAP(LayerMetric):
    def __init__(self):
        pass

    def evaluate_node_type(
        self,
        node_type,
        gt: VesselLayer,
        layer: VesselLayer,
        oks_threshold=0.75,
    ):
        gt_keypoints = [node for node in gt.nodes if isinstance(node, node_type)]
        detections = [node for node in layer.nodes if isinstance(node, node_type)]
        fod_distance = gt.retina.fovea_location.distance_to(
            gt.retina.disc.center_of_mass
        )

        return match_keypoints(
            detections, gt_keypoints, s=fod_distance, oks_threshold=oks_threshold
        )

    def precision_recall(
        self,
        node_type,
        gt: VesselLayer,
        layer: VesselLayer,
        oks_threshold=0.75,
    ):
        _, metrics = self.evaluate_node_type(node_type, gt, layer, oks_threshold)
        precision = metrics["TP"] / (metrics["TP"] + metrics["FP"])
        recall = metrics["TP"] / (metrics["TP"] + metrics["FN"])
        return precision, recall

    def edge_recall(self, gt: VesselLayer, layer: VesselLayer, oks_threshold=0.75):
        fod_distance = gt.retina.fovea_location.distance_to(
            gt.retina.disc.center_of_mass
        )

        matches, metrics = match_keypoints(
            layer.bifurcations,
            gt.bifurcations,
            s=fod_distance,
            oks_threshold=oks_threshold,
        )
        gt_node_to_detection_node = {gt.node: detection for detection, gt in matches}

        gt_bifurcation_nodes = set([n.node for n in gt.bifurcations])
        bifurcation_edges = [
            (u, v)
            for u, v in gt.digraph.edges()
            if u in gt_bifurcation_nodes and v in gt_bifurcation_nodes
        ]

        # print("len(bifurcation_edges)", len(bifurcation_edges))

        N = len(bifurcation_edges)
        tp = 0
        for u, v in bifurcation_edges:
            if u in gt_node_to_detection_node and v in gt_node_to_detection_node:
                det_u = gt_node_to_detection_node[u].node
                det_v = gt_node_to_detection_node[v].node
                if layer.digraph.has_edge(det_u, det_v) or layer.digraph.has_edge(
                    det_v, det_u
                ):
                    tp += 1
        return tp / N

    def plot_matches_node_type(
        self,
        node_type,
        gt: VesselLayer,
        layer: VesselLayer,
        oks_threshold=0.75,
    ):
        gt_keypoints = [node for node in gt.nodes if isinstance(node, node_type)]
        detections = [node for node in layer.nodes if isinstance(node, node_type)]
        fod_distance = gt.retina.fovea_location.distance_to(
            gt.retina.disc.center_of_mass
        )

        matches, metrics = match_keypoints(
            detections, gt_keypoints, s=fod_distance, oks_threshold=oks_threshold
        )

        fig, ax = gt._get_base_fig(None, None)
        fig, ax = gt.plot_skeleton(ax=ax, fig=fig)

        threshold_distance = oks_to_distance(oks_threshold, s=fod_distance)
        for node in gt_keypoints:
            # ax.scatter(node[1], node[0], c="w", s=3, marker="x")
            ax.add_patch(
                mpl.patches.Circle(
                    node.position.tuple_xy,
                    threshold_distance,
                    edgecolor=(0, 1, 0, 1.0),
                    facecolor=(0, 1, 0, 0.5),
                    fill=True,
                    lw=0.5,
                )
            )

        for match in matches:
            pt_gt = match[1].position
            pt_lbl = match[0].position
            ax.plot(*pt_lbl.tuple_xy, marker="+", color="red", markersize=5)
            # ax.plot([pt_gt.x, pt_lbl.x], [pt_gt.y, pt_lbl.y], c="g", lw=1, alpha=0.5)

    def __call__(self, layer1: VesselLayer, layer2: VesselLayer) -> Any:
        pass
