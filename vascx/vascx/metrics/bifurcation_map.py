from typing import Any

import numpy as np
from rtnls_enface.base import Point
from vascx.layer import VesselLayer
from vascx.metrics.base import LayerMetric


def calculate_OKS(detections, gt_keypoint, s, k):
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


def evaluate_keypoints(detections, gt_keypoints, s, k, oks_threshold):
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
    pairs = []

    for gt_keypoint in gt_keypoints:
        oks_scores = calculate_OKS(detections, gt_keypoint, s, k)
        if len(oks_scores) > 0 and max(oks_scores) >= oks_threshold:
            tp += 1
            # multiple ground truth keypoints can match the same detection
            det_idx = np.argmax(oks_scores)
            matched_detections.add(det_idx)
            pairs.append((Point(*detections[det_idx]), Point(*gt_keypoint)))

    fp = len(detections) - len(matched_detections)
    fn = len(gt_keypoints) - tp

    return {"TP": tp, "FP": fp, "FN": fn}, pairs


class BifurcationMAP(LayerMetric):
    def __init__(self):
        pass

    def evaluate_node_type(
        self,
        node_type,
        gt: VesselLayer,
        layer: VesselLayer,
        s=20,
        k=0.5,
        oks_threshold=0.75,
    ):
        gt_keypoints = np.array(
            [node.position.tuple for node in gt.nodes if isinstance(node, node_type)],
            dtype=float,
        )
        detections = np.array(
            [
                node.position.tuple
                for node in layer.nodes
                if isinstance(node, node_type)
            ],
            dtype=float,
        )

        return evaluate_keypoints(detections, gt_keypoints, s, k, oks_threshold)

    def plot_matches_node_type(
        self,
        node_type,
        gt: VesselLayer,
        layer: VesselLayer,
        s=20,
        k=0.5,
        oks_threshold=0.75,
    ):
        gt_keypoints = np.array(
            [node.position.tuple for node in gt.nodes if isinstance(node, node_type)],
            dtype=float,
        )
        detections = np.array(
            [
                node.position.tuple
                for node in layer.nodes
                if isinstance(node, node_type)
            ],
            dtype=float,
        )

        metrics, pairs = evaluate_keypoints(
            detections, gt_keypoints, s, k, oks_threshold
        )

        print(metrics)

        fig, ax = gt._get_base_fig(None, None)
        for node in gt_keypoints:
            ax.scatter(node[1], node[0], c="w", s=3, marker="x")

        for pair in pairs:
            ax.plot(
                [pair[0].x, pair[1].x], [pair[0].y, pair[1].y], c="g", lw=1, alpha=0.5
            )

    def __call__(self, layer1: VesselLayer, layer2: VesselLayer) -> Any:
        pass
