from __future__ import annotations

from typing import Callable, List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from vascx.shared.segment import Segment

def get_value(text: Union[Callable, str], segment: Segment) -> Union[str, float, int]:
    """Helper function to get value from text parameter.
    
    If text is callable, calls it with segment. Otherwise returns text as-is.
    """
    if callable(text):
        return text(segment)
    return text

class Vessels:
    def __init__(self, layer, segments: List[Segment]):
        self.layer = layer
        self.segments = segments

    def get_segment_by_index(self, index: int):
        for seg in self.segments:
            if seg.index == index:
                return seg
        return None

    def filter_segments_by_numpoints(self, min_numpoints=4) -> List[Segment]:
        return [s for s in self.segments if len(s.skeleton) >= min_numpoints]

    def plot_skeleton(self, ax, plot_endpoints=True):
        im = np.full_like(self.layer.binary, 0)
        for i, segment in enumerate(self.segments):
            for p in segment.skeleton:
                im[p[0], p[1]] = 255

        cmap = plt.get_cmap("binary")
        cmap.set_bad((0, 0, 0, 0))
        masked = np.ma.masked_where(im == 0, im)
        ax.imshow(masked, cmap=cmap, interpolation="nearest")
        # ax.imshow(im, cmap=cmap, interpolation="nearest")

        if plot_endpoints:
            for segment in self.segments:
                ax.plot(
                    segment.skeleton[0][1],
                    segment.skeleton[0][0],
                    marker="o",
                    markersize=1,
                    color="green",
                )
            for segment in self.segments:
                ax.plot(
                    segment.skeleton[-1][1],
                    segment.skeleton[-1][0],
                    marker="o",
                    markersize=0.5,
                    color="red",
                )

        return fig, ax

    def plot_gaps(self, ax, segments=None):
        """Plots the skeleton and also plots the gaps between subsequent points in the skeleton."""
        if segments is None:
            segments = self.segments

        for segment in segments:
            skel = segment.skeleton
            if len(skel) == 0:
                continue

            # # Plot skeleton
            # ys = [p[0] for p in skel]
            # xs = [p[1] for p in skel]
            # ax.plot(xs, ys, ".", markersize=0.5, color="cyan")

            # Plot gaps
            for i in range(len(skel) - 1):
                p1 = skel[i]
                p2 = skel[i+1]
                if abs(p1[0] - p2[0]) > 1 or abs(p1[1] - p2[1]) > 1:
                    ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color="green", linewidth=1)

        return ax

    def plot_splines(
        self,
        ax,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1,
    ):
        """Iterates the segments and for each evaluates the spline and plots it using cv2.polylines."""
        import cv2

        h, w = self.layer.binary.shape
        img = np.zeros((h, w, 4), dtype=np.uint8)

        if len(color) == 3:
            color = (*color, 255)

        color = tuple(int(c) for c in color)

        for segment in self.segments:
            if segment.spline is None:
                continue

            n_points = max(2, int(segment.length))
            t = np.linspace(0, 1, n_points)

            # cv2 uses (x, y) coordinates
            x = segment.spline.x_spline(t)
            y = segment.spline.y_spline(t)

            pts = np.stack([x, y], axis=1).astype(np.int32)
            pts = pts.reshape((-1, 1, 2))

            cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)

        ax.imshow(img)
        return ax



    def plot_text(
        self,
        ax,
        segments: List[Segment],
        text: Union[Callable, str],
        show_index: bool = True,
    ):
        for segment in segments:
            text_value = get_value(text, segment)
            if show_index:
                seg_text = f"{segment.index}: {text_value:s}"
            else:
                seg_text = f"{text_value:s}"
            ax.text(
                segment.mean_xy[1],
                segment.mean_xy[0],
                seg_text,
                fontsize=3.6,
                color="white",
            )
        return ax

    def plot_endpoints(self, ax, segments: List[Segment]):
        for segment in segments:
            ax.plot(
                segment.skeleton[0][1],
                segment.skeleton[0][0],
                marker="o",
                markersize=1,
                color="green",
            )
            ax.plot(
                segment.skeleton[-1][1],
                segment.skeleton[-1][0],
                marker="o",
                markersize=1,
                color="red",
            )

            ax.plot(
                segment.start.x,
                segment.start.y,
                marker="o",
                markersize=1,
                color="green",
            )
            ax.plot(
                segment.end.x,
                segment.end.y,
                marker="o",
                markersize=1,
                color="red",
            )
        return ax

    def plot_chords(self, ax, segments: List[Segment]):
        for segment in segments:
            ax.plot(
                [segment.skeleton[0][1], segment.skeleton[-1][1]],
                [segment.skeleton[0][0], segment.skeleton[-1][0]],
                marker=None,
                linewidth=0.2,
                color="white",
            )
        return ax

    def plot(
        self,
        text: Union[Callable, str] = None,
        filter_fn=None,
        show_index=True,
        image=True,
        segments=False,
        endpoints=False,
        chord=False,
        splines=False,
        spline_points=False,
        gaps=False,
        ax=None,
        cmap="tab20",
        min_numpoints=4,
        min_numpoints_caliber=25,
        mask: np.ndarray = None,
    ):
        ax = self.layer._get_base_axes(ax)
        # if ax is None:
        #     fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
        #     ax.set_axis_off()
        #     ax.imshow(np.zeros_like(self.layer.binary), cmap="binary")

        if image and self.layer.retina is not None:
            self.layer.retina.plot_image(ax=ax)

        segments_to_plot = self.filter_segments_by_numpoints(min_numpoints)

        if filter_fn is not None:
            segments_to_plot = [s for s in segments_to_plot if filter_fn(s)]

        if segments:
            color_values = [i % 20 for i in range(len(segments_to_plot))]
            max_color_value = max(color_values) if color_values else 0

            im = np.full_like(self.layer.binary, np.nan, dtype=np.float32)
            for i, segment in enumerate(segments_to_plot):
                w, h = im.shape
                for p in segment.pixels:
                    if p[0] > w or p[1] > h:
                        continue
                    im[p] = (
                        max_color_value - color_values[i] + 1
                        if color_values[i] is not None
                        else 0
                    )

            if mask is not None:
                im[mask] = np.nan

            cmap = plt.get_cmap(cmap)
            cmap.set_bad((0, 0, 0, 0))
            ax.imshow(im, cmap=cmap, interpolation="nearest")

        if text is not None:
            self.plot_text(ax, segments_to_plot, text, show_index)

        if endpoints:
            self.plot_endpoints(ax, segments_to_plot)

        if chord:
            self.plot_chords(ax, segments_to_plot)

        if splines:
            print("plotting splines")
            self.plot_splines(ax=ax, color=(255, 255, 255))

        if spline_points:
            for segment in self.filter_segments_by_numpoints(min_numpoints_caliber):
                for diam in segment.diameter_measurements:
                    edge1, edge2 = diam.edge_0, diam.edge_1
                    ax.plot(
                        [edge1[1], edge2[1]],
                        [edge1[0], edge2[0]],
                        linewidth=0.2,
                        color="black",
                    )

        if gaps:
            self.plot_gaps(ax=ax, segments=segments_to_plot)

        return ax
