from __future__ import annotations

from typing import Callable, List, Union

import numpy as np
from matplotlib import pyplot as plt

from vascx.shared.segment import Segment


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

    def plot_skeleton(self, ax=None, fig=None, plot_endpoints=True):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()
            ax.imshow(np.zeros_like(self.layer.binary))

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

    def plot(
        self,
        text: Union[Callable, str] = None,
        filter_fn=None,
        show_index=True,
        plot_image=True,
        plot_endpoints=False,
        plot_chord=False,
        plot_spline_points=False,
        ax=None,
        cmap="tab20",
        min_numpoints=4,
        min_numpoints_caliber=25,
        mask: np.ndarray = None,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
            ax.set_axis_off()
            ax.imshow(np.zeros_like(self.layer.binary), cmap="binary")

        if plot_image and self.layer.retina is not None:
            self.layer.retina.plot_image(ax=ax)

        segments = self.filter_segments_by_numpoints(min_numpoints)

        if filter_fn is not None:
            segments = [s for s in segments if filter_fn(s)]

    

        color_values = [i % 20 for i in range(len(segments))]
        max_color_value = max(color_values) if color_values else 0

        im = np.full_like(self.layer.binary, np.nan, dtype=np.float32)
        for i, segment in enumerate(segments):
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
        # masked = np.ma.masked_where(im == 0, im)
        # ax.imshow(masked, cmap=cmap, interpolation="nearest")

        for i, segment in enumerate(segments):
            if text is not None:
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

            if plot_endpoints:
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

            if plot_chord:
                ax.plot(
                    [segment.skeleton[0][1], segment.skeleton[-1][1]],
                    [segment.skeleton[0][0], segment.skeleton[-1][0]],
                    marker=None,
                    linewidth=0.2,
                    color="white",
                )

        if plot_spline_points:
            for segment in self.filter_segments_by_numpoints(min_numpoints_caliber):
                for diam in segment.diameter_measurements:
                    edge1, edge2 = diam.edge_0, diam.edge_1
                    ax.plot(
                        [edge1[1], edge2[1]],
                        [edge1[0], edge2[0]],
                        linewidth=0.2,
                        color="black",
                    )

        return ax
