from __future__ import annotations

from typing import Callable, List, Union

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt

from vascx.segment import Segment


def default_seg_color(seg):
    return seg.index % 20


class Vessels:
    def __init__(self, layer, segments):
        self.layer = layer
        self.segments = segments

    def get_segment_by_index(self, index: int):
        for seg in self.segments:
            if seg.index == index:
                return seg
        return None

    def filter_segments_by_numpoints(self, min_numpoints=4) -> List[Segment]:
        return [s for s in self.segments if len(s.skeleton) >= min_numpoints]

    def plot(
        self,
        text: Union[Callable, str] = None,
        color: Union[Callable, str] = default_seg_color,
        filter_fn=None,
        min_length_factor=0,
        show_index=True,
        plot_skeleton=True,
        plot_endpoints=False,
        plot_chord=False,
        plot_spline_points=False,
        ax=None,
        fig=None,
        cmap="viridis",
        min_numpoints=4,
        min_numpoints_caliber=25,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()
            ax.imshow(np.zeros_like(self.layer.binary))

        segments = self.filter_segments_by_numpoints(min_numpoints)

        if filter_fn is not None:
            segments = [s for s in segments if filter_fn(s)]

        im = np.full_like(self.layer.binary, np.nan)

        def get_value(accesor: Callable | str | None, segment, default=None):
            if isinstance(accesor, str):
                return getattr(segment, accesor)
            elif callable(accesor):
                return accesor(segment)
            else:
                return default

        color_values = [get_value(color, s) for s in segments]
        max_color_value = max([c for c in color_values if c is not None])

        skeleton_overlay = np.zeros_like(im)
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

                if plot_skeleton:
                    for p in segment.skeleton:
                        skeleton_overlay[p[0], p[1]] = 255

        # ax.imshow(np.zeros_like(im))
        cmap = plt.cm.tab20
        colors = cmap.colors
        colors = ((0, 0, 0, 0), *colors)
        cmap = mcolors.ListedColormap(colors)

        # ax.imshow(np.zeros_like(im))
        # masked = np.ma.masked_where(im == 0, im)
        ax.imshow(im, cmap=cmap, interpolation="nearest")

        for i, segment in enumerate(segments):
            if text != False:
                text_value = get_value(text, segment)
                if text_value is None:
                    seg_text = f"{segment.index}"
                else:
                    if show_index:
                        seg_text = f"{segment.index}: {text_value:s}"
                    else:
                        seg_text = f"{text_value:s}"
                ax.text(
                    segment.mean_xy[1],
                    segment.mean_xy[0],
                    seg_text,
                    fontsize=2.4,
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
                for diam in segment.diameters:
                    edge1, edge2 = diam.edge_0, diam.edge_1
                    ax.plot(
                        [edge1[1], edge2[1]],
                        [edge1[0], edge2[0]],
                        linewidth=0.2,
                        color="black",
                    )

        return fig, ax
