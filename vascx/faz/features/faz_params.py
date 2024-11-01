from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from matplotlib.colors import LinearSegmentedColormap
from vascx.faz.layer import FAZLayer

from .base import FAZLayerFeature

if TYPE_CHECKING:
    pass


class FazParameterType(Enum):
    PerimeterLength = 1
    Area = 2


@dataclass
class FazParameter(FAZLayerFeature):
    def __init__(self, parameter: FazParameterType):
        self.param = parameter

    def compute(self, layer: FAZLayer):
        if self.param == FazParameterType.PerimeterLength:
            return layer.retina.faz.perimeter_length
        elif self.param == FazParameterType.Area:
            return layer.retina.faz.area
        else:
            raise ValueError(f"Unknown parameter type: {self.param}")

    def plot_perimeter(self, ax, layer: FAZLayer, **kwargs):
        ax = layer.retina.plot(ax=ax, layers=[], faz=False)

        colors = [(0, 0, 0, 0), (0, 1, 0, 1)]
        cmap = LinearSegmentedColormap.from_list("binary", colors, N=2)
        ax.imshow(layer.retina.faz.contour_image, cmap=cmap)

    def plot_area(self, ax, layer: FAZLayer, **kwargs):
        raise NotImplementedError()

    def plot(self, *args, **kwargs):
        if self.param == FazParameterType.PerimeterLength:
            return self.plot_perimeter(*args, **kwargs)
        elif self.param == FazParameterType.Area:
            return self.plot_area(*args, **kwargs)
        else:
            raise ValueError(f"Unknown parameter type: {self.param}")
