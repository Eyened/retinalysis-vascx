from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from vascx.faz.layer import FazLayer

from .base import FazLayerFeature

if TYPE_CHECKING:
    pass


@dataclass
class VascularDensity(FazLayerFeature):
    def plot(self, layer: FazLayer, fig=None, ax=None, **kwargs):
        layer.plot(mask=True)

    def compute(self, layer: FazLayer):
        mask = layer.binary
        return np.sum(mask) / mask.size
