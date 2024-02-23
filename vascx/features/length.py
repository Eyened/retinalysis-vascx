from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List, Tuple, Any
from abc import ABC, abstractmethod

from matplotlib import pyplot as plt

from .base import LayerFeature

if TYPE_CHECKING:
    from vascx.retina import Retina


class Length(LayerFeature):
    # Ideas
    # tortuosity for different levels of caliber
    #   what happens when small vessels not visible
    # tortuosity for different generations

    def hist(self):
        for vessel in self.layer.vessels.segments:
            plt.bar()

    def compute(self, parameters):
        pass

    def calc_auxiliary(self, parameters):
        pass

    def plot(self, **kwargs):
        return self.layer.vessels.plot(
            text=lambda x: f"{x.length:.1f} | {x.chord_length:.1f}",
            show_id=True,
            plot_endpoints=True,
            cmap="tab20",
            **kwargs,
        )
