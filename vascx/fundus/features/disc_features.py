from __future__ import annotations

from typing import TYPE_CHECKING

from .base import RetinaFeature

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina


class DiscFoveaDistance(RetinaFeature):
    def __init__(self):
        pass

    def compute(self, retina: Retina):
        return retina.disc_fovea_distance

    def plot(self):
        pass
