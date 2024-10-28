from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import numpy as np

from .base import RetinaFeature

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina
    from rtnls_enface.grids.base import GridField
    from vascx.fundus.layer import VesselTreeLayer
    from vascx.fundus.vessels_layer import FundusVesselsLayer


class VarianceOfLaplacian(RetinaFeature):
    def __init__(self
    ):
        pass

    def compute(self, retina: Retina):
        return np.nanvar(retina.laplacian)

    def plot(self, retina: Retina, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)

        ax.imshow(retina.laplacian)
        ax.text(5, 45, f'{self.compute(retina):.4f}')

        return ax
    
