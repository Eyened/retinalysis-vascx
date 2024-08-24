from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize as skimage_skeletonize

from rtnls_enface.base import LayerType
from rtnls_enface.disc import OpticDisc
from vascx.shared.base import JointVesselsLayer
from vascx.shared.masks import binarize_and_fill
from vascx.utils.plotting import plot_mask

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina


class FundusVesselsLayer(JointVesselsLayer):
    """Represents an artery or vein layer with a (probably imperfect) tree structure for the vessel graph."""

    def __init__(
        self,
        mask: np.ndarray,
        retina: Retina = None,
        name: Union[str, LayerType] = "vessels",
        color: Tuple = (1, 1, 1),
    ):
        self.mask: np.ndarray = mask
        self.retina: Retina = retina
        if not isinstance(type, LayerType):
            self.type = LayerType[name.upper()]
        else:
            self.type = type
        self.color = color

    @property
    def disc(self) -> OpticDisc:
        return self.retina.disc

    @cached_property
    def binary(self) -> np.ndarray:
        return binarize_and_fill(self.mask)

    @cached_property
    def binary_nodisc(self) -> np.ndarray:
        return self.binary & ~self.disc.mask

    # STAGE 1 of processing, calc the skeleton
    @cached_property
    def skeleton(self) -> np.ndarray:
        skeleton = skimage_skeletonize(self.binary)[:, :]
        if self.disc is not None:
            # mask out the skeletonization using the disc
            skeleton = skeleton & ~self.disc.mask
        return skeleton

    @cached_property
    def distance_transform(self) -> np.ndarray:
        skeleton = self.skeleton.astype(np.uint8) * 255
        bounds_mask = self.retina.mask.astype(np.uint8) * 255

        dt_skeleton = distance_transform_edt(~skeleton)
        dt_bounds = distance_transform_edt(bounds_mask)

        dt_skeleton[dt_bounds < dt_skeleton] = np.nan
        return dt_skeleton / self.retina.disc_fovea_distance

    @cached_property
    def mean_distance_to_vessel(self) -> float:
        return np.nanmean(self.distance_transform)

    def plot(
        self,
        ax=None,
        mask=True,
        color=None,
        skeleton=True,
    ):
        ax = self._get_base_axes(ax)
        if color is None:
            color = self.color

        if mask:
            self.plot_mask(ax, color=color)

        if skeleton:
            self.plot_skeleton(ax, color=(1, 1, 1))

        return ax

    def _get_base_axes(self, ax):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
            ax.imshow(np.zeros(self.binary.shape))
            ax.set_axis_off()

            if self.retina.image is not None:
                ax.imshow(self.retina.image)
        return ax

    def plot_mask(self, ax=None, **kwargs):
        ax = self._get_base_axes(ax)
        plot_mask(ax, self.binary, **kwargs)

    def plot_skeleton(self, ax=None, **kwargs):
        ax = self._get_base_axes(ax)
        plot_mask(ax, self.skeleton, **kwargs)
