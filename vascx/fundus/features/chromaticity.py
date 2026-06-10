from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np
from rtnls_enface.grids.specifications import BaseGridFieldSpecification
from skimage.color import rgb2lab

from .base import RetinaFeature, get_grid_field_suffix, grid_field_masks_and_fraction

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina


class ChromaticityChannel(Enum):
    L = "l"
    a = "a"
    b = "b"


_CHANNEL_INDEX = {
    ChromaticityChannel.L: 0,
    ChromaticityChannel.a: 1,
    ChromaticityChannel.b: 2,
}


class Chromaticity(RetinaFeature):
    """Median Lab chromaticity of retinal background pixels.

    Uses Retina.background_mask to sample fundus pixels outside vessels and disc.
    The disc mask is binary (no cup); the original RPS pipeline uses disc+cup
    segmentation with flood-fill.

    Args (constructor):
    - channel: Lab channel to return (L, a, or b).
    - grid_field: optional region limiting computation within the background mask.
    """

    def __init__(
        self,
        channel: ChromaticityChannel,
        grid_field: Optional[BaseGridFieldSpecification] = None,
    ):
        super().__init__(grid_field_spec=grid_field)
        self.channel = channel

    def compute(self, retina: "Retina"):
        if retina.image is None:
            return None

        mask = retina.background_mask
        if not np.any(mask):
            return None

        if self.grid_field_spec is not None:
            _, in_bounds_mask, frac = grid_field_masks_and_fraction(
                retina, self.grid_field_spec
            )
            if frac < 0.5:
                return None
            mask = mask & in_bounds_mask
            if not np.any(mask):
                return None

        image = np.asarray(retina.image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        vals = rgb2lab(image[mask])
        return float(np.median(vals[:, _CHANNEL_INDEX[self.channel]]))

    def display_name(self, key: str = None, **kwargs) -> str:
        field = get_grid_field_suffix(self.grid_field_spec)
        label = self.channel.value.upper()
        return f"Chromaticity {label}{field} - IM"

    def feature_name_tokens(self) -> list[str]:
        return ["chrom", self.channel.value]

    def parameter_name_tokens(self) -> list[str]:
        return []

    def _plot(self, ax, retina: "Retina", **kwargs):
        ax.imshow(retina.image)
        ax.imshow(retina.background_mask, alpha=0.14)
        retina.plot(ax=ax, image=False, bounds=True, av=False)

        field = self._get_grid_field(retina)
        if field is not None:
            field.plot(ax)

        return ax
