# BSD 2-Clause License
#
# Copyright (c) 2023, uw-biomedical-ml
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Adapted from https://github.com/uw-biomedical-ml/retinal-pigmentation-score
# Upstream revision: c3404ecdf9368a4e0621ef51e6bdcc528ec2998e

"""Retinal background Lab measurements used to derive RPS."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from rtnls_enface.grids.specifications import BaseGridFieldSpecification
from skimage.color import rgb2lab

from .base import (
    RetinaFeature,
    get_grid_field_suffix,
    grid_field_masks_and_fraction,
    resolve_min_area_within_bounds,
    validate_min_area_within_bounds,
)
from vascx.shared.features import FeatureOutput

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina


class BackgroundLab(RetinaFeature):
    """Median L*, a*, and b* over VascX's retinal background mask.

    Returns a fixed [L*, a*, b*] vector, exported as three scalar columns.
    Uses unenhanced RGB and ``Retina.background_mask``: excludes artery/vein
    masks, the optic disc, and darkest pixels, dilates the exclusions, and
    clips to the visible retina. Missing RGB, AV layers, disc, or background
    produces a missing value. With no ``grid_field``, this measures the full
    visible background. A grid field restricts the existing background mask.

    Grid visibility is evaluated before background exclusions, using
    ``min_area_within_bounds`` from the feature, field, or grid, in that order,
    falling back to 0.80. Insufficient visibility produces a missing vector.

    The upstream RPS implementation uses skimage grayscale thresholding and
    no explicit field-of-view clipping; VascX retains its existing mask.
    The final RPS is a dataset-level PCA of a* and b*, not L*.
    """

    general_description = (
        "Retinal background colour is measured in CIELAB colour space as an "
        "input to the retinal pigmentation score (RPS)."
    )

    outputs = (
        FeatureOutput("l", "L*", "Median CIELAB lightness over retinal background pixels."),
        FeatureOutput("a", "a*", "Median CIELAB green–red coordinate over retinal background pixels."),
        FeatureOutput("b", "b*", "Median CIELAB blue–yellow coordinate over retinal background pixels."),
    )

    default_min_area_within_bounds = 0.80

    def __init__(
        self,
        grid_field: Optional[BaseGridFieldSpecification] = None,
        *,
        min_area_within_bounds: Optional[float] = None,
        plot: bool = False,
    ):
        self.min_area_within_bounds = validate_min_area_within_bounds(
            min_area_within_bounds
        )
        super().__init__(grid_field_spec=grid_field, plot=plot)

    def compute(self, retina: "Retina") -> np.ndarray:
        if retina.image is None:
            return np.full(3, np.nan)
        image = np.asarray(retina.image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("BackgroundLab requires an RGB image with three channels")
        mask = retina.background_mask
        if image.shape[:2] != mask.shape:
            raise ValueError("RGB image and background mask must have matching shapes")
        if self.grid_field_spec is not None:
            _, in_bounds_mask, fraction = grid_field_masks_and_fraction(
                retina, self.grid_field_spec
            )
            if fraction < resolve_min_area_within_bounds(
                self.grid_field_spec,
                self.min_area_within_bounds,
                self.default_min_area_within_bounds,
            ):
                return np.full(3, np.nan)
            mask = mask & in_bounds_mask
        if not np.any(mask):
            return np.full(3, np.nan)
        # Convert pixels before taking the median (Lab is nonlinear).
        lab = rgb2lab(image[mask])
        return np.median(lab, axis=0)

    def feature_name_tokens(self) -> list[str]:
        return ["background", "lab"]

    def display_name(self, key: str = None, **kwargs) -> str:
        field = get_grid_field_suffix(self.grid_field_spec)
        return f"Background Lab{field} - IM"

    def implementation_description(self, **kwargs) -> str:
        return (
            "Calculated from unenhanced RGB converted to CIELAB, excluding "
            "vessels, optic disc, and dark image borders with a dilated mask."
        )

    def aggregation_description(self, **kwargs) -> str:
        return "Reported as the median across the sampled background pixels."

    def _plot(self, ax, retina: "Retina", **kwargs):
        """Sampled retinal background pixels retain their original RGB colour;
        excluded pixels are dimmed and the configured grid field is overlaid.
        """
        if retina.image is None:
            return ax
        from skimage.util import img_as_float

        image = img_as_float(retina.image)
        mask = retina.background_mask
        field = self._get_grid_field(retina)
        if field is not None:
            mask = mask & field.mask.astype(bool) & retina.mask
        display = np.where(mask[..., None], image, image * 0.15)
        ax.imshow(display)
        if field is not None:
            field.plot(ax)
        return ax
