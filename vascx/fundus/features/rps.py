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

from typing import TYPE_CHECKING

import numpy as np
from skimage.color import rgb2lab

from .base import RetinaFeature
from vascx.shared.features import FeatureOutput

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina


class BackgroundLab(RetinaFeature):
    """Median L*, a*, and b* over VascX's retinal background mask.

    Returns a fixed [L*, a*, b*] vector, exported as three scalar columns.
    Uses unenhanced RGB and ``Retina.background_mask``: excludes artery/vein
    masks, the optic disc, and darkest pixels, dilates the exclusions, and
    clips to the visible retina. Missing RGB, AV layers, disc, or background
    produces a missing value. This measures the full visible background,
    independently of the feature set's vascular measurement grid.

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

    def __init__(self, *, plot: bool = False):
        super().__init__(plot=plot)

    def compute(self, retina: "Retina") -> np.ndarray:
        if retina.image is None:
            return np.full(3, np.nan)
        image = np.asarray(retina.image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("BackgroundLab requires an RGB image with three channels")
        mask = retina.background_mask
        if image.shape[:2] != mask.shape:
            raise ValueError("RGB image and background mask must have matching shapes")
        if not np.any(mask):
            return np.full(3, np.nan)
        # Convert pixels before taking the median (Lab is nonlinear).
        lab = rgb2lab(image[mask])
        return np.median(lab, axis=0)

    def feature_name_tokens(self) -> list[str]:
        return ["background", "lab"]

    def parameter_name_tokens(self) -> list[str]:
        return []

    def display_name(self, key: str = None, **kwargs) -> str:
        return "Background Lab - IM"

    def implementation_description(self, **kwargs) -> str:
        return (
            "Calculated from unenhanced RGB converted to CIELAB, excluding "
            "vessels, optic disc, and dark image borders with a dilated mask."
        )

    def aggregation_description(self, **kwargs) -> str:
        return "Reported as the median across the sampled background pixels."

    def _plot(self, ax, retina: "Retina", **kwargs):
        """Unmasked retinal background pixels retain their original RGB colour;
        excluded pixels are dimmed.
        """
        if retina.image is None:
            return ax
        from skimage.util import img_as_float

        image = img_as_float(retina.image)
        display = np.where(retina.background_mask[..., None], image, image * 0.15)
        ax.imshow(display)
        return ax
