from __future__ import annotations

from typing import TYPE_CHECKING

from .base import RetinaFeature

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina


class DiscFoveaDistance(RetinaFeature):
    """Scalar OD–fovea distance from Retina.

    Representation: Uses Retina optic disc and fovea spatial coordinates from the segmentation 
    model outputs to compute geometric relationships.

    Computation: Calculates the Euclidean distance between the optic disc center of mass and 
    the fovea location. This distance serves as a fundamental scaling factor for many other 
    biomarkers and provides anatomical context for spatial measurements.

    Options: None.
    """
    
    def __init__(self):
        super().__init__()

    def compute(self, retina: Retina):
        return retina.disc_fovea_distance

    def display_name(self, key: str = None, **kwargs) -> str:
        return "Disc-Fovea Distance - IM"

    def feature_name_tokens(self) -> list[str]:
        return ["disc", "fovea", "distance"]

    def _plot(self, ax, retina: Retina, **kwargs):
        retina.plot(ax=ax, image=True, disc=True, fovea=True)
        return ax
