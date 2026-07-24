from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal, Union

from .base import RetinaFeature

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina


class DiscFoveaDistanceMode(str, Enum):
    Center = "center"
    Edge = "edge"


class DiscFoveaDistance(RetinaFeature):
    """Scalar OD–fovea distance from Retina.

    Representation: Uses Retina optic disc and fovea spatial coordinates from the segmentation
    model outputs to compute geometric relationships.

    Computation: Calculates the Euclidean distance between the fovea and either the optic disc
    center of mass (`center`) or the optic disc edge point closest to the fovea (`edge`). For
    `center`, returns None unless the disc is more than 15 px from visible bounds.

    Options:
    - mode: `center` or `edge` (default `center`).
    """

    def __init__(
        self,
        mode: Union[DiscFoveaDistanceMode, Literal["center", "edge"]] = DiscFoveaDistanceMode.Center,
    ):
        super().__init__()
        self.mode = DiscFoveaDistanceMode(mode)

    def compute(self, retina: Retina):
        """Return disc–fovea distance according to the configured mode."""
        if retina.disc is None or retina.fovea_location is None:
            raise ValueError("Disc or fovea location not set")

        if self.mode == DiscFoveaDistanceMode.Center:
            if retina.disc.distance_to_visible_bounds() <= 15:
                return None
            return retina.disc_fovea_distance

        edge_point = retina.disc.closest_point(retina.fovea_location)
        return edge_point.distance_to(retina.fovea_location)

    def display_name(self, key: str = None, **kwargs) -> str:
        if self.mode == DiscFoveaDistanceMode.Center:
            return "Disc-Fovea Distance (Center) - IM"
        return "Disc-Fovea Distance - IM"

    def feature_name_tokens(self) -> list[str]:
        return ["disc", "fovea", "distance"]

    def parameter_name_tokens(self) -> list[str]:
        if self.mode != DiscFoveaDistanceMode.Edge:
            return [self.mode.value]
        return []

    def _plot(self, ax, retina: Retina, **kwargs):
        retina.plot(ax=ax, image=True, disc=True, fovea=True, bounds=True, av=False)
        return ax
