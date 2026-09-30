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
    reconstructed ellipse center (`center`) or its nearest boundary point (`edge`),
    including for clipped discs.

    Options:
    - mode: `center` or `edge` (default `center`).
    """

    general_description = (
        "Disc–fovea distance describes the anatomical separation between the optic "
        "disc and fovea."
    )

    def implementation_description(self, **kwargs) -> str:
        if self.mode == DiscFoveaDistanceMode.Center:
            return "Calculated from the optic-disc center to the foveal center."
        return "Calculated from the nearest optic-disc margin to the foveal center."

    def __init__(
        self,
        mode: Union[DiscFoveaDistanceMode, Literal["center", "edge"]] = DiscFoveaDistanceMode.Center,
        plot: bool = False,
    ):
        super().__init__(plot=plot)
        self.mode = DiscFoveaDistanceMode(mode)

    def compute(self, retina: Retina):
        """Return disc–fovea distance according to the configured mode."""
        if retina.disc is None or retina.fovea_location is None:
            raise ValueError("Disc or fovea location not set")

        if self.mode == DiscFoveaDistanceMode.Center:
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
        """The fundus image shows the optic disc and fovea landmarks used to measure their
        separation, together with the image bounds.
        """
        retina.plot(ax=ax, image=True, disc=True, fovea=True, bounds=True, av=False)
        return ax
