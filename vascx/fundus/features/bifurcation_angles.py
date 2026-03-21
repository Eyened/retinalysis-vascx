from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from matplotlib import patches
from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from vascx.shared.aggregators import mean

from .base import LayerFeature, grid_field_fraction_in_bounds

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


class BifurcationAngles(LayerFeature):
    """Aggregation of angles at bifurcations measured at distance delta along outgoing branches.

    Representation: Uses Bifurcation geometry from digraph with outgoing branch directions computed
    from skeleton points at specified distances from bifurcation nodes.

    Computation: For each bifurcation point, measures the angle between outgoing vessel branches by
    sampling points at distance 'delta' along each branch and computing the angle between the resulting
    direction vectors. Angles larger than an internal threshold (max_angle=160°) are discarded before
    aggregation.

    Args (constructor):
    - delta: pixel distance from the bifurcation where branch directions are sampled.
    - grid_field: optional `GridFieldEnum` to restrict bifurcations to a region.
    - max_angle: angles above this (degrees) are excluded before aggregation.
    - min_bifurcations: if fewer valid angles remain, compute returns None.
    - aggregator: function to aggregate per-bifurcation angles (e.g., mean/median).
    """

    def __init__(
        self,
        delta: int = 20,
        grid_field: Optional[BaseGridFieldSpecification] = None,
        max_angle: int = 135,
        min_bifurcations: int = 3,
        aggregator=mean,
    ):
        """Configure sampling distance, optional grid field and aggregation function."""
        self.delta = delta
        self.max_angle = max_angle
        self.min_bifurcations = min_bifurcations
        super().__init__(grid_field_spec=grid_field)
        self.aggregator = aggregator

    def _get_bifurcation_points(self, layer: VesselTreeLayer):
        field = self._get_grid_field(layer)
        bifurcations = layer.filter_bifurcations(field)
        bifurcations = [
            bif for bif in bifurcations if bif.outgoing_min_length > self.delta
        ]
        return bifurcations

    def compute(self, layer: VesselTreeLayer):
        if self.grid_field_spec is not None:
            frac = grid_field_fraction_in_bounds(layer.retina, self.grid_field_spec)
            if frac < 0.5:
                return None
        bifurcations = self._get_bifurcation_points(layer)
        angles: list[float] = []
        for bif in bifurcations:
            angle = bif.angle(self.delta)
            if angle <= self.max_angle:
                angles.append(angle)
        if len(angles) < self.min_bifurcations:
            return None
        return self.aggregator(angles)

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_aggregator_prefix, get_grid_field_suffix, get_layer_suffix

        agg = get_aggregator_prefix(self.aggregator)
        field = get_grid_field_suffix(self.grid_field_spec)
        layer = get_layer_suffix(layer_name)
        return f"{agg}Bifurcation Angle{field}{layer}"

    def feature_name_tokens(self) -> list[str]:
        return ["bifangle"]

    def parameter_name_tokens(self) -> list[str]:
        tokens: list[str] = []
        if self.delta != 20:
            tokens.extend(["delta", str(self.delta)])
        if self.max_angle != 135:
            tokens.extend(["max_angle", str(self.max_angle)])
        if self.min_bifurcations != 3:
            tokens.extend(["min_bifurcations", str(self.min_bifurcations)])
        return tokens

    def _plot(self, ax, layer: VesselTreeLayer, **kwargs):
        field = self._get_grid_field(layer)
        ax = layer.plot(
            ax=ax,
            segments=True,
            grid_field=field,
        )

        bifurcations = self._get_bifurcation_points(layer)

        for bif in bifurcations:
            line1, line2 = bif.lines(self.delta)
            angle = bif.angle(self.delta)

            if angle > self.max_angle:
                continue

            radius = line1.length

            arc = patches.Arc(
                bif.position.tuple_xy,
                2 * radius,
                2 * radius,
                angle=0,
                theta1=line2.orientation(),
                theta2=line1.orientation(),
                color="white",
                linewidth=0.5,
            )

            line1.plot(ax, color="white", linewidth=0.5)
            line2.plot(ax, color="white", linewidth=0.5)
            ax.add_patch(arc)

            ax.text(
                bif.position.x + 5,
                bif.position.y,
                # f"{line1.orientation():.1f} - {line2.orientation():.1f} [{line1.counterclockwise_angle_to(line2):.1f}]",
                f"{angle:.1f}",
                fontsize=3.6,
                color="white",
            )
        return ax
