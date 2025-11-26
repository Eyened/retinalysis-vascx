from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from matplotlib import patches
from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from vascx.shared.aggregators import mean_median

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
    - aggregator: function to aggregate per-bifurcation angles (e.g., mean/median).
    """

    def __init__(
        self,
        delta: int = 20,
        grid_field: Optional[BaseGridFieldSpecification] = None,
        aggregator=mean_median,
    ):
        """Configure sampling distance, optional grid field and aggregation function."""
        self.delta = delta
        self.max_angle = 160
        super().__init__(grid_field_spec=grid_field)
        self.aggregator = aggregator

    def __repr__(self) -> str:
        def fmt(v):
            from enum import Enum

            import numpy as np

            if v is None:
                return "None"
            if isinstance(v, Enum):
                return f"{v.__class__.__name__}.{v.name}"
            if callable(v):
                return getattr(v, "__name__", v.__class__.__name__)
            if isinstance(v, np.generic):
                return repr(v.item())
            return repr(v)

        return (
            f"BifurcationAngles(delta={fmt(self.delta)}, "
            f"grid_field_spec={fmt(self.grid_field_spec)}, "
            f"aggregator={fmt(self.aggregator)})"
        )

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
        return self.aggregator([bif.angle(self.delta) for bif in bifurcations])

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
