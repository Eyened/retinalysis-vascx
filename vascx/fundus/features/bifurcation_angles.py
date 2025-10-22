from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib import patches

from rtnls_enface.base import Line, Point
from vascx.shared.aggregators import mean_median

from .base import LayerFeature

if TYPE_CHECKING:
    from rtnls_enface.grids.base import GridFieldEnum
    from vascx.fundus.layer import VesselTreeLayer


@dataclass
class BifurcationAngles(LayerFeature):
    """Aggregation of angles at bifurcations measured at distance delta along outgoing branches.

    Representation: Uses Bifurcation geometry from digraph with outgoing branch directions computed 
    from skeleton points at specified distances from bifurcation nodes.

    Computation: For each bifurcation point, measures the angle between outgoing vessel branches by 
    sampling points at distance 'delta' along each branch and computing the angle between the resulting 
    direction vectors. Filters out angles larger than max_angle and aggregates results.

    Options: delta (measurement distance from bifurcation), max_angle (angle filter), grid_field 
    (spatial filtering), aggregator (statistical aggregation function).
    """
    
    def __init__(self, delta: int = 20, grid_field: GridFieldEnum = None, aggregator=mean_median,):
        """
        Calculation of bifurcation angles.

        Args:
            delta (int): The distance from the bifurcation point to calculate the angle.
            max_angle (str): Angles larger than this value will be ignored.

        """
        self.delta = delta
        self.max_angle = 160
        self.grid_field = grid_field
        self.aggregator = aggregator

    def __repr__(self) -> str:
        def fmt(v):
            import inspect, numpy as np
            from enum import Enum
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
            f"grid_field={fmt(self.grid_field)}, "
            f"aggregator={fmt(self.aggregator)})"
        )

    def _get_bifurcation_points(self, layer: VesselTreeLayer):
        grid = layer.retina.grids[self.grid_field.grid()]
        field = grid.field(self.grid_field)
        if grid.field_within_bounds(field) < 0.75:
            raise RuntimeError(f"Field {self.grid_field} is not within the bounds of the retina.")
        bifurcations = layer.filter_bifurcations(field)
        bifurcations = [bif for bif in bifurcations if bif.outgoing_min_length > self.delta]
        return bifurcations

    def compute(self, layer: VesselTreeLayer):
        bifurcations = self._get_bifurcation_points(layer)
        return self.aggregator([bif.angle(self.delta) for bif in bifurcations])

    def plot(self, ax, layer: VesselTreeLayer, **kwargs):
        field = None
        if self.grid_field is not None:
            grid = layer.retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
        ax = layer.plot(
            ax=ax,
            segments=True,
            grid_field=field,
        )

        bifurcations = self._get_bifurcation_points(layer)
        
        # plot ETDRS region
        if self.grid_field is not None:
            grid = layer.retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
            grid.plot(ax, field)

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
