from __future__ import annotations

from typing import TYPE_CHECKING

from .base import LayerFeature, grid_field_fraction_in_bounds

if TYPE_CHECKING:
    from rtnls_enface.grids.base import GridFieldEnum
    from vascx.fundus.layer import VesselTreeLayer


class BifurcationCount(LayerFeature):
    """Count of Bifurcation nodes in layer.nodes, optionally within a GridField.

    Representation: Uses Bifurcation nodes from layer.nodes in the directed graph representation.

    Computation: Counts the number of bifurcation points (vessel branch points) across the retinal 
    vasculature, optionally filtered to those within a specified ETDRS GridField region.

    Options: grid_field (spatial filtering to specific retinal regions).
    """
    
    def __init__(self, grid_field: GridFieldEnum = None):
        """
        Calculation of the number of bifurcation points.

        """
        self.grid_field = grid_field

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
        return f"BifurcationCount(grid_field={fmt(self.grid_field)})"

    def _get_bifurcation_points(self, layer: VesselTreeLayer):
        if self.grid_field is None:
            return layer.filter_bifurcations(None)
        grid = layer.retina.grids[self.grid_field.grid()]
        field = grid.field(self.grid_field)
        return layer.filter_bifurcations(field)

    def compute(self, layer: VesselTreeLayer):
        if self.grid_field is not None:
            frac = grid_field_fraction_in_bounds(layer.retina, self.grid_field)
            if frac < 0.5:
                return None
        bifurcations = self._get_bifurcation_points(layer)
        return len(bifurcations)

    def _plot(self, ax, layer: VesselTreeLayer, **kwargs):
        field = None
        if self.grid_field is not None:
            grid = layer.retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
        ax = layer.plot(
            ax=ax,
            image=True,
            grid_field=field,
        )

        bifurcations = self._get_bifurcation_points(layer)

        # plot ETDRS region
        if self.grid_field is not None:
            grid = layer.retina.grids[self.grid_field.grid()]
            field = grid.field(self.grid_field)
            grid.plot(ax, field)

        for bif in bifurcations:
            ax.scatter(*bif.position.tuple_xy, s=6, color="w", marker="x")
        return ax
