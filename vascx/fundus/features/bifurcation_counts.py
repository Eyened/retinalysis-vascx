from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from .base import LayerFeature, grid_field_fraction_in_bounds

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


class BifurcationCount(LayerFeature):
    """Count of Bifurcation nodes in layer.nodes, optionally within a GridField.

    Representation: Uses Bifurcation nodes from layer.nodes in the directed graph representation.

    Computation: Counts the number of bifurcation points (vessel branch points) across the retinal
    vasculature, optionally filtered to those within a specified ETDRS GridField region.

    Args (constructor):
    - grid_field: optional `GridFieldEnum` restricting the count to a predefined retinal region.
    """

    def __init__(self, grid_field: Optional[BaseGridFieldSpecification] = None):
        """
        Calculation of the number of bifurcation points.

        """
        super().__init__(grid_field_spec=grid_field)

    def _get_bifurcation_points(self, layer: VesselTreeLayer):
        field = self._get_grid_field(layer)
        return layer.filter_bifurcations(field)

    def compute(self, layer: VesselTreeLayer):
        if self.grid_field_spec is not None:
            frac = grid_field_fraction_in_bounds(layer.retina, self.grid_field_spec)
            if frac < 0.5:
                return None
        bifurcations = self._get_bifurcation_points(layer)
        return len(bifurcations)

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_grid_field_suffix, get_layer_suffix

        field = get_grid_field_suffix(self.grid_field_spec)
        layer = get_layer_suffix(layer_name)
        return f"Bifurcation Count{field}{layer}"

    def feature_name_tokens(self) -> list[str]:
        return ["bifurcation", "count"]

    def _plot(self, ax, layer: VesselTreeLayer, **kwargs):
        field = self._get_grid_field(layer)
        ax = layer.plot(
            ax=ax,
            image=True,
            bounds=True,
            grid_field=field,
        )

        bifurcations = self._get_bifurcation_points(layer)

        # plot ETDRS region
        if field is not None:
            field.plot(ax)

        for bif in bifurcations:
            ax.scatter(*bif.position.tuple_xy, s=6, color="w", marker="x")
        return ax
