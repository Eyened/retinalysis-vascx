from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from rtnls_enface.grids.base import GridField
from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from vascx.shared.features import Feature

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from vascx.fundus.layer import VesselTreeLayer
    from vascx.fundus.retina import Retina
    from vascx.fundus.vessels_layer import FundusVesselsLayer


class RetinaFeature(Feature):
    def __init__(
        self, grid_field_spec: Optional[BaseGridFieldSpecification] = None
    ) -> None:
        self.grid_field_spec = grid_field_spec

    def _get_grid_field(self, retina: "Retina") -> Optional[GridField]:
        if self.grid_field_spec is None:
            return None
        return retina.get_grid_field(self.grid_field_spec)

    @abstractmethod
    def compute(self, retina: "Retina"):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def _plot(self, ax: "Axes", retina: "Retina", **kwargs) -> "Axes":
        """Subclass draws onto ax for the given Retina and returns ax."""

    def plot(self, ax: "Axes", retina: "Retina", **kwargs) -> "Axes":
        return super().plot(ax=ax, layer=retina, **kwargs)


class LayerFeature(Feature):
    def __init__(
        self, grid_field_spec: Optional[BaseGridFieldSpecification] = None
    ) -> None:
        self.grid_field_spec = grid_field_spec

    def _get_grid_field(self, layer: "VesselTreeLayer") -> Optional[GridField]:
        if self.grid_field_spec is None:
            return None
        if layer.retina is None:
            raise ValueError("Cannot resolve grid field without an associated retina")
        return layer.retina.get_grid_field(self.grid_field_spec)

    @abstractmethod
    def compute(self, layer: "VesselTreeLayer"):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def _plot(self, ax: "Axes", layer: "VesselTreeLayer", **kwargs) -> "Axes":
        """Subclass draws onto ax for the given VesselTreeLayer and returns ax."""

    def plot(self, ax: "Axes", layer: "VesselTreeLayer", **kwargs) -> "Axes":
        return super().plot(ax=ax, layer=layer, **kwargs)


class VesselsLayerFeature(Feature):
    def __init__(
        self, grid_field_spec: Optional[BaseGridFieldSpecification] = None
    ) -> None:
        self.grid_field_spec = grid_field_spec

    def _get_grid_field(self, layer: "FundusVesselsLayer") -> Optional[GridField]:
        if self.grid_field_spec is None:
            return None
        if layer.retina is None:
            raise ValueError("Cannot resolve grid field without an associated retina")
        return layer.retina.get_grid_field(self.grid_field_spec)

    @abstractmethod
    def compute(self, layer: "FundusVesselsLayer"):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def _plot(self, ax: "Axes", layer: "FundusVesselsLayer", **kwargs) -> "Axes":
        """Subclass draws onto ax for the given FundusVesselsLayer and returns ax."""

    def plot(self, ax: "Axes", layer: "FundusVesselsLayer", **kwargs) -> "Axes":
        return super().plot(ax=ax, layer=layer, **kwargs)


def grid_field_masks_and_fraction(
    retina: "Retina", grid_field_spec: BaseGridFieldSpecification
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (field_mask, in_bounds_mask, fraction_in_bounds) for a grid field specification.

    fraction_in_bounds = sum(field_mask & retina.mask) / sum(field_mask), 0.0 if empty.
    """
    field = retina.get_grid_field(grid_field_spec)
    field_mask = field.mask.astype(bool)
    if field_mask.size == 0:
        return field_mask, field_mask, 0.0
    in_bounds_mask = field_mask & retina.mask
    total = int(np.count_nonzero(field_mask))
    if total == 0:
        return field_mask, in_bounds_mask, 0.0
    frac = float(np.count_nonzero(in_bounds_mask)) / float(total)
    return field_mask, in_bounds_mask, frac


def grid_field_fraction_in_bounds(
    retina: "Retina", grid_field_spec: BaseGridFieldSpecification
) -> float:
    """Convenience: return only the fraction of the grid field within retina bounds."""
    _, _, frac = grid_field_masks_and_fraction(retina, grid_field_spec)
    return frac
