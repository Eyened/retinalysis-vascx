from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Tuple

import numpy as np

from vascx.shared.features import Feature

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from vascx.fundus.retina import Retina
    from vascx.fundus.layer import VesselTreeLayer
    from vascx.fundus.vessels_layer import FundusVesselsLayer
    from rtnls_enface.grids.base import GridFieldEnum


class RetinaFeature(Feature):
    @abstractmethod
    def compute(self, retina: 'Retina'):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def _plot(self, ax: 'Axes', retina: 'Retina', **kwargs) -> 'Axes':
        """Subclass draws onto ax for the given Retina and returns ax."""

    def plot(self, ax: 'Axes', retina: 'Retina', **kwargs) -> 'Axes':
        return super().plot(ax=ax, layer=retina, **kwargs)

class LayerFeature(Feature):
    @abstractmethod
    def compute(self, layer: 'VesselTreeLayer'):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def _plot(self, ax: 'Axes', layer: 'VesselTreeLayer', **kwargs) -> 'Axes':
        """Subclass draws onto ax for the given VesselTreeLayer and returns ax."""

    def plot(self, ax: 'Axes', layer: 'VesselTreeLayer', **kwargs) -> 'Axes':
        return super().plot(ax=ax, layer=layer, **kwargs)


class VesselsLayerFeature(Feature):
    @abstractmethod
    def compute(self, layer: 'FundusVesselsLayer'):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def _plot(self, ax: 'Axes', layer: 'FundusVesselsLayer', **kwargs) -> 'Axes':
        """Subclass draws onto ax for the given FundusVesselsLayer and returns ax."""

    def plot(self, ax: 'Axes', layer: 'FundusVesselsLayer', **kwargs) -> 'Axes':
        return super().plot(ax=ax, layer=layer, **kwargs)


def grid_field_masks_and_fraction(retina: 'Retina', grid_field: 'GridFieldEnum') -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (field_mask, in_bounds_mask, fraction_in_bounds) for a grid field.

    fraction_in_bounds = sum(field_mask & retina.mask) / sum(field_mask), 0.0 if empty.
    """
    grid = retina.grids[grid_field.grid()]
    field = grid.field(grid_field)
    field_mask = field.mask.astype(bool)
    if field_mask.size == 0:
        return field_mask, field_mask, 0.0
    in_bounds_mask = field_mask & retina.mask
    total = int(np.count_nonzero(field_mask))
    if total == 0:
        return field_mask, in_bounds_mask, 0.0
    frac = float(np.count_nonzero(in_bounds_mask)) / float(total)
    return field_mask, in_bounds_mask, frac


def grid_field_fraction_in_bounds(retina: 'Retina', grid_field: 'GridFieldEnum') -> float:
    """Convenience: return only the fraction of the grid field within retina bounds."""
    _, _, frac = grid_field_masks_and_fraction(retina, grid_field)
    return frac
