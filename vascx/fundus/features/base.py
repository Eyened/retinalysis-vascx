from __future__ import annotations

import re
from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from rtnls_enface.grids.base import GridField
from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from vascx.shared.features import Feature

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from vascx.fundus.layer import VesselTreeLayer
    from vascx.fundus.retina import Retina
    from vascx.fundus.vessels_layer import FundusVesselsLayer


_MISSING = object()


def normalize_name_token(value: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = text.strip("_").lower()
    return re.sub(r"_+", "_", text)


def format_name_value(value: Any) -> str:
    if isinstance(value, Enum):
        return normalize_name_token(value.value if isinstance(value.value, str) else value.name)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        text = format(float(value), ".12g")
        text = text.replace("-", "neg_").replace(".", "p")
        return normalize_name_token(text)
    if callable(value):
        return normalize_name_token(
            getattr(value, "name", getattr(value, "__name__", value.__class__.__name__))
        )
    return normalize_name_token(str(value))


def get_layer_token(layer_name: str) -> str:
    return normalize_name_token(layer_name)


def get_layer_tokens(layer_name: str) -> List[str]:
    return [get_layer_token(layer_name)]


def get_aggregator_token(aggregator=None) -> Optional[str]:
    if aggregator is None:
        return None
    return format_name_value(aggregator)


def get_aggregator_tokens(aggregator=None) -> List[str]:
    token = get_aggregator_token(aggregator=aggregator)
    return [token] if token else []


def get_grid_spec_token(grid_spec: Any) -> str:
    name = getattr(grid_spec, "name", None)
    if not name:
        raise AttributeError(
            f"{grid_spec.__class__.__name__} must define a non-empty 'name' attribute"
        )
    return normalize_name_token(name)


def get_grid_spec_display_name(grid_spec: Any) -> str:
    display_name = getattr(grid_spec, "display_name", "")
    if display_name:
        return str(display_name)
    return str(getattr(grid_spec, "name", ""))


def _field_name_token(field: Any) -> Optional[str]:
    if field is None:
        return None

    raw = field.name if isinstance(field, Enum) else str(field)
    token = normalize_name_token(raw)
    if token.endswith("_full_grid"):
        return token[: -len("_full_grid")] or "full"
    if token == "full_grid":
        return "full"
    return token


def _nondefault_object_parameter_tokens(obj: Any) -> List[str]:
    if obj is None or not hasattr(obj, "__dict__"):
        return []

    try:
        default_obj = obj.__class__()
        default_values = getattr(default_obj, "__dict__", {})
    except Exception:
        default_values = {}

    tokens: List[str] = []
    for attr, value in vars(obj).items():
        if attr.startswith("_"):
            continue
        default_value = default_values.get(attr, _MISSING)
        if default_value is _MISSING or value != default_value:
            tokens.extend([normalize_name_token(attr), format_name_value(value)])
    return tokens


def get_grid_field_tokens(spec: Optional[BaseGridFieldSpecification]) -> List[str]:
    if spec is None:
        return []

    from rtnls_enface.grids.specifications import GridFieldSpecification

    if isinstance(spec, GridFieldSpecification):
        tokens = [get_grid_spec_token(spec.grid_spec)]
        tokens.extend(_nondefault_object_parameter_tokens(spec.grid_spec))
        field_token = _field_name_token(spec.field)
        if field_token:
            tokens.append(field_token)
        return tokens

    tokens = [get_grid_spec_token(spec)]
    tokens.extend(_nondefault_object_parameter_tokens(spec))
    return tokens


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

    def name_prefix_tokens(self) -> List[str]:
        return get_aggregator_tokens(getattr(self, "aggregator", None))

    def feature_name_tokens(self) -> List[str]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement canonical feature tokens"
        )

    def parameter_name_tokens(self) -> List[str]:
        return []

    def name_tokens(self, layer_name: str = "retina", **kwargs) -> List[str]:
        return [
            *self.name_prefix_tokens(),
            *self.feature_name_tokens(),
            *self.parameter_name_tokens(),
            *get_grid_field_tokens(self.grid_field_spec),
            *get_layer_tokens(layer_name),
        ]


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

    def name_prefix_tokens(self) -> List[str]:
        return get_aggregator_tokens(getattr(self, "aggregator", None))

    def feature_name_tokens(self) -> List[str]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement canonical feature tokens"
        )

    def parameter_name_tokens(self) -> List[str]:
        return []

    def name_tokens(self, layer_name: str, **kwargs) -> List[str]:
        return [
            *self.name_prefix_tokens(),
            *self.feature_name_tokens(),
            *self.parameter_name_tokens(),
            *get_grid_field_tokens(self.grid_field_spec),
            *get_layer_tokens(layer_name),
        ]


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

    def name_prefix_tokens(self) -> List[str]:
        return get_aggregator_tokens(getattr(self, "aggregator", None))

    def feature_name_tokens(self) -> List[str]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement canonical feature tokens"
        )

    def parameter_name_tokens(self) -> List[str]:
        return []

    def name_tokens(self, layer_name: str, **kwargs) -> List[str]:
        return [
            *self.name_prefix_tokens(),
            *self.feature_name_tokens(),
            *self.parameter_name_tokens(),
            *get_grid_field_tokens(self.grid_field_spec),
            *get_layer_tokens(layer_name),
        ]


def get_layer_suffix(layer_name: str) -> str:
    if layer_name == "arteries":
        return " - A"
    elif layer_name == "veins":
        return " - V"
    elif layer_name == "vessels":
        return " - VSL"
    elif layer_name == "retina":
        return " - IM"
    return f" - {layer_name}"


def get_grid_field_suffix(spec: Optional[BaseGridFieldSpecification]) -> str:
    if spec is None:
        return ""

    from rtnls_enface.grids.hemifields import HemifieldField
    from rtnls_enface.grids.specifications import GridFieldSpecification, HemifieldGridSpecification

    if isinstance(spec, GridFieldSpecification):
        if isinstance(spec.grid_spec, HemifieldGridSpecification):
            if spec.field == HemifieldField.Superior:
                return " [Sup]"
            if spec.field == HemifieldField.Inferior:
                return " [Inf]"

        if _field_name_token(spec.field) == "full":
            display_name = get_grid_spec_display_name(spec.grid_spec)
            if display_name:
                return f" [{display_name}]"

    return ""


def get_aggregator_prefix(aggregator=None) -> str:
    if aggregator is None:
        return ""

    if callable(aggregator):
        display_name = getattr(aggregator, "display_name", "")
        if display_name:
            return f"{display_name} "

        name = getattr(aggregator, "name", getattr(aggregator, "__name__", ""))
        if name == "mean":
            return "Mean "
        if name == "median":
            return "Median "
        if name == "sum":
            return "Sum "
        if name == "std":
            return "Std "
        if name == "max":
            return "Max "
        if name == "min":
            return "Min "
    return ""


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
