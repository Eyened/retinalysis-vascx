from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import getdoc, signature
from textwrap import fill
from typing import TYPE_CHECKING, Any, Iterable, List, Sequence, Tuple

from .naming import (
    FeatureName,
    NamePart,
    NamingConvention,
    coerce_naming_convention,
    make_feature_names,
)

from .outputs import FeatureOutput, output_values, validate_schema
from .units import effective_schema, scale_values, spatial_power

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


_MISSING = object()


def _add_plot_caption(ax, caption: str) -> None:
    """Render the plotting implementation's docstring below its image panel."""
    if not caption:
        return
    # A stable artist ID also allows callers to find or replace the caption.
    for artist in list(ax.texts):
        if artist.get_gid() == "biomarker-caption":
            artist.remove()
    width_inches = ax.get_position().width * ax.figure.get_figwidth()
    wrapped = fill(" ".join(caption.split()), width=max(28, int(width_inches * 14)))
    ax.text(0.0, -0.035, wrapped, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="black", clip_on=False, gid="biomarker-caption")


class FeatureSet:
    _registry = {}

    def __init__(
        self,
        name: str,
        features: Iterable["Feature"],
        *,
        description: str = "",
    ):
        self.name = name
        self.features = list(features)
        self.description = description
        self.__class__._register_instance(self)

    def __iter__(self):
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    @classmethod
    def _register_instance(cls, instance: FeatureSet):
        if instance.name in cls._registry:
            raise ValueError(
                f"Attempt to register more than one FeatureSet instance with name '{instance.name}'. FeatureSet names must be unique."
            )
        cls._registry[instance.name] = instance

    @classmethod
    def get_by_name(cls, name):
        return cls._registry.get(name, None)


class Feature(ABC):
    outputs = (FeatureOutput(units="dimensionless"),)
    general_description: str = ""
    plot_in_report: bool = False

    def __init__(self, *, plot: bool = False) -> None:
        self.plot_in_report = bool(plot)

    def output_schema(self) -> tuple[FeatureOutput, ...]:
        return validate_schema(self.outputs)

    def reported_schema(self, use_mm_per_pixel=False):
        return effective_schema(self.output_schema(), use_mm_per_pixel)

    def convert_output(self, value, scale=1.0):
        values = scale_values(self.output_schema(), self.output_values(value), scale)
        return np.asarray(values, dtype=float) if self.is_vector else values[0]

    @property
    def is_vector(self) -> bool:
        return bool(self.output_schema()[0].key)

    def output_values(self, value) -> tuple:
        return output_values(self.output_schema(), value)

    def output_name_parts(self, component: str, **kwargs) -> Sequence[NamePart]:
        parts = list(self.name_parts(**kwargs))
        if component:
            output = next(item for item in self.output_schema() if item.key == component)
            position = next((i for i, part in enumerate(parts) if part.family), 0) + 1
            parts.insert(position, NamePart("component", (component,),
                         output.display_name or component.replace("_", " "), always_keep=True))
        return parts

    def output_description(self, component: str, use_mm_per_pixel=False, **kwargs) -> str:
        output = next(item for item in self.reported_schema(use_mm_per_pixel) if item.key == component)
        native = next(item for item in self.output_schema() if item.key == component)
        conversion = (f"Converted from native {native.units} using original-image mm_per_pixel × crop_scale_factor."
                      if use_mm_per_pixel and spatial_power(native.units) else "")
        return " ".join(text for text in (self.description(**kwargs), output.description,
                         conversion, f"Units: {output.units}." if output.units else "") if text)

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the feature value for provided domain arguments."""
        pass

    @abstractmethod
    def display_name(self, **kwargs) -> str:
        """Return the display name for the feature."""
        pass

    def name_tokens(self, **kwargs: Any) -> List[str]:
        """Return semantic tokens used to build the canonical machine name."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement canonical naming tokens"
        )

    def canonical_name(self, **kwargs: Any) -> str:
        """Return the canonical machine-readable feature name."""
        return "_".join(token for token in self.name_tokens(**kwargs) if token)

    def name_parts(self, **kwargs: Any) -> Sequence[NamePart]:
        """Return structured naming parts for feature-set-aware naming."""
        return (
            NamePart(
                key="feature",
                tokens=tuple(self.name_tokens(**kwargs)),
                display=self.display_name(**kwargs),
                family=True,
            ),
        )


    def implementation_description(self, **kwargs: Any) -> str:
        """Return a brief publication-facing explanation of the measurement."""
        return ""

    def aggregation_unit(self, **kwargs: Any) -> str:
        """Return the units summarized by an optional aggregator."""
        return "measurements"

    def aggregation_description(self, **kwargs: Any) -> str:
        """Return a publication-facing explanation of aggregation."""
        aggregator = getattr(self, "aggregator", None)
        describe = getattr(aggregator, "describe", None)
        if describe is None:
            return ""
        return describe(self.aggregation_unit(**kwargs))

    def region_description(self, **kwargs: Any) -> str:
        """Return the configured region or the full-image fallback."""
        region = getattr(self, "hemifield_spec", None)
        if region is None:
            region = getattr(self, "grid_field_spec", None)
        if region is None:
            return "Computed on the full image."
        return f"Computed in {region.description}."

    def description(self, **kwargs: Any) -> str:
        """Compose a publication-facing biomarker description."""
        general = self.general_description.strip()
        if not general:
            return "No publication description available."
        parts = (
            general,
            self.implementation_description(**kwargs),
            self.aggregation_description(**kwargs),
            self.region_description(**kwargs),
        )
        return " ".join(part.strip() for part in parts if part and part.strip())
    @abstractmethod
    def _plot(self, ax: 'Axes', layer: Any, **kwargs: Any) -> 'Axes':
        """Subclass draws onto ax for the given layer and returns ax."""

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if self.is_vector:
            values = self.output_values(value)
            return "\n".join(
                f"{output.display_name or output.key}: " + ("N/A" if np.isnan(v) else f"{v:.3g}")
                for output, v in zip(self.output_schema(), values)
            )
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            return "N/A"
        if isinstance(value, np.generic):
            display_value = value.item()
        else:
            display_value = value
            
        if isinstance(display_value, (int, float)) and not isinstance(display_value, bool):
            return f"{display_value:.3g}"
        else:
            return str(display_value)

    def _get_retina_for_plot(self, layer: Any) -> Any:
        """Return the retina object associated with a feature plot target."""
        if getattr(layer, "fovea_location", None) is not None:
            return layer
        return getattr(layer, "retina", None)

    def _plot_fovea_location(self, ax: 'Axes', layer: Any) -> 'Axes':
        """Overlay the fovea marker using the retina plotting helper."""
        retina = self._get_retina_for_plot(layer)
        if retina is None or getattr(retina, "fovea_location", None) is None:
            return ax

        plot = getattr(retina, "plot", None)
        if plot is None:
            return ax

        try:
            parameters = signature(plot).parameters
        except (TypeError, ValueError):
            return ax

        if "fovea" not in parameters:
            return ax

        plot_kwargs = {
            "ax": ax,
            "image": False,
            "disc": False,
            "fovea": True,
            "bounds": False,
        }
        if "av" in parameters:
            plot_kwargs["av"] = False

        return plot(**plot_kwargs)

    def plot_description(self) -> str:
        """Return the explanatory caption for the configured measurement."""
        return getdoc(self._plot) or ""

    def plot(self, ax: 'Axes', layer: Any, **kwargs: Any) -> 'Axes':
        """Draw the biomarker, its value, and the caption from ``_plot.__doc__``.

        Pass ``plot_caption=False`` to omit the caption in custom layouts.
        """
        from .inputs import prepare_retina, processed_mm_per_pixel
        use_mm = kwargs.pop("use_mm_per_pixel", False)
        use_fovea = kwargs.pop("use_fovea", True)
        retina = getattr(layer, "retina", layer)
        if hasattr(retina, "fovea_location"):
            prepared = prepare_retina(retina, use_mm_per_pixel=use_mm, use_fovea=use_fovea)
            if prepared is not retina:
                layer = prepared if layer is retina else prepared.layers[layer.name]
            retina = prepared
        # Supplied computed_value is already in reported units.
        display_scale = kwargs.pop("display_scale", None)
        if display_scale is None:
            display_scale = processed_mm_per_pixel(retina) if use_mm else 1.0
        schema = kwargs.pop("display_schema", self.reported_schema(use_mm))
        plot_fovea = kwargs.pop("plot_fovea", True)
        plot_caption = kwargs.pop("plot_caption", True)
        computed_value = kwargs.pop("computed_value", _MISSING)
        value = (
            self.compute(layer, **kwargs)
            if computed_value is _MISSING
            else computed_value
        )
        if computed_value is _MISSING:
            value = self.convert_output(value, display_scale)
        self.output_values(value)
        # Local drawing context; no feature instance state is changed.
        ax._vascx_output_schema = schema
        ax._vascx_display_scale = display_scale
        ax = self._plot_with_value(ax, layer, value, **kwargs)
        if plot_fovea:
            ax = self._plot_fovea_location(ax, layer)

        self._annotate_value(ax, layer, value)
        if plot_caption:
            _add_plot_caption(ax, self.plot_description())

        return ax

    def _plot_with_value(self, ax, layer, value, **kwargs):
        return self._plot(ax, layer, **kwargs)

    def _annotate_value(self, ax, layer, value):
        text = self._format_value(value)
        schema = getattr(ax, "_vascx_output_schema", self.output_schema())
        if not self.is_vector and schema[0].units not in ("", "dimensionless"):
            text += " " + schema[0].units
        elif self.is_vector:
            text = "\n".join(
                f"{o.display_name or o.key}: {v:.3g} {o.units}" if np.isfinite(v)
                else f"{o.display_name or o.key}: N/A"
                for o, v in zip(schema, self.output_values(value)))
        ax.text(0.01, 0.99, text, transform=ax.transAxes,
                ha="left", va="top", color="white", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))

    def plot_figure(
        self,
        layer: Any,
        figsize: Tuple[float, float] | None = None,
        dpi: int = 128,
        axis_off: bool = True,
        **kwargs: Any,
    ):
        """Render this feature plot into a Matplotlib figure."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        figsize = figsize if figsize is not None else getattr(self, "preferred_plot_size", (8, 8))
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, layout="constrained")
        self.plot(ax, layer, **kwargs)
        if axis_off:
            ax.set_axis_off()
        return fig

    def plot_png_bytes(
        self,
        layer: Any,
        figsize: Tuple[float, float] | None = None,
        dpi: int = 128,
        axis_off: bool = True,
        bbox_inches: str = "tight",
        pad_inches: float = 0.0,
        close: bool = True,
        **kwargs: Any,
    ) -> bytes:
        """Render this feature plot to PNG bytes using a non-interactive backend."""
        import io

        fig = self.plot_figure(
            layer=layer,
            figsize=figsize,
            dpi=dpi,
            axis_off=axis_off,
            **kwargs,
        )
        from matplotlib import pyplot as plt
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
        if close:
            plt.close(fig)
        return buf.getvalue()

    def plot_base64_png(self, layer: Any, **kwargs: Any) -> str:
        """Render this feature plot to a base64-encoded PNG string."""
        import base64

        png = self.plot_png_bytes(layer=layer, **kwargs)
        return base64.b64encode(png).decode("ascii")

    def plot_data_uri(self, layer: Any, **kwargs: Any) -> str:
        """Render this feature plot to a browser-ready PNG data URI."""
        return f"data:image/png;base64,{self.plot_base64_png(layer=layer, **kwargs)}"
