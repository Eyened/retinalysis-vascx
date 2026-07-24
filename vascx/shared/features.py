from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Any, Iterable, List, Sequence, Tuple

from .naming import (
    FeatureName,
    NamePart,
    NamingConvention,
    coerce_naming_convention,
    make_feature_names,
)

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


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

    @abstractmethod
    def _plot(self, ax: 'Axes', layer: Any, **kwargs: Any) -> 'Axes':
        """Subclass draws onto ax for the given layer and returns ax."""

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if value is None:
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

    def plot(self, ax: 'Axes', layer: Any, **kwargs: Any) -> 'Axes':
        """Compute value, delegate drawing to _plot, annotate value at upper-left, return ax."""
        plot_fovea = kwargs.pop("plot_fovea", True)
        value = self.compute(layer, **kwargs)
        ax = self._plot(ax, layer, **kwargs)
        if plot_fovea:
            ax = self._plot_fovea_location(ax, layer)

        # Display values starting from top-left, going down
        y_start = 0.99
        formatted_value = self._format_value(value)
        ax.text(0.01, y_start, formatted_value, transform=ax.transAxes, ha='left', va='top', color='white', fontsize=8)

        return ax

    def plot_figure(
        self,
        layer: Any,
        figsize: Tuple[float, float] = (8, 8),
        dpi: int = 128,
        axis_off: bool = True,
        **kwargs: Any,
    ):
        """Render this feature plot into a Matplotlib figure."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        self.plot(ax=ax, layer=layer, **kwargs)
        if axis_off:
            ax.set_axis_off()
        return fig

    def plot_png_bytes(
        self,
        layer: Any,
        figsize: Tuple[float, float] = (8, 8),
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
