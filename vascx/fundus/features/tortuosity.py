from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from rtnls_enface.grids.specifications import BaseGridFieldSpecification

from vascx.shared.aggregators import LengthWeightedAggregator, median
from vascx.shared.segment import Segment
from vascx.shared.vessels import Vessels

from .base import LayerFeature, grid_field_fraction_in_bounds

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


class TortuosityMode(str, Enum):
    Segments = "segments"
    Vessels = "vessels"


class LengthMeasure(str, Enum):
    Splines = "spl"
    Skeleton = "skl"


class TortuosityMeasure(str, Enum):
    Distance = "dist"
    Curvature = "curv"
    Inflections = "infl"


class Tortuosity(LayerFeature):
    """Segment- or vessel-level tortuosity by distance ratio, curvature, or inflection counts.

    Representation: Uses segments or resolved_segments; for curvature uses Segment spline;
    optionally length-weighted. Operates on either individual segments or merged vessel trees.

    Computation: Measures vessel tortuosity using three methods: distance ratio (arc length / chord length),
    curvature (mean curvature along spline), or inflection counts (number of direction changes).
    Can be computed per-segment or per-vessel (resolved segments), with optional spline-length weighting.

    Args (constructor):
    - mode: `TortuosityMode` selecting per-segment or per-vessel computation.
    - measure: `TortuosityMeasure` (distance, curvature, inflections).
    - length_measure: length source (splines or skeleton) for distance-based measure.
    - min_numpoints: minimum skeleton points required for inclusion.
    - max_segment_len: optional maximum segment-piece length as a fraction of OD-fovea
      distance; only applied for segment-mode distance tortuosity.
    - max_tortuosity: optional upper bound for distance-based tortuosity; values above it are discarded.
    - grid_field: optional `GridFieldEnum` restricting segments to a region.
    - aggregator: callable aggregator returning a single scalar over per-entity values.
    """

    # Ideas
    # tortuosity for different levels of caliber
    #   what happens when small vessels not visible
    # tortuosity for different generations

    def __init__(
        self,
        mode: TortuosityMode = TortuosityMode.Segments,
        measure: TortuosityMeasure = TortuosityMeasure.Distance,
        length_measure: LengthMeasure = LengthMeasure.Splines,
        min_numpoints: int = 25,
        max_segment_len: Optional[float] = None,
        max_tortuosity: Optional[float] = None,
        grid_field: Optional[BaseGridFieldSpecification] = None,
        aggregator: Callable = median,
        spline_error_fraction: Optional[float] = None,
    ):
        """Configure tortuosity computation and optional segment filtering."""
        self.mode = mode
        self.measure = measure
        self.length_measure = length_measure
        self.min_numpoints = min_numpoints
        self.max_segment_len = (
            None if max_segment_len is None else float(max_segment_len)
        )
        self.max_tortuosity = self._default_max_tortuosity(max_tortuosity)
        self.spline_error_fraction = (
            None if spline_error_fraction is None else float(spline_error_fraction)
        )
        super().__init__(grid_field_spec=grid_field)
        self.aggregator = aggregator

    def _default_max_tortuosity(
        self, max_tortuosity: Optional[float]
    ) -> Optional[float]:
        if max_tortuosity is not None:
            return max_tortuosity
        if self.measure == TortuosityMeasure.Distance:
            return 1.5
        return None

    def _get_curvature_scale(self, layer: VesselTreeLayer):
        if layer.retina.fovea_location is None:
            warnings.warn("Fovea location not known, using default scale.")
            return 1.0
        return layer.retina.disc_fovea_distance

    def _resolved_spline_error_fraction(self) -> float:
        if self.spline_error_fraction is not None:
            return self.spline_error_fraction
        if self.measure in (
            TortuosityMeasure.Curvature,
            TortuosityMeasure.Inflections,
        ):
            return 0.25
        return 0.05

    def _compute_for_segment(self, segment: Segment):
        spline_error_fraction = self._resolved_spline_error_fraction()
        if self.measure == TortuosityMeasure.Distance:
            if self.length_measure == LengthMeasure.Splines:
                spline = segment.get_spline(error_fraction=spline_error_fraction)
                tortuosity = (
                    spline.length() / segment.chord_length
                    if spline is not None
                    else np.nan
                )
            elif self.length_measure == LengthMeasure.Skeleton:
                tortuosity = np.mean(segment.length / segment.chord_length)
            else:
                raise NotImplementedError()
            if self.max_tortuosity is not None and tortuosity > self.max_tortuosity:
                return np.nan
            return tortuosity
        elif self.measure == TortuosityMeasure.Curvature:
            spline = segment.get_spline(error_fraction=spline_error_fraction)
            return np.mean(spline.curvatures()) * self._get_curvature_scale(
                segment.layer
            )
        elif self.measure == TortuosityMeasure.Inflections:
            spline = segment.get_spline(error_fraction=spline_error_fraction)
            return len(spline.inflection_points(every=10))
        else:
            raise NotImplementedError()

    def _compute_weight_for_segment(self, segment: Segment) -> float:
        spline = segment.get_spline(
            error_fraction=self._resolved_spline_error_fraction()
        )
        return spline.length() if spline is not None else np.nan

    def _resolved_max_segment_len_pixels(
        self, layer: VesselTreeLayer
    ) -> Optional[float]:
        if self.max_segment_len is None:
            return None

        disc_fovea_distance = getattr(layer.retina, "disc_fovea_distance", None)
        if disc_fovea_distance is None or not np.isfinite(disc_fovea_distance):
            warnings.warn(
                "Disc-fovea distance not known, skipping tortuosity segment splitting."
            )
            return None
        max_segment_len_pixels = self.max_segment_len * float(disc_fovea_distance)
        if max_segment_len_pixels <= 0:
            raise ValueError("max_segment_len must resolve to a positive pixel length")
        return max_segment_len_pixels

    def _should_split_segments(self) -> bool:
        return (
            self.mode == TortuosityMode.Segments
            and self.measure == TortuosityMeasure.Distance
            and self.max_segment_len is not None
        )

    def get_segments(self, layer: VesselTreeLayer):
        if self.mode == TortuosityMode.Segments:
            segments = layer.get_region_segments(self.grid_field_spec)
        elif self.mode == TortuosityMode.Vessels:
            segments = layer.get_region_resolved_segments(
                field_spec=self.grid_field_spec,
                spline_error_fraction=self._resolved_spline_error_fraction(),
            )
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        segments = [
            segment
            for segment in segments
            if len(segment.skeleton) >= self.min_numpoints
        ]

        if self._should_split_segments():
            max_segment_len_pixels = self._resolved_max_segment_len_pixels(layer)
            if max_segment_len_pixels is not None:
                split_segments: list[Segment] = []
                for segment in segments:
                    split_segments.extend(
                        segment.split_equal_length(max_segment_len_pixels)
                    )
                segments = split_segments
        return segments

    def raw(self, layer: VesselTreeLayer):
        segments = self.get_segments(layer)
        if isinstance(self.aggregator, LengthWeightedAggregator) and len(segments) < 5:
            return None

        vals = [self._compute_for_segment(vessel) for vessel in segments]

        if isinstance(self.aggregator, LengthWeightedAggregator):
            weights = [self._compute_weight_for_segment(seg) for seg in segments]
            return list(zip(weights, vals))

        return np.asarray(vals)

    def compute(self, layer: VesselTreeLayer):
        if self.grid_field_spec is not None:
            frac = grid_field_fraction_in_bounds(layer.retina, self.grid_field_spec)
            if frac < 0.5:
                return None
        tortuosities = self.raw(layer)
        if tortuosities is None:
            return None
        return self.aggregator(tortuosities)

    def display_name(self, layer_name: str, key: str = None) -> str:
        from .base import get_aggregator_prefix, get_grid_field_suffix, get_layer_suffix

        agg = get_aggregator_prefix(self.aggregator)
        if isinstance(self.aggregator, LengthWeightedAggregator):
            agg = "Length-Weighted "

        field = get_grid_field_suffix(self.grid_field_spec)
        layer = get_layer_suffix(layer_name)
        measure = self.measure.name.title()
        return f"{agg}{measure} Tortuosity{field}{layer}"

    def name_prefix_tokens(self) -> list[str]:
        from .base import get_aggregator_tokens

        if isinstance(self.aggregator, LengthWeightedAggregator):
            return ["lw"]
        return get_aggregator_tokens(self.aggregator)

    def feature_name_tokens(self) -> list[str]:
        return ["tort"]

    def parameter_name_tokens(self) -> list[str]:
        from .base import format_name_value

        tokens = [self.measure.value]
        if self.mode != TortuosityMode.Segments:
            tokens.append(self.mode.value)
        if self.length_measure != LengthMeasure.Splines:
            tokens.extend(["length", self.length_measure.value])
        if self.min_numpoints != 25:
            tokens.extend(["min_numpoints", str(self.min_numpoints)])
        if self._should_split_segments():
            tokens.extend(["max_segment_len", format_name_value(self.max_segment_len)])
        default_max_tortuosity = (
            1.5 if self.measure == TortuosityMeasure.Distance else None
        )
        if self.max_tortuosity != default_max_tortuosity:
            tokens.extend(["max_tortuosity", str(self.max_tortuosity)])
        if self._resolved_spline_error_fraction() != (
            0.25
            if self.measure
            in (TortuosityMeasure.Curvature, TortuosityMeasure.Inflections)
            else 0.05
        ):
            tokens.extend(
                [
                    "spline_error_fraction",
                    format_name_value(self._resolved_spline_error_fraction()),
                ]
            )
        return tokens

    def _plot(self, ax, layer: VesselTreeLayer, **kwargs):
        segments = self.get_segments(layer)

        vessels = Vessels(layer, segments)
        ax = vessels.plot(
            ax=ax,
            show_index=False,
            # text=lambda x: f"{100 * (self._compute_for_segment(x) - 1):.2f}",
            cmap="tab20",
            bounds=True,
            min_numpoints=0,
            min_numpoints_caliber=self.min_numpoints,
            endpoints=True,
            # segments=True,
            arcs=True,
            chords=True,
            spline_error_fraction=self._resolved_spline_error_fraction(),
            **kwargs,
        )

        # plot ETDRS region
        field = self._get_grid_field(layer)
        if field is not None:
            field.plot(ax)
        return ax
