"""Scalar feature dependencies, evaluated once per image and target.

Share configured feature instances to share computations. The cache never uses
resolved variable names, and never outlives an extraction call.
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from vascx.shared.features import Feature
from vascx.shared.naming import NamePart
from .base import LayerFeature, RetinaFeature


@dataclass(frozen=True)
class FeatureInput:
    feature: Feature
    target: str


class DerivedFeature(RetinaFeature):
    @property
    @abstractmethod
    def dependencies(self) -> Mapping[str, FeatureInput]:
        """Named scalar inputs; inputs may themselves be derived features."""

    @abstractmethod
    def compute_from(self, values: Mapping[str, object]):
        """Combine evaluated inputs without accessing the image."""

    def compute(self, retina):
        evaluator = FeatureEvaluator(retina)
        evaluator.validate([self])
        return evaluator.evaluate(self, "retina")

    def _plot(self, ax, retina, **kwargs):
        """The fundus image provides context for the derived measurement.
        Its component measurements have their own explanatory plots.
        """
        ax.imshow(retina.image)
        return ax


class FeatureEvaluator:
    def __init__(self, retina):
        self.retina = retina
        self.values = {}
        self.failures = {}
        self.active = set()

    def validate(self, features):
        """Reject dependency cycles and incompatible targets before computing."""
        visited, active = set(), set()

        def visit(feature):
            key = id(feature)
            if key in active:
                raise ValueError("Cycle in derived feature dependencies")
            if key in visited:
                return
            active.add(key)
            if isinstance(feature, DerivedFeature):
                for dependency in feature.dependencies.values():
                    if not isinstance(dependency, FeatureInput):
                        raise TypeError("Dependencies must be FeatureInput instances")
                    if dependency.target not in self.retina._target_names_for_feature(dependency.feature):
                        raise ValueError(f"Invalid dependency target: {dependency.target}")
                    visit(dependency.feature)
            active.remove(key)
            visited.add(key)

        for feature in features:
            visit(feature)

    def evaluate(self, feature, target):
        key = (id(feature), target)
        if key in self.values:
            return self.values[key]
        if key in self.failures:
            raise self.failures[key]
        if key in self.active:
            raise ValueError("Cycle in derived feature dependencies")
        self.active.add(key)
        try:
            if isinstance(feature, DerivedFeature):
                value = feature.compute_from({
                    name: self.evaluate(dep.feature, dep.target)
                    for name, dep in feature.dependencies.items()
                })
            else:
                targets = dict(self.retina.feature_targets(feature))
                if target not in targets:
                    raise ValueError(f"Unavailable feature target: {target}")
                value = feature.compute(targets[target])
            self.values[key] = value
            return value
        except Exception as exc:
            self.failures[key] = exc
            raise
        finally:
            self.active.remove(key)


class AVR(DerivedFeature):
    """Artery/vein ratio of the same configured A/V measurement."""
    general_description = (
        "Artery-vein ratio compares the arterial and venous values of the same measurement."
    )

    def __init__(self, source: LayerFeature):
        if not isinstance(source, LayerFeature):
            raise TypeError("AVR requires a feature with artery and vein targets")
        super().__init__(plot=False)
        self.source = source
        self.grid_field_spec = getattr(source, "grid_field_spec", None)
        self.hemifield_spec = getattr(source, "hemifield_spec", None)

    @property
    def dependencies(self):
        return {"arterial": FeatureInput(self.source, "arteries"),
                "venous": FeatureInput(self.source, "veins")}

    def compute_from(self, values):
        a, v = values["arterial"], values["venous"]
        if a is None or v is None or not np.isfinite(a) or not np.isfinite(v) or v == 0:
            return None
        ratio = float(a) / float(v)
        return ratio if np.isfinite(ratio) else None

    def display_name(self, **kwargs):
        return f"Artery-vein ratio of {self.source.display_name(layer_name='retina')}"

    def name_parts(self, **kwargs):
        parts = [NamePart("derived", ("avr",), "Artery-vein ratio of", always_keep=True)]
        parts.extend(replace(part) for part in self.source.name_parts(layer_name="retina")
                     if part.key != "layer")
        return parts

    def name_tokens(self, **kwargs):
        return [token for part in self.name_parts() for token in part.tokens]

    def implementation_description(self, **kwargs):
        a = self.source.canonical_name(layer_name="arteries")
        v = self.source.canonical_name(layer_name="veins")
        return f"Calculated as `{a}` / `{v}`. Missing inputs or a zero denominator give a missing result."

    def region_description(self, **kwargs):
        return self.source.region_description(**kwargs)
