from types import SimpleNamespace
import uuid

import numpy as np
import pytest
from joblib import Parallel, delayed

from vascx.fundus.features.base import LayerFeature
from vascx.fundus.features.cre import CRE, CREMode
from vascx.fundus.features.derived import AVR, DerivedFeature, FeatureEvaluator, FeatureInput
from vascx.fundus.retina import Retina
from vascx.shared.features import FeatureSet
from vascx.utils.feature_docs import get_biomarker_definitions


class Measurement(LayerFeature):
    def __init__(self):
        super().__init__()
        self.calls = []

    def compute(self, layer):
        self.calls.append(layer.name)
        return layer.value

    def display_name(self, **kwargs):
        return "Measurement"

    def feature_name_tokens(self):
        return ["measurement"]

    def _plot(self, ax, layer, **kwargs):
        return ax


class ExampleRetina:
    _target_names_for_feature = Retina._target_names_for_feature
    calc_features = Retina.calc_features
    _register_canonical_name = Retina._register_canonical_name
    id = "example"
    fovea_location = (0., 0.)

    def __init__(self, a=2, v=4):
        self.layers = {"arteries": SimpleNamespace(name="arteries", value=a),
                       "veins": SimpleNamespace(name="veins", value=v)}

    def feature_targets(self, feature):
        return [(target, self if target == "retina" else self.layers[target])
                for target in self._target_names_for_feature(feature)]


def feature_set(features):
    return FeatureSet(str(uuid.uuid4()), features)


@pytest.mark.parametrize("derived_first", [False, True])
def test_dependency_reuse_order_and_hidden_inputs(derived_first):
    source = Measurement()
    ratio = AVR(source)
    roots = [ratio, source] if derived_first else [source, ratio]
    result = ExampleRetina().calc_features(feature_set(roots))
    assert result["avr_measurement"] == 0.5
    assert source.calls == ["arteries", "veins"]
    source.calls.clear()
    assert ExampleRetina().calc_features(feature_set([ratio])) == {"avr_measurement": 0.5}
    assert source.calls == ["arteries", "veins"]


@pytest.mark.parametrize("a,v", [(None, 4), (2, None), (np.nan, 4), (2, 0), (np.inf, 4)])
def test_missing_ratios_and_cached_missing(a, v):
    source = Measurement()
    evaluator = FeatureEvaluator(ExampleRetina(a, v))
    ratio = AVR(source)
    assert evaluator.evaluate(ratio, "retina") is None
    assert evaluator.evaluate(ratio, "retina") is None
    assert source.calls == ["arteries", "veins"]


def test_standalone_parallel_and_per_image_lifetime():
    ratio = AVR(Measurement())
    assert ratio.compute(ExampleRetina()) == 0.5
    results = Parallel(n_jobs=2)(delayed(ratio.compute)(ExampleRetina(a, 4)) for a in [1, 2])
    assert results == [0.25, 0.5]
    assert ratio.compute(ExampleRetina(3, 4)) == 0.75


def test_invalid_bindings_and_cycles():
    class Custom(AVR):
        @property
        def dependencies(self):
            return self.inputs
    feature = Custom(Measurement())
    feature.inputs = {"x": FeatureInput(feature.source, "retina")}
    with pytest.raises(ValueError, match="Invalid dependency target"):
        FeatureEvaluator(ExampleRetina()).validate([feature])
    feature.inputs = {"x": FeatureInput(feature, "retina")}
    with pytest.raises(ValueError, match="Cycle"):
        FeatureEvaluator(ExampleRetina()).validate([feature])


def test_distinct_instances_are_independent():
    a, b = Measurement(), Measurement()
    evaluator = FeatureEvaluator(ExampleRetina())
    evaluator.evaluate(a, "arteries")
    evaluator.evaluate(b, "arteries")
    assert a.calls == b.calls == ["arteries"]


@pytest.mark.parametrize("feature_set_name, expected", [
    ("macula_centered", {"avr_cre", "avr_cre_aric"}),
    ("macula_centered_rs", {"avr_cre", "avr_cre_aric"}),
    ("od_centered", {
        "avr_full_cre", "avr_nasal_cre", "avr_temporal_cre", "avr_cre_aric",
    }),
    ("od_centered_rs", {
        "avr_full_cre", "avr_nasal_cre", "avr_temporal_cre", "avr_cre_aric",
    }),
    ("od_centered_narrow_rs", {
        "avr_full_cre", "avr_nasal_cre", "avr_temporal_cre", "avr_cre_aric",
    }),
])
def test_feature_sets_include_expected_avr_names(feature_set_name, expected):
    definitions = get_biomarker_definitions(feature_set_name)
    ratios = [item for item in definitions if item.variable.startswith("avr_")]
    assert {item.variable for item in ratios} == expected
    assert all(" / " in item.description for item in ratios)


def test_avr_canonical_name_preserves_source_parameters():
    ratio = AVR(CRE(CREMode.Temporal, max_vessels=3))
    assert ratio.canonical_name() == "avr_temporal_cre_max_vessels_3"


def test_chained_dependencies_and_cached_errors():
    class Doubled(AVR):
        @property
        def dependencies(self):
            return {"ratio": FeatureInput(self.ratio, "retina")}

        def compute_from(self, values):
            return 2 * values["ratio"]

    source = Measurement()
    ratio = AVR(source)
    doubled = Doubled(source)
    doubled.ratio = ratio
    evaluator = FeatureEvaluator(ExampleRetina())
    evaluator.validate([doubled])
    assert evaluator.evaluate(doubled, "retina") == 1
    assert evaluator.evaluate(ratio, "retina") == .5
    assert source.calls == ["arteries", "veins"]

    class Broken(Measurement):
        def compute(self, layer):
            self.calls.append(layer.name)
            raise ValueError("bad measurement")
    broken = Broken()
    for _ in range(2):
        with pytest.raises(ValueError, match="bad measurement"):
            evaluator.evaluate(broken, "arteries")
    assert broken.calls == ["arteries"]


def test_derived_report_reuses_source_grid(tmp_path):
    import pandas as pd
    from vascx.utils.biomarker_report import render_biomarker_plots
    source = Measurement()
    source.plot_in_report = True
    fs = feature_set([source, AVR(source)])
    frame = pd.DataFrame({"measurement_arteries":[2.], "measurement_veins":[4.],
                          "avr_measurement":[.5]}, index=["example"])
    links = render_biomarker_plots([("example", ExampleRetina())], fs, tmp_path, dataframe=frame)
    assert links["avr_measurement"] == links["measurement_arteries"]
    assert len(list((tmp_path / "plots").glob("*.png"))) == 1
