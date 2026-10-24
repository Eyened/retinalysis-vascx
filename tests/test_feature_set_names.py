"""Feature-set construction supplies the same bindings to every consumer."""
import pickle

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from vascx.fundus.features.base import RetinaFeature
from vascx.fundus.features.cre import CRE, CREMode
from vascx.fundus.features.derived import AVR
from vascx.fundus.features.od_morphology import ODMorphology
from vascx.fundus.features.tortuosity import Tortuosity
from vascx.fundus.retina import Retina
from vascx.shared.features import FeatureSet
from vascx.shared.naming import NamingConvention
from vascx.shared.outputs import FeatureOutput
from vascx.utils.biomarker_report import render_biomarker_plots
from vascx.utils.feature_docs import get_biomarker_definitions


@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch):
    monkeypatch.setattr(FeatureSet, "_registry", dict(FeatureSet._registry))


class Coordinates(RetinaFeature):
    outputs = (FeatureOutput("x"), FeatureOutput("y"))

    def feature_name_tokens(self):
        return ["coordinates"]

    def display_name(self, **kwargs):
        return "Coordinates"

    def compute(self, retina):
        self.calls = getattr(self, "calls", 0) + 1
        return np.array([2., 4.])

    def _plot(self, ax, retina, **kwargs):
        """The image shows the location of the measured coordinates."""
        ax.imshow(np.zeros((8, 8)))
        return ax


def test_constructor_resolves_layers_components_and_derived_features():
    cre = CRE(CREMode.Full)
    od = ODMorphology()
    ratio = AVR(cre)
    fs = FeatureSet("bindings", [cre, od, ratio])
    artery = fs.resolve("cre_arteries")
    vein = fs.resolve("cre_veins")
    assert artery.feature is vein.feature is cre
    assert (artery.target, vein.target) == ("arteries", "veins")
    assert artery.component == ""
    assert artery.canonical_name == "full_cre_arteries"
    assert artery.resolved_name == artery.name == "cre_arteries"
    canonical = fs.resolve(artery.canonical_name, naming="canonical")
    assert canonical.feature is cre and canonical.resolved_name == artery.name
    area = fs.resolve("od_area")
    distance = fs.resolve("od_fovea_distance")
    assert area.feature is distance.feature is od
    assert area.component == "area" and distance.component == "fovea_distance"
    assert area.target == "retina"
    assert fs.resolve("avr_cre").feature is ratio
    assert fs.resolve("avr_cre").feature.source is cre


def test_membership_and_indexes_are_read_only():
    features = [CRE()]
    fs = FeatureSet("immutable_membership", features)
    features.append(ODMorphology())
    assert len(fs) == 1
    with pytest.raises(AttributeError):
        fs.features.append(ODMorphology())
    with pytest.raises(AttributeError):
        fs.features = ()
    with pytest.raises(TypeError):
        fs.names().by_name["new"] = fs.resolve("cre_arteries")
    with pytest.raises(TypeError):
        fs.names().outputs[(0, "arteries", "")] = None
    with pytest.raises(KeyError, match="Unknown resolved output.*immutable_membership"):
        fs.resolve("od_area")


def test_duplicate_outputs_fail_before_registration():
    with pytest.raises(ValueError, match="Duplicate canonical"):
        FeatureSet("duplicate_outputs", [CRE(), CRE()])
    assert FeatureSet.get_by_name("duplicate_outputs") is None


def test_explicit_rebuild_refreshes_names_after_parameter_changes():
    variant = Tortuosity(max_segment_len=0.2)
    fs = FeatureSet("rebuild_names", [Tortuosity(), variant])
    old = fs.resolve("tort_dist_max_segment_len_0p2_arteries")
    variant.max_segment_len = 0.3
    fs.rebuild_names()
    assert fs.resolve("tort_dist_max_segment_len_0p3_arteries").feature is variant
    with pytest.raises(KeyError):
        fs.resolve(old.name)
    with pytest.raises(KeyError):
        fs.resolve(old.canonical_name, naming="canonical")


def test_consumers_reuse_the_constructed_index(tmp_path, monkeypatch):
    feature = Coordinates(plot=True)
    fs = FeatureSet("shared_consumers", [feature])
    index = fs.names()
    assert fs.names(NamingConvention.RESOLVED) is index
    assert fs.names(target_names_for_feature=Retina._target_names_for_feature) is index

    def unexpected(*args, **kwargs):
        raise AssertionError("Consumers must reuse the feature-set name mapping")

    monkeypatch.setattr("vascx.shared.naming.make_output_names", unexpected)
    monkeypatch.setattr("vascx.shared.naming.make_feature_names", unexpected)
    retina = Retina(layers={}, resolution=(8, 8), fovea_location=(4, 4))
    values = retina.calc_features(fs)
    assert values == {"coordinates_x": 2., "coordinates_y": 4.}
    assert feature.calls == 1
    assert retina.calc_features(fs, naming="canonical") == {
        "coordinates_x_retina": 2., "coordinates_y_retina": 4.}
    np.testing.assert_equal(retina.calc_features(fs, structured=True)["coordinates"], [2., 4.])
    definitions = get_biomarker_definitions(fs)
    assert {item.variable for item in definitions} == set(index.by_name)
    assert set(Retina.make_feature_display_names(fs)) == set(index.by_name)
    links = render_biomarker_plots(
        [("sample", retina)], fs, tmp_path,
        dataframe=pd.DataFrame([values], index=["sample"]),
    )
    assert links["coordinates_x"] == links["coordinates_y"]
    assert feature.calls == 3  # Report plots reused the extracted component values.


def test_target_layouts_have_separate_reusable_indexes():
    from vascx.faz.features.base import FAZLayerFeature
    from vascx.faz.retina import FAZRetina
    from types import SimpleNamespace

    class FAZMeasurement(FAZLayerFeature):
        def compute(self, layer):
            return 7.

        def name_tokens(self, layer_name=None, **kwargs):
            return ["measurement", layer_name]

        def display_name(self, **kwargs):
            return "Measurement"

        def _plot(self, ax, layer, **kwargs):
            return ax

    fs = FeatureSet("faz_target_layouts", [FAZMeasurement()])
    all_layers = fs.names()
    vessels = fs.names(target_names_for_feature=lambda feature: ("vessels",))
    assert vessels is not all_layers
    assert vessels is fs.names(target_names_for_feature=lambda feature: ("vessels",))
    assert len(all_layers.outputs) == 3 and len(vessels.outputs) == 1
    retina = SimpleNamespace(layers={"vessels": object()})
    assert FAZRetina.calc_features(retina, fs) == {"measurement_vessels": 7.}
    assert fs.names() is all_layers


def test_pickling_keeps_bindings_and_derived_dependencies_together():
    source = CRE()
    fs = FeatureSet("pickled_bindings", [source, AVR(source)])
    restored = pickle.loads(pickle.dumps(fs))
    assert restored.resolve("cre_arteries").feature is restored.features[0]
    assert restored.resolve("avr_cre").feature.source is restored.features[0]
    assert restored.names() is restored.names()
