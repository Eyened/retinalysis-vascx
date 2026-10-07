"""Named vector outputs remain numeric and preserve scalar/export conventions."""
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vascx.fundus.features.base import RetinaFeature
from vascx.fundus.features.rps import BackgroundLab
from vascx.fundus.features.od_morphology import ODMorphology
from vascx.fundus.retina import Retina
from vascx.shared.features import FeatureOutput, FeatureSet
from vascx.shared.outputs import output_values, validate_schema
from vascx.shared.naming import make_feature_names, make_output_names
from vascx.utils.biomarker_report import render_biomarker_plots
from vascx.utils.feature_docs import get_biomarker_definitions


@pytest.fixture(autouse=True)
def isolate_feature_registry(monkeypatch):
    monkeypatch.setattr(FeatureSet, "_registry", dict(FeatureSet._registry))


class VectorExample(RetinaFeature):
    outputs = (FeatureOutput('x', 'X', 'Horizontal coordinate.', 'px'),
               FeatureOutput('y', 'Y', 'Vertical coordinate.', 'px'))

    def feature_name_tokens(self):
        return ['example']

    def display_name(self, **kwargs):
        return 'Example - IM'

    def compute(self, retina):
        self.calls = getattr(self, 'calls', 0) + 1
        return np.array([2., np.nan])

    def _plot(self, ax, retina, **kwargs):
        """The image provides context for both coordinates."""
        ax.imshow(np.zeros((10, 10)))
        return ax


class ScalarExample(VectorExample):
    outputs = (FeatureOutput(),)

    def feature_name_tokens(self):
        return ['scalar']

    def compute(self, retina):
        return 3.


def test_flat_structured_and_scalar_outputs():
    retina = Retina(layers={}, resolution=(32, 32))
    vector = VectorExample()
    fs = FeatureSet('test_vector_extraction', [vector, ScalarExample()])
    flat = retina.calc_features(fs)
    assert list(flat) == ['example_x', 'example_y', 'scalar']
    assert flat['example_x'] == 2 and np.isnan(flat['example_y'])
    assert flat['scalar'] == 3 and vector.calls == 1
    assert all(np.isscalar(v) for v in flat.values())
    structured = retina.calc_features(fs, structured=True)
    np.testing.assert_equal(structured['example'], [2., np.nan])
    assert structured['scalar'] == 3 and vector.calls == 2
    canonical = retina.calc_features(fs, naming='canonical')
    assert list(canonical) == ['example_x_retina', 'example_y_retina', 'scalar_retina']


@pytest.mark.parametrize('value', [[1], [1, 2, 3], [[1, 2]], ['a', 'b'], [1, np.inf]])
def test_invalid_vector_is_a_programming_error(value):
    feature = VectorExample()
    feature.compute = lambda _: value
    with pytest.raises(ValueError, match='vector|finite'):
        Retina(layers={}, resolution=(32, 32)).calc_features(
            FeatureSet('test_invalid_vector', [feature]))


def test_missing_vector_preserves_every_component():
    feature = VectorExample()
    feature.compute = lambda _: None
    result = Retina(layers={}, resolution=(32, 32)).calc_features(
        FeatureSet('test_missing_vector', [feature]))
    assert set(result) == {'example_x', 'example_y'}
    assert all(np.isnan(v) for v in result.values())


@pytest.mark.parametrize('schema', [(), (FeatureOutput('x'), FeatureOutput('x')),
    (FeatureOutput(), FeatureOutput('x')), (FeatureOutput('Bad Name'),)])
def test_invalid_schemas_are_rejected(schema):
    with pytest.raises(ValueError):
        validate_schema(schema)


def test_single_component_vector_and_legacy_scalar():
    assert output_values((FeatureOutput('x'),), [4]) == (4.,)
    assert output_values((FeatureOutput(),), None) == (None,)
    with pytest.raises(ValueError, match='Scalar'):
        output_values((FeatureOutput(),), [1, 2])


def test_component_metadata_and_lab_column_compatibility():
    fs = FeatureSet('test_vector_metadata', [ODMorphology(), BackgroundLab()])
    for naming, suffix in [('resolved', ''), ('canonical', '_retina')]:
        definitions = get_biomarker_definitions(fs, naming=naming)
        assert len(definitions) == 15
        assert len({d.display_name for d in definitions}) == 15
        assert {d.variable for d in definitions if d.feature_group == 'background_lab'+suffix} == {
            'background_lab_'+c+suffix for c in ('l', 'a', 'b')}
        centroid = next(d for d in definitions if d.component == 'centroid_x')
        assert centroid.units == 'px' and 'estimated' in centroid.description
        assert centroid.feature_group == 'od'+suffix
        names = make_output_names(fs, Retina._target_names_for_feature, naming)
        assert {n.name for n in names.values()} == {d.variable for d in definitions}
        assert len(make_feature_names(fs, Retina._target_names_for_feature, naming)) == 2


def test_vector_report_uses_stored_values_once_and_links_every_component(tmp_path, monkeypatch):
    feature = VectorExample(plot=True)
    def unexpected_compute(_):
        raise AssertionError('Report must use stored vector components')
    monkeypatch.setattr(feature, 'compute', unexpected_compute)
    fs = FeatureSet('test_vector_report', [feature])
    retina = Retina(layers={}, resolution=(32, 32))
    # A partial dataframe must retain missing components instead of recomputing.
    frame = pd.DataFrame({'example_x': [9.]}, index=['sample'])
    from matplotlib.figure import Figure
    original = Figure.savefig
    def check(fig, *args, **kwargs):
        labels = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any('X: 9' in text and 'Y: N/A' in text for text in labels)
        return original(fig, *args, **kwargs)
    monkeypatch.setattr(Figure, 'savefig', check)
    paths = render_biomarker_plots([('sample', retina)], fs, tmp_path, dataframe=frame)
    assert paths['example_x'] == paths['example_y']
    assert len(list((tmp_path/'plots').glob('*.png'))) == 1


def test_faz_extraction_supports_the_same_vector_contract():
    from vascx.faz.retina import FAZRetina
    from vascx.faz.features.base import FAZLayerFeature
    class FAZVector(FAZLayerFeature):
        outputs = VectorExample.outputs
        def compute(self, layer):
            return [1., 2.]
        def name_tokens(self, layer_name=None, **kwargs):
            return ['coordinates', layer_name]
        def display_name(self, **kwargs):
            return 'Coordinates'
        def _plot(self, ax, layer, **kwargs):
            return ax
    retina = SimpleNamespace(layers={'vessels': object()})
    fs = FeatureSet('test_faz_vector', [FAZVector()])
    flat = FAZRetina.calc_features(retina, fs, naming='canonical')
    assert len(flat) == 2 and list(flat.values()) == [1., 2.]
    structured = FAZRetina.calc_features(retina, fs, structured=True)
    np.testing.assert_equal(next(iter(structured.values())), [1., 2.])
