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
    retina = Retina(layers={}, resolution=(32, 32), fovea_location=(16, 16))
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
        Retina(layers={}, resolution=(32, 32), fovea_location=(16, 16)).calc_features(
            FeatureSet('test_invalid_vector', [feature]))


def test_missing_vector_preserves_every_component():
    feature = VectorExample()
    feature.compute = lambda _: None
    result = Retina(layers={}, resolution=(32, 32), fovea_location=(16, 16)).calc_features(
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
    retina = Retina(layers={}, resolution=(32, 32), fovea_location=(16, 16))
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


def _bifurcation_layer(samples):
    """(angle, shortest daughter length, parent/child mean diameters)."""
    nodes = []
    for angle, length, diameters in samples:
        segments = [SimpleNamespace(get_mean_diameter=lambda smoothing, d=d: d) for d in diameters]
        nodes.append(SimpleNamespace(outgoing_min_length=length,
            angle=lambda delta, smoothing, a=angle: a,
            incoming=segments[0], outgoing=segments[1:]))
    return SimpleNamespace(filter_bifurcations=lambda field: nodes, retina=object())


def test_bifurcation_formulas_filtering_and_independent_sample_sizes():
    from vascx.fundus.features.bifurcation import Bifurcation
    samples = [(60, 30, (10, 8, 6)), (90, 30, (10, 7, 5)),
               (120, 30, (10, 0, 5)), (136, 30, (10, 8, 6)),
               (60, 20, (10, 8, 6)), (np.nan, 30, (10, 8, 6))]
    layer = _bifurcation_layer(samples)
    result = Bifurcation(min_bifurcations=2).compute(layer)
    expected_bc = np.mean([(8**2+6**2)/100, (7**2+5**2)/100])
    expected_jed = np.mean([np.cbrt(abs(1000-8**3-6**3))/10,
                           np.cbrt(abs(1000-7**3-5**3))/10])
    np.testing.assert_allclose(result, [90, expected_bc, expected_jed, 3])
    result = Bifurcation().compute(layer)
    assert result[0] == 90 and np.isnan(result[1:3]).all() and result[3] == 3


@pytest.mark.parametrize('scale,swap', [(1, False), (3.2, False), (1, True)])
def test_bifurcation_ratios_are_scale_and_daughter_order_invariant(scale, swap):
    from vascx.fundus.features.bifurcation import Bifurcation
    daughters = [8, 6][::-1] if swap else [8, 6]
    layer = _bifurcation_layer([(70, 40, np.array([10, *daughters])*scale)])
    result = Bifurcation(min_bifurcations=1).compute(layer)
    np.testing.assert_allclose(result, [70, 1, np.cbrt(272)/10, 1], atol=1e-12)


def test_bifurcation_murray_optimum_empty_and_threshold_boundaries():
    from vascx.fundus.features.bifurcation import Bifurcation
    d0 = np.cbrt(3**3+4**3)
    result = Bifurcation(min_bifurcations=1).compute(
        _bifurcation_layer([(135, 20.01, (d0, 3, 4))]))
    assert result[0] == 135 and result[3] == 1
    assert result[2] == pytest.approx(0., abs=1e-5)
    result = Bifurcation().compute(_bifurcation_layer([]))
    assert result[3] == 0 and np.isnan(result[:3]).all()
    result = Bifurcation().compute(_bifurcation_layer([(60, 30, (10, 8, 6))]))
    assert result[3] == 1 and np.isnan(result[:3]).all()


@pytest.mark.parametrize('diameters', [(np.nan, 8, 6), (0, 8, 6), (10, -8, 6),
                                       (10, np.inf, 6), (10, None, 6)])
def test_bifurcation_missing_diameters_preserve_angles_and_count(diameters):
    from vascx.fundus.features.bifurcation import Bifurcation
    result = Bifurcation(min_bifurcations=1).compute(_bifurcation_layer([(60, 30, diameters)]))
    assert result[0] == 60 and result[3] == 1 and np.isnan(result[1:3]).all()


def test_bifurcation_region_coverage_and_shared_measurement_cache(monkeypatch):
    from vascx.fundus.features.bifurcation import Bifurcation
    feature = Bifurcation(grid_field=object(), min_bifurcations=1)
    layer = _bifurcation_layer([(60, 30, (10, 8, 6))])
    monkeypatch.setattr('vascx.fundus.features.bifurcation.grid_field_passes_qc', lambda *a, **kw: False)
    assert np.isnan(feature.compute(layer)).all()
    feature = Bifurcation(min_bifurcations=1)
    first = feature._measurements(layer)
    def unexpected(*args):
        raise AssertionError('Cached records must serve computation and plotting')
    layer.filter_bifurcations = unexpected
    assert feature._measurements(layer) is first
    assert feature.compute(layer)[3] == 1


@pytest.mark.parametrize('feature_limit,field_limit,grid_limit,expected', [
    (.7, .8, .9, .7), (None, .8, .9, .8),
    (None, None, .9, .9), (None, None, None, .95),
])
def test_bifurcation_visibility_precedence_and_visible_empty_region(
    monkeypatch, feature_limit, field_limit, grid_limit, expected,
):
    from vascx.fundus.features.bifurcation import Bifurcation
    spec = SimpleNamespace(min_area_within_bounds=field_limit,
                           grid_spec=SimpleNamespace(min_area_within_bounds=grid_limit))
    feature = Bifurcation(grid_field=spec, min_area_within_bounds=feature_limit)
    field = object()
    monkeypatch.setattr(feature, '_get_grid_field', lambda layer: field)
    def check_coverage(retina, grid_spec, *, min_fraction_in_bounds):
        assert grid_spec is spec and min_fraction_in_bounds == expected
        return True
    monkeypatch.setattr('vascx.fundus.features.bifurcation.grid_field_passes_qc', check_coverage)
    layer = _bifurcation_layer([])
    def filtered_nodes(selected_field):
        assert selected_field is field
        return []
    layer.filter_bifurcations = filtered_nodes
    result = feature.compute(layer)
    assert result[3] == 0 and np.isnan(result[:3]).all()


def test_bifurcation_compound_names_and_feature_set_migration():
    from tests.regression_helpers import discover_feature_set_names
    from vascx.fundus.features.bifurcation import Bifurcation
    expected = {'full', 'full_v2', 'full_v3', 'macula_centered', 'macula_centered_rs',
                'od_centered', 'od_centered_rs', 'od_centered_narrow_rs'}
    actual = set()
    for name in discover_feature_set_names():
        fs = FeatureSet.get_by_name(name)
        features = [f for f in fs if isinstance(f, Bifurcation)]
        if features:
            actual.add(name)
        for feature in features:
            assert not hasattr(feature, 'aggregator')
        assert all(type(f).__name__ not in ('BifurcationAngles', 'BifurcationCount') for f in fs)
    assert actual == expected
    definitions = get_biomarker_definitions(FeatureSet('test_bif_names', [Bifurcation()]))
    assert {d.variable for d in definitions} == {
        f'bif_{key}_{layer}' for key in ('angle_mean', 'branching_coefficient_mean', 'jed_mean', 'count')
        for layer in ('arteries', 'veins')}
    for layer in ('arteries', 'veins'):
        grouped = [d for d in definitions if d.feature_group == f'bif_{layer}']
        assert len(grouped) == 4
