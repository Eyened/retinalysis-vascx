"""Numerical unit conversion and per-image contracts at real output boundaries."""
import json
from uuid import uuid4
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vascx.fundus.retina import Retina
from vascx.fundus.features.base import RetinaFeature
from vascx.fundus.features.od_morphology import ODMorphology
from vascx.fundus.features.sparsity import Sparsity
from vascx.fundus.features.tortuosity import Tortuosity, TortuosityMeasure
from vascx.shared.features import FeatureOutput, FeatureSet
from vascx.shared.inputs import InputContractError, prepare_retina
from vascx.utils.analysis import extract_one, extract_in_parallel, extract_biomarkers_to_folder
from vascx.utils.feature_docs import get_biomarker_definitions


class Measured(RetinaFeature):
    outputs = (FeatureOutput('length', units='px'), FeatureOutput('area', units='px²'),
               FeatureOutput('curvature', units='px⁻¹'), FeatureOutput('ratio'),
               FeatureOutput('missing', units='px'))

    def feature_name_tokens(self):
        return ['measured']

    def display_name(self, **kwargs):
        return 'Measurements'

    def compute(self, retina):
        return np.array([10., 100., .2, .5, np.nan])

    def _plot(self, ax, retina, **kwargs):
        ax.plot([0, 10], [0, 10])
        return ax


def fs(*features):
    return FeatureSet('units_' + uuid4().hex, features or [Measured()])


def retina(identifier='a', mm=.02, fovea=(15., 16.)):
    return Retina(layers={}, resolution=(32, 32), id=identifier,
                  mm_per_pixel=mm, fovea_location=fovea)


def test_mixed_vector_and_dictionary_scale_once():
    image, features = retina(), fs()
    native = image.calc_features(features, structured=True)['measured']
    physical = image.calc_features(features, structured=True, use_mm_per_pixel=True)['measured']
    np.testing.assert_allclose(native, [10, 100, .2, .5, np.nan])
    np.testing.assert_allclose(physical, [.2, .04, 10., .5, np.nan])
    assert image.mm_per_pixel == .02
    assert [d.units for d in get_biomarker_definitions(features, use_mm_per_pixel=True)] == ['mm', 'mm²', 'mm⁻¹', 'dimensionless', 'mm']
    assert [d.variable for d in get_biomarker_definitions(features)] == [d.variable for d in get_biomarker_definitions(features, use_mm_per_pixel=True)]


@pytest.mark.parametrize('calibration', [None, 0, -1, float('nan'), float('inf'), [1], True, 'bad', pytest.param(10**1000, id='overflow')])
def test_calibration_contract_checked_before_loading(calibration):
    with pytest.raises(InputContractError, match='mm_per_pixel'):
        extract_one({'id': 'bad', 'mm_per_pixel': calibration, 'fovea_location': (1, 2)}, fs(), use_mm_per_pixel=True)


@pytest.mark.parametrize('fovea', [None, [1], [1, 2, 3], [np.nan, 1], [1, np.inf], pytest.param([10**1000, 1], id="overflow")])
def test_fovea_contract_before_loading(fovea):
    with pytest.raises(InputContractError, match='fovea'):
        extract_one({'id': 'bad', 'fovea_location': fovea}, fs())


@pytest.mark.parametrize('jobs', [1, 2])
def test_bulk_skips_preserving_identity_and_report_counts(tmp_path, jobs):
    inputs = [retina('a'), retina('bad', fovea=None), retina('c', mm=.04)]
    frame = extract_in_parallel(inputs, fs(), n_jobs=jobs, use_mm_per_pixel=True)
    assert list(frame.index) == ['a', 'c']
    assert list(frame.attrs['input_contract_failures']) == [1]
    np.testing.assert_allclose(frame.measured_length, [.2, .4])
    output = tmp_path / 'report'
    frame = extract_biomarkers_to_folder(inputs, fs(), output, n_jobs=jobs, use_mm_per_pixel=True)
    report = json.loads((output / 'report.json').read_text())
    assert report['dataset']['image_count'] == 2
    assert report['dataset']['report_sample_ids'] == ['a', 'c']
    assert report['inputs'] == {'use_fovea': True, 'use_mm_per_pixel': True}
    assert pd.read_csv(output / 'data_dictionary.csv').units.tolist() == ['mm', 'mm²', 'mm⁻¹', 'dimensionless', 'mm']


def test_disabled_values_cannot_influence_results_or_mutate_original():
    original = retina(fovea=(float('nan'), 1))
    prepared = prepare_retina(original, use_fovea=False)
    assert prepared.fovea_location is None and prepared.mm_per_pixel is None
    assert original.fovea_location is not None and original.mm_per_pixel == .02
    frame = original.calc_features(fs(ODMorphology()), use_fovea=False)
    assert frame['od_valid'] == 0 and all(pd.isna(v) for k, v in frame.items() if k != 'od_valid')


def test_single_failure_propagates_and_all_failed_folder_fails(tmp_path):
    with pytest.raises(InputContractError):
        retina(fovea=None).calc_features(fs())
    with pytest.raises(ValueError, match='No images'):
        extract_biomarkers_to_folder([retina(fovea=None)], fs(), tmp_path, n_jobs=1)
    assert not (tmp_path / 'report.json').exists()


def test_native_sparsity_curvature_and_fractional_split_requirement():
    layer = SimpleNamespace(distance_transform=np.array([[2., 4.], [6., 8.]]),
                            retina=SimpleNamespace(sparsity_fovea_mask=np.zeros((2, 2), bool)))
    assert Sparsity().compute(layer) == 5
    segment = SimpleNamespace(get_spline=lambda **kw: SimpleNamespace(curvatures=lambda: np.array([.1, .3])))
    assert Tortuosity(measure=TortuosityMeasure.Curvature)._compute_for_segment(segment) == pytest.approx(.2)
    with pytest.raises(ValueError, match='Disc-fovea distance'):
        Tortuosity(max_segment_len=.2)._resolved_max_segment_len_pixels(layer)


def test_plot_converts_annotations_without_moving_geometry():
    fig, ax = plt.subplots()
    Measured().plot(ax, retina(), use_mm_per_pixel=True, plot_caption=False)
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 10])
    assert '0.2 mm' in ax.texts[-1].get_text() and '10 mm⁻¹' in ax.texts[-1].get_text()
    plt.close(fig)


def test_native_dependency_cache_and_scalar_conversion():
    from tests.test_derived_features import ExampleRetina, Measurement
    from vascx.fundus.features.derived import AVR
    source = Measurement()
    source.outputs = (FeatureOutput(units='px'),)
    image = ExampleRetina()
    image.mm_per_pixel = .01
    result = image.calc_features(fs(source, AVR(source)), use_mm_per_pixel=True)
    assert result['measurement_arteries'] == .02
    assert result['measurement_veins'] == .04
    assert result['avr_measurement'] == .5
    assert source.calls == ['arteries', 'veins']


def test_fovea_disabled_preserves_disc_shape_but_not_dependent_measurements():
    from vascx.fundus.loader import RetinaLoader
    from vascx.fundus.features.caliber import Caliber
    from vascx.fundus.feature_sets.macula_centered import fs_macula_centered
    field = next(f.grid_field_spec for f in fs_macula_centered if isinstance(f, Caliber))
    from tests.regression_helpers import SAMPLES_DIR
    image = next(r for r in RetinaLoader.from_folder(SAMPLES_DIR)
                 if r.disc is not None and ODMorphology().compute(r)[1] == 0)
    image.scaling_factor = .01
    features = fs(ODMorphology(), Caliber(grid_field=field),
                  Tortuosity(max_segment_len=.2))
    with pytest.warns(UserWarning):
        result = image.calc_features(features, use_mm_per_pixel=True, use_fovea=False)
    assert result['od_valid'] == 1 and result['od_area'] > 0
    assert result['od_ellipse_a'] > 0
    for column in ('od_fovea_distance', 'od_area_norm', 'od_torsion'):
        assert pd.isna(result[column])
    assert all(pd.isna(value) for key, value in result.items() if not key.startswith('od_'))
    assert image.fovea_location is not None


def test_outside_image_landmark_is_valid():
    result = retina(fovea=(-10., 1000.)).calc_features(fs(), use_mm_per_pixel=True)
    assert result['measured_length'] == pytest.approx(.2)


def test_brightness_normalization_changes_only_intensity_units():
    from vascx.fundus.features.edge_strength import EdgeStrength
    for beta, units in [(0, "intensity"), (.5, "intensity^0.5"), (1, "dimensionless")]:
        feature = EdgeStrength(beta=beta)
        assert feature.reported_schema(True)[0].units == units
        assert feature.convert_output(7., .02) == 7.
