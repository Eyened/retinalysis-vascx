"""Optic disc visibility gating, anatomy, units, and compound plotting."""
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from rtnls_enface.base import Point
from rtnls_enface.disc_ellipse import OpticDiscEllipseFit

from tests.regression_helpers import SAMPLES_DIR, discover_feature_set_names
from vascx.fundus.features.od_morphology import ODMorphology
from vascx.fundus.features.disc_features import DiscFoveaDistance
from vascx.fundus.features.rps import BackgroundLab
from vascx.fundus.loader import RetinaLoader
from vascx.shared.features import FeatureSet


def synthetic_retina(angle=90., visible=1., fovea=(140., 100.), axes=(28., 20.)):
    fit = OpticDiscEllipseFit(center=Point(100., 80.), semi_axes=axes,
        angle_degrees=angle, contour_xy=np.empty((0, 2)), excluded_contour_xy=np.empty((0, 2)),
        contour_rmse_px=0., observed_arc_degrees=360., visible_fraction=visible,
        visible_iou=1., boundary_margin_px=0., roundness_weight=0., flags=())
    mask = fit.to_mask((200, 200))
    return SimpleNamespace(disc=SimpleNamespace(ellipse=fit, mask=mask),
        fovea_location=None if fovea is None else Point(fovea[1], fovea[0]),
        image=np.zeros((200, 200, 3)), mask=np.ones((200, 200)), resolution=(200, 200))


def values(retina):
    return dict(zip((o.key for o in ODMorphology.outputs), ODMorphology().compute(retina)))


def test_full_disc_geometry_and_units():
    retina = synthetic_retina()
    v = values(retina)
    assert v['valid'] == 1 and v['fraction_outside_bounds'] == 0
    assert (v['centroid_x'], v['centroid_y']) == (80, 100)
    assert v['fovea_distance'] == 60
    assert (v['ellipse_a'], v['ellipse_b']) == (28, 20)
    assert v['ovality_index'] == 1.4
    assert v['torsion'] == 0
    assert v['area'] == retina.disc.mask.sum()
    assert v['area'] == pytest.approx(np.pi*28*20, rel=.02)
    assert v['area_norm'] == pytest.approx(v['area']/3600)
    # Elliptical contour circularity from the analytic perimeter (Ramanujan).
    h = ((28-20)/(28+20))**2
    perimeter = np.pi*48*(1+3*h/(10+np.sqrt(4-3*h)))
    assert v['circularity'] == pytest.approx(4*np.pi**2*560/perimeter**2, rel=.04)


@pytest.mark.parametrize('angle, expected', [(90, 0), (70, 20), (110, 20), (0, 90), (179, 89)])
def test_torsion_and_mirror_symmetry(angle, expected):
    assert values(synthetic_retina(angle=angle))['torsion'] == pytest.approx(expected)
    assert values(synthetic_retina(angle=180-angle, fovea=(20, 100)))['torsion'] == pytest.approx(expected)


def test_torsion_uses_fovea_reference_not_image_vertical():
    # Rotate both the disc and disc–fovea line by 30 degrees.
    fovea = (80+60*np.cos(np.pi/6), 100+60*np.sin(np.pi/6))
    assert values(synthetic_retina(angle=140, fovea=fovea))['torsion'] == pytest.approx(20)


@pytest.mark.parametrize('visible', [.99999, .7, .2])
def test_clipped_disc_retains_position_but_withholds_shape(visible):
    v = ODMorphology().compute(synthetic_retina(visible=visible))
    np.testing.assert_allclose(v[:5], [1, 1-visible, 80, 100, 60])
    assert np.isnan(v[5:]).all()


@pytest.mark.parametrize('ovality,accepted', [(1.499, True), (1.5, False), (1.501, False)])
@pytest.mark.parametrize('visible', [1.0, .2])
def test_ovality_gate_preserves_clipped_distance_below_limit(ovality, accepted, visible):
    v = ODMorphology().compute(synthetic_retina(axes=(20*ovality, 20), visible=visible))
    assert v[0] == int(accepted)
    if not accepted:
        assert np.isnan(v[1:]).all()
    else:
        assert v[4] == 60
        assert np.isnan(v[5:]).all() if visible < 1 else np.isfinite(v[5:]).all()


@pytest.mark.parametrize('fovea', [None, (np.nan, 100), (np.inf, 100), (80, 100)])
def test_missing_or_coincident_fovea_keeps_disc_shape(fovea):
    v = values(synthetic_retina(fovea=fovea))
    assert v['valid'] == 1 and np.isfinite(v['area'])
    assert np.isnan(v['area_norm']) and np.isnan(v['torsion'])
    assert v['fovea_distance'] == 0 if fovea == (80, 100) else np.isnan(v['fovea_distance'])


def test_circle_has_no_unique_torsion_axis():
    v = values(synthetic_retina(axes=(25, 25)))
    assert v['ovality_index'] == 1
    assert v['circularity'] == pytest.approx(1., abs=.035)
    assert np.isnan(v['torsion'])


@pytest.mark.parametrize('missing', [True, False])
def test_invalid_disc_always_outputs_zero_validity(missing):
    retina = synthetic_retina()
    if missing:
        retina.disc = None
    else:
        retina.disc.ellipse.visible_iou = .5
        retina.disc.ellipse.contour_rmse_px = 0  # A good contour fit cannot rescue low overlap.
    v = ODMorphology().compute(retina)
    assert v[0] == 0 and np.isnan(v[1:]).all()


@pytest.mark.parametrize('clipped,missing', [(False, False), (True, False), (False, True)])
def test_plot_contains_every_measurement_over_full_fundus(clipped, missing):
    retina = synthetic_retina(visible=.7 if clipped else 1.)
    if missing:
        retina.disc = None
    feature = ODMorphology()
    fig = feature.plot_figure(retina, figsize=(4, 4))
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        assert not ax.child_axes
        assert len(ax.images) == 1
        np.testing.assert_allclose(ax.get_xlim(), [-.5, 199.5])
        np.testing.assert_allclose(ax.get_ylim(), [199.5, -.5])
        background = next(p for p in ax.patches if p.get_gid() == "od-measurements-background")
        assert background.get_x() == 0 and background.get_width() == 1
        assert 0 < background.get_alpha() < 1
        renderer = fig.canvas.get_renderer()
        overlay_bounds = background.get_window_extent(renderer)
        for text_artist in ax.texts:
            if text_artist.get_zorder() == 11:
                assert overlay_bounds.contains(*text_artist.get_window_extent(renderer).get_points()[0])
                assert overlay_bounds.contains(*text_artist.get_window_extent(renderer).get_points()[1])
        text = '\n'.join(t.get_text() for t in ax.texts)
        for label in ('Valid', 'Outside bounds', 'Centroid x', 'Centroid y', 'Fovea distance',
                      'Semi-major a', 'Semi-minor b', 'Ovality a/b', 'Circularity',
                      'Area', 'Area / distance²', 'Torsion'):
            assert label in text
        assert ('position estimated' in text) == clipped
        assert ('No valid optic disc' in text) == missing
    finally:
        plt.close(fig)


def test_all_applicable_feature_sets_use_one_compound_disc_feature():
    expected = {'full', 'full_v2', 'full_v3', 'macula_centered', 'macula_centered_rs',
                'od_centered', 'od_centered_rs', 'od_centered_narrow_rs'}
    seen = set()
    for name in discover_feature_set_names():
        fs = FeatureSet.get_by_name(name)
        assert not any(isinstance(f, DiscFoveaDistance) for f in fs)
        if any(isinstance(f, ODMorphology) for f in fs):
            assert sum(isinstance(f, ODMorphology) for f in fs) == 1
            seen.add(name)
    assert seen == expected


def test_sample_visibility_and_lab_values_match_previous_exports():
    # The Lab vector replaces three independent computations with identical values.
    reference = pd.read_parquet(SAMPLES_DIR.parents[1]/'tests/reference/macula_centered.parquet')
    loader = RetinaLoader.from_folder(SAMPLES_DIR)
    visible_counts = [0, 0]
    for image_id in loader.items:
        retina = loader.make_object(image_id)
        v = ODMorphology().compute(retina)
        assert v[0] == 1
        clipped = v[1] > 0
        visible_counts[int(clipped)] += 1
        assert np.isnan(v[5:]).all() if clipped else np.isfinite(v).all()
        expected = reference.loc[image_id, ['background_lab_l_retina', 'background_lab_a_retina', 'background_lab_b_retina']].to_numpy(dtype=float)
        np.testing.assert_allclose(BackgroundLab().compute(retina), expected, rtol=1e-12, atol=1e-12)
    assert all(n > 0 for n in visible_counts)


@pytest.mark.parametrize('fovea_row', [None, (np.nan, np.nan), (100., 200.)])
def test_cli_keeps_images_without_disc_or_fovea(tmp_path, fovea_row):
    from vascx.cli import make_examples
    (tmp_path/'artery_vein').mkdir()
    (tmp_path/'artery_vein/sample.png').touch()
    pd.DataFrame({'bounds': ['[]']}, index=['sample']).to_csv(tmp_path/'bounds.csv')
    if fovea_row is not None:
        pd.DataFrame([fovea_row], columns=['x_fovea', 'y_fovea'], index=['sample']).to_csv(tmp_path/'fovea.csv')
    examples = make_examples(tmp_path)
    assert len(examples) == 1 and examples[0]['disc_path'] is None
    expected = None if fovea_row is None or not np.isfinite(fovea_row).all() else fovea_row
    if fovea_row is not None and not np.isfinite(fovea_row).all():
        np.testing.assert_allclose(examples[0]['fovea_location'], fovea_row, equal_nan=True)
    else:
        assert examples[0]['fovea_location'] == expected
