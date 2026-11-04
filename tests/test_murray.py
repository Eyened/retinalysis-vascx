"""MED arithmetic, region geometry, and integration with VascX outputs."""
from types import SimpleNamespace
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from rtnls_enface.base import Circle, Point, Laterality
from rtnls_enface.grids.circle import CircleField
from rtnls_enface.grids.disc_centered import (
    DiscCenteredHemifield, DiscCenteredQuadrant, DiscCenteredRing,
)
from rtnls_enface.grids.specifications import (
    CircleGridSpecification, DiscCenteredGridSpecification, GridFieldSpecification,
)

from vascx.fundus.features.murray import MurrayEquivalentDiameter
from vascx.shared.features import FeatureSet
from vascx.utils.feature_docs import get_biomarker_definitions


class Vessel:
    def __init__(self, diameter, angle=-np.pi / 2):
        self.diameter = diameter
        self.angle = angle

    def get_spline(self, error_fraction):
        return self

    def get_point(self, point):
        return point

    def get_median_diameter(self, error_fraction):
        return self.diameter


def layer(diameters=(3., 4.), *, center=(128, 128), fovea=None, shape=(256, 256)):
    center = Point(*center)
    if fovea is None:
        fovea = Point(center.y, center.x + 100)
    retina = SimpleNamespace(
        disc=SimpleNamespace(center=center, circle=Circle(center=center, r=20)),
        fovea_location=fovea, resolution=shape, mask=np.ones(shape, dtype=bool),
        disc_fovea_distance=center.distance_to(fovea),
        laterality=Laterality.L if fovea.x > center.x else Laterality.R,
    )
    retina.get_grid = lambda spec: spec.grid_cls(retina, resolution=1., mask=retina.mask,
                                                **spec.init_kwargs())
    retina.get_grid_field = lambda spec: retina.get_grid(spec.grid_spec).field(spec.field)
    vessels = [Vessel(d) for d in diameters]
    result = SimpleNamespace(retina=retina, name="arteries", vessels=vessels)
    result.get_circle_intersections = lambda circle, error: [
        (v, (center.y + circle.r * np.sin(v.angle), center.x + circle.r * np.cos(v.angle)))
        for v in result.vessels]
    return result


def region(field, grid=None):
    return GridFieldSpecification(grid or DiscCenteredGridSpecification(), field)


@pytest.mark.parametrize("power", [1., 1.25, 1.5, 2., 2.8, 3., 4.])
@pytest.mark.parametrize("layer_name", ["arteries", "veins"])
def test_power_sum_scaling_and_single_vessel(power, layer_name):
    feature = MurrayEquivalentDiameter(**{f"p_{layer_name}": power})
    def target(diameters=(3., 4.)):
        result = layer(diameters)
        result.name = layer_name
        return result
    expected = (3**power + 4**power)**(1 / power)
    assert feature.compute(target()) == pytest.approx(expected)
    assert feature.compute(target((9., 12.))) == pytest.approx(3 * expected)
    assert feature.compute(target((7.,))) == pytest.approx(7)
    assert feature.convert_output(expected, .02) == pytest.approx(.02 * expected)


@pytest.mark.parametrize("layer_name,power", [("arteries", 1.5), ("veins", 1.25)])
def test_default_powers_and_independent_configuration(layer_name, power):
    target = layer()
    target.name = layer_name
    default = MurrayEquivalentDiameter()
    assert default.compute(target) == pytest.approx((3**power + 4**power)**(1 / power))
    other_layer = "veins" if layer_name == "arteries" else "arteries"
    customized = MurrayEquivalentDiameter(**{f"p_{other_layer}": 3.0})
    assert customized.compute(target) == pytest.approx(default.compute(target))
    assert customized.canonical_name(layer_name=layer_name) == default.canonical_name(layer_name=layer_name)
    assert f"power={power:g}" in customized.implementation_description(layer_name=layer_name)
    assert "power_3" in customized.canonical_name(layer_name=other_layer)
    assert "power=3" in customized.implementation_description(layer_name=other_layer)


def test_parent_and_daughters_give_same_equivalent():
    power = 2.8
    parent = (5**power + 8**power)**(1 / power)
    feature = MurrayEquivalentDiameter(p_arteries=power)
    assert feature.compute(layer((parent,))) == pytest.approx(feature.compute(layer((5, 8))))


def test_median_is_taken_before_root_with_even_number_of_arcs():
    target = layer()
    original = target.get_circle_intersections
    def intersections(circle, error):
        index = round((circle.r - 40) / 5)
        if index == 4:
            return []
        target.vessels = [Vessel([2, 4, 8, 16][index])]
        return original(circle, error)
    target.get_circle_intersections = intersections
    assert MurrayEquivalentDiameter(p_arteries=3).compute(target) == pytest.approx(((4**3 + 8**3) / 2)**(1 / 3))


def test_all_vessels_contribute_and_repeated_crossings_are_deduplicated():
    target = layer(tuple(range(1, 10)))
    original = target.get_circle_intersections
    target.get_circle_intersections = lambda circle, error: original(circle, error) * 2
    assert MurrayEquivalentDiameter(p_arteries=3).compute(target) == pytest.approx(sum(d**3 for d in range(1, 10))**(1 / 3))


def test_hemifields_select_crossings_without_clipping_whole_segment_width():
    target = layer()
    target.vessels[1].angle = np.pi / 2
    superior = MurrayEquivalentDiameter(region=region(DiscCenteredHemifield.Superior))
    inferior = MurrayEquivalentDiameter(region=region(DiscCenteredHemifield.Inferior))
    assert superior.compute(target) == pytest.approx(3)
    assert inferior.compute(target) == pytest.approx(4)


@pytest.mark.parametrize("field,angle", [
    (DiscCenteredQuadrant.Superior, -np.pi/2),
    (DiscCenteredQuadrant.Inferior, np.pi/2),
    (DiscCenteredQuadrant.Temporal, 0),
    (DiscCenteredQuadrant.Nasal, np.pi),
    (DiscCenteredQuadrant.SuperiorTemporal, -np.pi/4),
    (DiscCenteredQuadrant.SuperiorNasal, -3*np.pi/4),
    (DiscCenteredQuadrant.InferiorTemporal, np.pi/4),
    (DiscCenteredQuadrant.InferiorNasal, 3*np.pi/4),
])
def test_both_quadrant_conventions(field, angle):
    target = layer((6, 9))
    target.vessels[0].angle = angle
    target.vessels[1].angle = angle + np.pi
    feature = MurrayEquivalentDiameter(region=region(field))
    assert feature.compute(target) == pytest.approx(6)


@pytest.mark.parametrize("angle", [0., .3, -.3])
def test_anatomical_quadrants_rotate_and_mirror_with_landmarks(angle):
    for direction in (1, -1):
        fovea = Point(128 + 100*np.sin(angle), 128 + direction*100*np.cos(angle))
        target = layer((6, 9), fovea=fovea)
        grid = target.retina.get_grid(DiscCenteredGridSpecification())
        masks = [grid.field(q).mask for q in (
            DiscCenteredQuadrant.SuperiorTemporal, DiscCenteredQuadrant.SuperiorNasal,
            DiscCenteredQuadrant.InferiorTemporal, DiscCenteredQuadrant.InferiorNasal)]
        np.testing.assert_array_equal(np.sum(masks, axis=0), grid.grid.astype(int))
        target.vessels[0].angle = -np.pi/4 if direction == 1 else -3*np.pi/4
        target.vessels[1].angle = np.pi/4 if direction == 1 else 3*np.pi/4
        feature = MurrayEquivalentDiameter(region=region(DiscCenteredQuadrant.SuperiorTemporal))
        assert feature.compute(target) == pytest.approx(6)


def test_other_grid_type_works_without_region_specific_measurement_logic():
    feature = MurrayEquivalentDiameter(p_arteries=3, region=region(
        CircleField.FullGrid, CircleGridSpecification(center=0)))
    assert feature.compute(layer()) == pytest.approx(91**(1/3))


@pytest.mark.parametrize("spec", [
    region(CircleField.FullGrid, CircleGridSpecification(center=0, radius_multiplier=.2)),
    region(DiscCenteredRing.Outer),
])
def test_radially_incomplete_region_is_missing_with_warning(spec):
    with pytest.warns(UserWarning, match="radially contain"):
        assert MurrayEquivalentDiameter(region=spec).compute(layer()) is None


@pytest.mark.parametrize("distance,expected_count", [(52, 3), (47, 2), (42, 1)])
def test_image_edge_clipping_keeps_only_complete_arcs(distance, expected_count):
    target = layer(center=(128, distance))
    feature = MurrayEquivalentDiameter(p_arteries=3)
    measurements = feature._measurements(target)
    assert sum(m.power_sum is not None for m in measurements) == expected_count
    if expected_count >= 2:
        assert feature.compute(target) == pytest.approx(91**(1/3))
    else:
        assert feature.compute(target) is None


def test_only_selected_arc_needs_to_be_visible():
    target = layer(center=(52, 128))
    target.vessels[0].angle = np.pi/2
    target.vessels[1].angle = np.pi/2
    feature = MurrayEquivalentDiameter(region=region(DiscCenteredHemifield.Inferior))
    assert all(m.power_sum is not None for m in feature._measurements(target))
    target.retina.mask[107:110, :] = False
    assert sum(m.power_sum is not None for m in feature._measurements(target)) == 3


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, True, "3", None,
                                     pytest.param(10**1000, id="overflow")])
@pytest.mark.parametrize("parameter", ["p_arteries", "p_veins"])
def test_invalid_power(bad, parameter):
    with pytest.raises(ValueError, match=parameter):
        MurrayEquivalentDiameter(**{parameter: bad})


@pytest.mark.parametrize("diameters", [(), (3, np.nan), (3, 0), (3, -1), (3, np.inf)])
def test_empty_or_invalid_selected_diameter_does_not_undercount(diameters):
    assert MurrayEquivalentDiameter().compute(layer(diameters)) is None


@pytest.mark.parametrize("missing", ["disc", "fovea_location"])
def test_missing_landmark(missing):
    target = layer()
    setattr(target.retina, missing, None)
    assert MurrayEquivalentDiameter().compute(target) is None


def test_names_distinguish_quadrants_hemifields_and_powers():
    features = [MurrayEquivalentDiameter(p_arteries=a, p_veins=v, region=region(f))
                for a, v in ((1.5, 1.25), (2.8, 2.8)) for f in (
                    DiscCenteredHemifield.Superior, DiscCenteredQuadrant.Superior,
                    DiscCenteredQuadrant.SuperiorTemporal)]
    fs = FeatureSet('med_' + uuid4().hex, features)
    names = fs.names('canonical').by_name
    assert len(names) == 12
    assert 'med_disc_superior_arteries' in names
    assert 'med_disc_quadrant_superior_arteries' in names
    assert 'med_power_2p8_disc_quadrant_superior_temporal_veins' in names
    definitions = get_biomarker_definitions(fs)
    assert all(d.units == 'px' and 'power=' in d.description for d in definitions)
    assert all(d.units == 'mm' for d in get_biomarker_definitions(fs, use_mm_per_pixel=True))


def test_hemifields_sets_add_six_med_outputs_each():
    from vascx.fundus.feature_sets.hemifields import fs_hemifields, fs_hemifields_rs
    for fs in (fs_hemifields, fs_hemifields_rs):
        med = [f for f in fs if isinstance(f, MurrayEquivalentDiameter)]
        assert len(med) == 3
        assert all(f.p_arteries == 1.5 and f.p_veins == 1.25 for f in med)
        assert sum(n.startswith('med_') for n in fs.names().by_name) == 6
        assert {n for n in fs.names('canonical').by_name if n.startswith('med_')} == {
            f'med_disc_{field}_{layer_name}'
            for layer_name in ("arteries", "veins")
            for field in ("full", "superior", "inferior")
        }
        assert all(f.region.grid_spec == DiscCenteredGridSpecification() for f in med)
        definitions = [d for d in get_biomarker_definitions(fs) if d.variable.startswith('med_')]
        assert len(definitions) == 6
        for definition in definitions:
            power = 1.5 if definition.target == 'arteries' else 1.25
            assert f'power={power:g}' in definition.description


def test_med_extraction_respects_vessel_specific_powers():
    from tests.regression_helpers import SAMPLES_DIR
    from vascx.fundus.feature_sets.hemifields import fs_hemifields
    from vascx.fundus.loader import RetinaLoader
    retina = RetinaLoader.from_folder(SAMPLES_DIR)[0]
    features = [f for f in fs_hemifields if isinstance(f, MurrayEquivalentDiameter)]
    fs = FeatureSet('med_targets_' + uuid4().hex, features)
    result = retina.calc_features(fs, naming='canonical', raise_on_error=True)
    assert set(result) == set(fs.names('canonical').by_name)
    assert len(result) == 6
    assert any(np.isfinite(value) for value in result.values())
    for feature in features:
        for target_name in feature.target_names:
            expected = feature.compute(retina.layers[target_name])
            value = result[feature.canonical_name(layer_name=target_name)]
            assert np.isnan(value) if expected is None else value == pytest.approx(expected)


def test_explanation_plot_on_packaged_image(tmp_path):
    from tests.regression_helpers import SAMPLES_DIR
    from vascx.fundus.loader import RetinaLoader
    retina = RetinaLoader.from_folder(SAMPLES_DIR)[0]
    feature = MurrayEquivalentDiameter(region=region(DiscCenteredHemifield.Superior))
    fig = feature.plot_figure(retina.arteries)
    try:
        assert fig.axes[0].images and fig.axes[0].lines
        fig.savefig(tmp_path/'med.png')
    finally:
        plt.close(fig)
