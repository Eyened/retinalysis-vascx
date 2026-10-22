"""Regional image measurements preserve FOV/background masks and export contracts."""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from rtnls_enface.grids.hemifields import HemifieldField
from rtnls_enface.grids.specifications import (
    GridFieldSpecification,
    HemifieldGridSpecification,
)
from skimage.color import rgb2lab

from vascx.fundus.features.luminance import Luminance
from vascx.fundus.features.rps import BackgroundLab
from vascx.shared.features import FeatureSet
from vascx.utils.feature_docs import get_biomarker_definitions


def field_spec(field_limit=None, grid_limit=None, field=HemifieldField.Superior):
    return GridFieldSpecification(
        HemifieldGridSpecification(min_area_within_bounds=grid_limit),
        field,
        min_area_within_bounds=field_limit,
    )


@pytest.fixture
def retina():
    # Four of the five requested pixels are visible: exactly 80% coverage.
    visible = np.ones((2, 5), dtype=bool)
    visible[0, 0] = False
    selected = np.zeros_like(visible)
    selected[0] = True
    image = np.full((2, 5, 3), 200, dtype=np.uint8)
    image[0] = 40
    image[0, 0] = 255
    field = SimpleNamespace(
        mask=selected,
        full_pixel_count=lambda: 5,
        plot=lambda ax: ax.axhline(0.5),
    )
    return SimpleNamespace(
        image=image,
        mask=visible,
        background_mask=visible.copy(),
        get_grid_field=lambda spec: field,
        plot=lambda **kwargs: None,
    )


@pytest.mark.parametrize("channels", [None, 1, 3])
def test_luminance_intersects_grid_with_visible_mask(retina, channels):
    if channels is None:
        retina.image = retina.image[..., 0]
    elif channels == 1:
        retina.image = retina.image[..., :1]
    factor = 0.9999 if channels == 3 else 1.0
    original_mask = retina.mask.copy()
    assert Luminance(grid_field=field_spec()).compute(retina) == pytest.approx(40 * factor)
    assert Luminance().compute(retina) == pytest.approx((4 * 40 + 5 * 200) / 9 * factor)
    np.testing.assert_array_equal(retina.mask, original_mask)


def test_background_lab_keeps_exclusions_without_counting_them_as_missing_fov(retina):
    retina.image[:] = [0, 0, 255]
    retina.image[0, 1] = [255, 0, 0]
    retina.background_mask[0] = False
    retina.background_mask[0, 1] = True
    original_mask = retina.background_mask.copy()
    # Only one background pixel remains in the field, but its visibility is 80%.
    expected_red = rgb2lab(np.array([[[1.0, 0.0, 0.0]]]))[0, 0]
    expected_blue = rgb2lab(np.array([[[0.0, 0.0, 1.0]]]))[0, 0]
    np.testing.assert_allclose(BackgroundLab(grid_field=field_spec()).compute(retina), expected_red)
    np.testing.assert_allclose(BackgroundLab().compute(retina), expected_blue)
    np.testing.assert_array_equal(retina.background_mask, original_mask)


@pytest.mark.parametrize("feature_cls", [Luminance, BackgroundLab])
@pytest.mark.parametrize("feature_limit,field_limit,grid_limit,passes", [
    (None, None, None, False),
    (0.5, 0.6, 0.7, True),
    (0.6, 0.5, 0.4, False),
    (None, 0.5, 0.6, True),
    (None, 0.6, 0.5, False),
    (None, None, 0.5, True),
])
def test_grid_visibility_uses_geometric_area_and_threshold_precedence(
    retina, feature_cls, feature_limit, field_limit, grid_limit, passes,
):
    # The complete field extends beyond the image; four of eight pixels are visible.
    retina.get_grid_field(None).full_pixel_count = lambda: 8
    feature = feature_cls(
        grid_field=field_spec(field_limit, grid_limit),
        min_area_within_bounds=feature_limit,
    )
    result = feature.compute(retina)
    if passes:
        assert np.isfinite(result).all()
    elif feature_cls is Luminance:
        assert result is None
    else:
        assert result.shape == (3,) and np.isnan(result).all()


@pytest.mark.parametrize("feature_cls", [Luminance, BackgroundLab])
@pytest.mark.parametrize("missing", ["image", "field", "visible"])
def test_missing_inputs_or_empty_intersection_keep_missing_output_contract(retina, feature_cls, missing):
    if missing == "image":
        retina.image = None
    elif missing == "field":
        retina.get_grid_field(None).mask[:] = False
    else:
        retina.mask[:] = False
    result = feature_cls(grid_field=field_spec(), min_area_within_bounds=0).compute(retina)
    if feature_cls is Luminance:
        assert result is None
    else:
        assert result.shape == (3,) and np.isnan(result).all()


def test_visible_field_without_background_pixels_is_missing(retina):
    retina.background_mask[0] = False
    assert np.isnan(BackgroundLab(grid_field=field_spec()).compute(retina)).all()


@pytest.mark.parametrize("feature_cls", [Luminance, BackgroundLab])
@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, True])
def test_invalid_visibility_threshold_is_rejected(feature_cls, threshold):
    with pytest.raises(ValueError, match="min_area_within_bounds"):
        feature_cls(min_area_within_bounds=threshold)


def test_regional_output_names_and_descriptions_are_distinct(monkeypatch):
    monkeypatch.setattr(FeatureSet, "_registry", dict(FeatureSet._registry))
    features = [
        cls(grid_field=spec)
        for cls in (Luminance, BackgroundLab)
        for spec in (None, field_spec(), field_spec(field=HemifieldField.Inferior))
    ]
    fs = FeatureSet("test_image_regions", features)
    for naming, suffix in (("resolved", ""), ("canonical", "_retina")):
        definitions = get_biomarker_definitions(fs, naming=naming)
        assert len(definitions) == 12
        assert len({d.variable for d in definitions}) == 12
        assert len({d.display_name for d in definitions}) == 12
        assert {"luminance" + suffix, *("background_lab_" + c + suffix for c in "lab")} <= {
            d.variable for d in definitions
        }
        assert sum("hf_superior" in d.variable for d in definitions) == 4
        assert sum("hf_inferior" in d.variable for d in definitions) == 4
    for cls in (Luminance, BackgroundLab):
        feature = cls(grid_field=field_spec(), min_area_within_bounds=0.5)
        assert "min_area_within_bounds_0p5" in feature.canonical_name()
        assert "[Sup]" in feature.display_name()
        assert "Computed in" in feature.description()


@pytest.mark.parametrize("feature_cls", [Luminance, BackgroundLab])
def test_plots_show_only_sampled_pixels_and_overlay_the_grid(retina, feature_cls):
    retina.background_mask[0, 1] = False
    original_visible = retina.mask.copy()
    original_background = retina.background_mask.copy()
    fig, ax = plt.subplots()
    try:
        feature_cls(grid_field=field_spec())._plot(ax, retina)
        fig.canvas.draw()
        shown = ax.images[0].get_array()
        if feature_cls is Luminance:
            assert np.ma.getmaskarray(shown)[1].all()
            assert np.ma.getmaskarray(shown)[0, 0]
            np.testing.assert_allclose(shown[0, 1:], 40 * 0.9999)
        else:
            np.testing.assert_allclose(shown[0, 2:], 40 / 255)
            np.testing.assert_allclose(shown[0, 1], 40 / 255 * 0.15)
            np.testing.assert_allclose(shown[1], 200 / 255 * 0.15)
        assert len(ax.lines) == 1
        np.testing.assert_array_equal(retina.mask, original_visible)
        np.testing.assert_array_equal(retina.background_mask, original_background)
    finally:
        plt.close(fig)
