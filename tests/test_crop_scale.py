"""Crop metadata follows each image and calibrates spatial outputs exactly once."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tests.test_units_contract import Measured, fs, retina
from vascx.fundus.loader import RetinaLoader
from vascx.shared.inputs import InputContractError
from vascx.shared.units import crop_scale_from_bounds
from vascx.utils.analysis import _load_retina_input, extract_one, extract_biomarkers_to_folder


def bounds(radius):
    return {"hw": (2048, 2048), "center": (1024., 1024.), "radius": radius, "lines": {}}


@pytest.mark.parametrize("radius", [256., 512., 1024.])
def test_crop_factor_inverts_actual_preprocessing_transform(radius):
    from rtnls_fundusprep.cfi_bounds import CFIBounds
    original = CFIBounds(**bounds(radius))
    for diameter in [512, 1024]:
        expected = 1 / original.get_cropping_transform(diameter).scale
        assert crop_scale_from_bounds(original, diameter) == pytest.approx(expected)
        assert crop_scale_from_bounds(original.to_dict(), diameter) == pytest.approx(expected)


def test_loaders_preserve_scale_before_bounds_become_roi():
    item = {"id": "sample", "bounds": bounds(1024), "mm_per_pixel": .01,
            "fovea_location": (500., 500.)}
    loader = RetinaLoader({"sample": item})
    from_loader = loader.by_id("sample")
    from_mapping = _load_retina_input(loader.to_dict()[0], use_mm_per_pixel=True)
    assert from_loader.crop_scale_factor == from_mapping.crop_scale_factor == 2
    assert from_loader.mm_per_pixel == from_mapping.mm_per_pixel == .01
    assert from_mapping.processed_mm_per_pixel == .02
    np.testing.assert_array_equal(from_mapping.roi_mask, from_loader.roi_mask)
    assert loader.items["sample"]["bounds"] == bounds(1024)
    assert loader.get_item("sample")["bounds"] is None
    values, _ = extract_one(item, fs(), use_mm_per_pixel=True)
    assert values["measured_length"] == pytest.approx(.2)
    assert values["crop_scale_factor"] == 2


@pytest.mark.parametrize("factor", [.5, 2.])
def test_original_calibration_controls_values_grids_and_plot_labels(factor):
    image = retina(crop=factor)
    features = fs()
    native = image.calc_features(features, structured=True)["measured"]
    scale = .02 * factor
    expected = [10 * scale, 100 * scale ** 2, .2 / scale, .5, np.nan]
    for _ in range(2):
        actual = image.calc_features(features, structured=True, use_mm_per_pixel=True)["measured"]
        np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(native, [10, 100, .2, .5, np.nan])
    assert image._grid_resolution() == pytest.approx(scale)
    assert image.mm_per_pixel == .02 and image.crop_scale_factor == factor
    fig, axes = plt.subplots(1, 2)
    try:
        Measured().plot(axes[0], image, use_mm_per_pixel=True, plot_caption=False)
        Measured().plot(axes[1], image, use_mm_per_pixel=True, plot_caption=False, computed_value=actual)
        assert axes[0].texts[-1].get_text() == axes[1].texts[-1].get_text()
        assert f'{10 * scale:g} mm' in axes[0].texts[-1].get_text()
        assert axes[0]._vascx_display_scale == scale
        np.testing.assert_array_equal(axes[0].lines[0].get_xdata(), [0, 10])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("factor", [None, 0, -1, np.nan, np.inf, True, 'bad'])
def test_enabled_calibration_requires_known_crop_scale(factor):
    with pytest.raises(InputContractError, match="crop_scale_factor"):
        retina(crop=factor).calc_features(fs(), use_mm_per_pixel=True)
    # Original-image calibration and crop metadata never change native measurements.
    assert retina(crop=factor).calc_features(fs())["measured_length"] == 10


@pytest.mark.parametrize("calibrated", [False, True])
def test_export_metadata_and_readme(tmp_path, calibrated):
    inputs = [retina("large", crop=2), retina("bad", fovea=None, crop=3), retina("small", crop=.5)]
    frame = extract_biomarkers_to_folder(inputs, fs(), tmp_path, n_jobs=1,
                                         use_mm_per_pixel=calibrated)
    exported = pd.read_csv(tmp_path / "biomarkers.csv", index_col="id")
    assert list(exported.index) == ["large", "small"]
    assert list(exported.crop_scale_factor) == list(frame.crop_scale_factor) == [2, .5]
    np.testing.assert_allclose(exported.measured_length, [.4, .1] if calibrated else [10, 10])
    dictionary = pd.read_csv(tmp_path / "data_dictionary.csv").set_index("variable")
    assert set(dictionary.index) == set(exported.columns)
    assert dictionary.loc["crop_scale_factor", "units"] == "dimensionless"
    assert "original-input pixels per processed-image pixel" in dictionary.loc["crop_scale_factor", "description"]
    readme = (tmp_path / "README.md").read_text()
    assert "`crop_scale_factor`" in readme and "Number of biomarkers: 5" in readme
    manifest = json.loads((tmp_path / "report.json").read_text())
    assert manifest["calibration_pixel_space"] == ("original" if calibrated else None)


def test_missing_bounds_export_unknown_scale_without_changing_native_values(tmp_path):
    item = {"id": "unknown", "fovea_location": (10., 20.)}
    frame = extract_biomarkers_to_folder([item], fs(), tmp_path, n_jobs=1, generate_report=False)
    assert pd.isna(frame.loc["unknown", "crop_scale_factor"])
    assert frame.loc["unknown", "measured_length"] == 10
    assert pd.isna(pd.read_csv(tmp_path / "biomarkers.csv").loc[0, "crop_scale_factor"])
    with pytest.raises(InputContractError, match="crop_scale_factor"):
        extract_one({**item, "mm_per_pixel": .01}, fs(), use_mm_per_pixel=True)


def test_cli_calibration_uses_original_pixels(biomarker_cli_input, tmp_path):
    from click.testing import CliRunner
    from vascx.cli import cli
    from vascx.fundus.features.od_morphology import ODMorphology
    from vascx.shared.units import spatial_power

    features = fs(ODMorphology())
    runner = CliRunner()
    native_folder, physical_folder = tmp_path / 'native', tmp_path / 'physical'
    args = ['calc-biomarkers', str(biomarker_cli_input), str(native_folder),
            '--feature-set', features.name, '--sample', '2', '--n-jobs', '1', '--no-report']
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    native = pd.read_csv(native_folder / 'biomarkers.csv', index_col='id')
    assert native.crop_scale_factor.notna().all()
    assert native.crop_scale_factor.gt(0).all()
    assert native.crop_scale_factor.ne(1).all()
    calibration = pd.Series([.005, .006], index=native.index, name='mm_per_pixel')
    calibration.to_csv(tmp_path / 'calibration.csv')
    args[2] = str(physical_folder)
    result = runner.invoke(cli, args + ['--use-mm-per-pixel', '--mm-per-pixel-csv', str(tmp_path / 'calibration.csv')])
    assert result.exit_code == 0, result.output
    physical = pd.read_csv(physical_folder / 'biomarkers.csv', index_col='id')
    units = pd.read_csv(native_folder / 'data_dictionary.csv').set_index('variable').units
    for column in native:
        expected = native[column] * (native.crop_scale_factor * calibration) ** spatial_power(units[column])
        np.testing.assert_allclose(physical[column], expected, equal_nan=True)
    assert physical.od_area.gt(0).all()
    help_text = runner.invoke(cli, ['calc-biomarkers', '--help']).output
    assert 'original image supplied to VascX' in ' '.join(help_text.split())
