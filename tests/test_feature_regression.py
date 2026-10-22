from __future__ import annotations

import pandas as pd
import pytest

from tests.regression_helpers import (
    assert_matches_reference,
    discover_feature_set_names,
    load_reference_frame,
    load_regression_config,
    write_reference_artifacts,
)

FEATURE_SET_NAMES = discover_feature_set_names()


@pytest.mark.reference
@pytest.mark.parametrize("feature_set_name", FEATURE_SET_NAMES, ids=FEATURE_SET_NAMES)
def test_feature_set_regression(
    feature_set_name: str, accept_vascx_reference: bool,
    run_vascx, biomarker_cli_input, tmp_path, pytestconfig,
) -> None:
    """Compare extracted biomarker outputs against stored references."""

    output = tmp_path / "report"
    run_vascx(
        "calc-biomarkers", biomarker_cli_input, output,
        "--feature-set", feature_set_name, "--n-jobs", "1",
        "--naming", "canonical", "--no-report", cwd=tmp_path,
    )
    for name in ["biomarkers.csv", "data_dictionary.csv"]:
        assert (output / name).is_file(), f"missing {name}"
    current = pd.read_csv(output / "biomarkers.csv", index_col=0)
    # Crop metadata is tested separately; reference tables describe biomarkers only.
    assert current["crop_scale_factor"].gt(0).all()
    current = current.drop(columns="crop_scale_factor")
    if accept_vascx_reference:
        write_reference_artifacts(feature_set_name, current)

    reference = load_reference_frame(feature_set_name)
    config = load_regression_config(
        feature_set_name,
        max_percent_change=pytestconfig.getoption("--vascx-max-percent-change"),
    )
    assert_matches_reference(feature_set_name, current, reference, config)
