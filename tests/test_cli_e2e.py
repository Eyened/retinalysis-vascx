from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from tests.regression_helpers import SAMPLES_DIR
from vascx.cli import cli


@pytest.mark.cli_e2e
def test_cli_readme_flow_run_models_then_calc_biomarkers(tmp_path: Path) -> None:
    """Run the README CLI flow from original images through biomarker CSV output."""

    input_dir = SAMPLES_DIR / "original"
    segmentations_dir = tmp_path / "segmentations"
    biomarkers_csv = tmp_path / "biomarkers.csv"
    expected_ids = sorted(path.stem for path in input_dir.iterdir() if path.is_file())

    runner = CliRunner()
    run_models_result = runner.invoke(
        cli,
        [
            "run-models",
            str(input_dir),
            str(segmentations_dir),
        ],
    )
    assert run_models_result.exit_code == 0, run_models_result.output

    for dirname in ["preprocessed_rgb", "vessels", "artery_vein", "disc", "overlays"]:
        output_dir = segmentations_dir / dirname
        assert output_dir.is_dir(), f"missing {output_dir}"
        output_ids = sorted(path.stem for path in output_dir.glob("*.png"))
        assert output_ids == expected_ids

    for csv_name in ["bounds.csv", "quality.csv", "fovea.csv"]:
        csv_path = segmentations_dir / csv_name
        assert csv_path.is_file(), f"missing {csv_path}"
        assert len(pd.read_csv(csv_path)) == len(expected_ids)

    calc_result = runner.invoke(
        cli,
        [
            "calc-biomarkers",
            str(segmentations_dir),
            str(biomarkers_csv),
            "--feature_set",
            "full_v3",
            "--n-jobs",
            "1",
        ],
    )
    assert calc_result.exit_code == 0, calc_result.output

    biomarkers = pd.read_csv(biomarkers_csv, index_col=0)
    assert sorted(biomarkers.index.astype(str)) == expected_ids
    assert biomarkers.shape[1] > 0
