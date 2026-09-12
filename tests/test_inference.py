from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tests import settings
from tests.regression_helpers import SAMPLES_DIR


@pytest.mark.cli_e2e
def test_inference(tmp_path: Path, run_vascx) -> None:
    """Compare real model inference with stored sample masks and landmarks."""
    input_dir = SAMPLES_DIR / "original"
    output = tmp_path / "segmentations"
    expected_ids = sorted(path.stem for path in input_dir.iterdir() if path.is_file())
    run_vascx("run-models", input_dir, output, "--n-jobs", "1", cwd=tmp_path)

    for dirname in ["preprocessed_rgb", "vessels", "artery_vein", "disc", "overlays"]:
        assert sorted(path.stem for path in (output / dirname).glob("*.png")) == expected_ids

    # Compare each foreground class separately: background must not hide lost vessels.
    # Permit small boundary differences across CPU/GPU and dependency versions.
    for generated, stored in {"vessels": "vessels", "artery_vein": "av", "disc": "discs"}.items():
        for image_id in expected_ids:
            current = np.asarray(Image.open(output / generated / f"{image_id}.png"))
            reference = np.asarray(Image.open(SAMPLES_DIR / stored / f"{image_id}.png"))
            assert current.shape == reference.shape, f"{generated}/{image_id}: shape changed"
            assert set(np.unique(current)) == set(np.unique(reference)), (
                f"{generated}/{image_id}: labels changed"
            )
            for label in np.unique(reference):
                if label == 0:
                    continue
                actual_mask, expected_mask = current == label, reference == label
                dice = 2 * np.count_nonzero(actual_mask & expected_mask) / (
                    np.count_nonzero(actual_mask) + np.count_nonzero(expected_mask)
                )
                assert dice >= settings.SEGMENTATION_MIN_DICE, (
                    f"{generated}/{image_id}, class {label}: Dice={dice:.6f} "
                    f"< {settings.SEGMENTATION_MIN_DICE:g}"
                )

    frames = {}
    for name in ["bounds", "quality", "fovea"]:
        frames[name] = pd.read_csv(output / f"{name}.csv", index_col=0)
        assert sorted(frames[name].index.astype(str)) == expected_ids
        assert frames[name].index.is_unique

    reference_fovea = pd.read_csv(SAMPLES_DIR / "fovea.csv", index_col=0)
    np.testing.assert_allclose(
        frames["fovea"].loc[expected_ids, ["x_fovea", "y_fovea"]],
        reference_fovea.loc[expected_ids, ["mean_x", "mean_y"]],
        atol=settings.FOVEA_ABS_TOL_PIXELS, rtol=0,
        err_msg=f"fovea coordinates changed by more than {settings.FOVEA_ABS_TOL_PIXELS:g} pixels",
    )
    # No stored quality-score baseline exists; require complete, finite logits.
    assert np.isfinite(frames["quality"][["q1", "q2", "q3"]].to_numpy()).all()
    assert frames["bounds"]["bounds"].notna().all()
