from pathlib import Path

import pytest

from tests.regression_helpers import SAMPLES_DIR
from vascx.fundus.loader import RetinaLoader
from vascx.utils.analysis import extract_biomarkers_to_folder


@pytest.mark.parametrize("feature_set_name", ["macula_centered", "od_centered"])
def test_biomarker_report_smoke(feature_set_name: str, tmp_path: Path) -> None:
    """Generate a real report from sample inputs and check its main artifacts."""
    output = tmp_path / feature_set_name
    frame = extract_biomarkers_to_folder(
        RetinaLoader.from_folder(SAMPLES_DIR).to_dict(),
        feature_set_name,
        output,
        n_jobs=1,
        naming="canonical",
        report_sample_size=2,
    )
    assert not frame.empty
    for name in ["biomarkers.csv", "data_dictionary.csv", "README.md", "report.json"]:
        assert (output / name).is_file(), f"Missing report artifact: {name}"
    assert any((output / "plots").glob("*.png")), "Report contains no plots"
