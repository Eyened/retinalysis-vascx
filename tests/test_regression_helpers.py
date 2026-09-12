import numpy as np
import pandas as pd

from tests.regression_helpers import RegressionConfig, compare_frames


def test_multiple_images_are_summarized_per_biomarker():
    reference = pd.DataFrame({"caliber": [1., 10., 100.]}, index=["a", "b", "c"])
    current = pd.DataFrame({"caliber": [3., 16., 100.]}, index=["a", "b", "c"])
    failures = compare_frames("sample", current, reference, RegressionConfig(rel_tol=0.05))
    assert len(failures) == 1
    assert "2/3 images differ" in failures[0]
    assert "ref=5.5 curr=9.5" in failures[0]
    assert "image=b ref=10 curr=16 abs_diff=6" in failures[0]


def test_schema_messages_identify_both_sets():
    reference = pd.DataFrame({"old": [1]}, index=["a"])
    current = pd.DataFrame({"new": [1]}, index=["a"])
    failures = compare_frames("sample", current, reference, RegressionConfig(rel_tol=0.05))
    assert any("old :: <schema> :: variable present in reference, absent from current" in f for f in failures)
    assert any("new :: <schema> :: variable present in current, absent from reference" in f for f in failures)


def test_missing_values_do_not_hide_largest_measurable_difference():
    reference = pd.DataFrame({"caliber": [np.nan, 2.]}, index=["a", "b"])
    current = pd.DataFrame({"caliber": [3., 5.]}, index=["a", "b"])
    failures = compare_frames("sample", current, reference, RegressionConfig(rel_tol=0.05))
    assert len(failures) == 1
    assert "image=b ref=2 curr=5 abs_diff=3" in failures[0]
    assert "undefined difference, example image=a ref=nan curr=3" in failures[0]


def test_all_nan_mismatches_are_reported_without_crashing():
    reference = pd.DataFrame({"caliber": [np.nan, np.nan]}, index=["a", "b"])
    current = pd.DataFrame({"caliber": [3., 5.]}, index=["a", "b"])
    failures = compare_frames("sample", current, reference, RegressionConfig(rel_tol=0.05))
    assert len(failures) == 1
    assert "largest absolute difference unavailable" in failures[0]


def test_single_mismatch_keeps_image_values_and_matching_values_pass():
    reference = pd.DataFrame({"caliber": [2.]}, index=["a"])
    assert compare_frames("sample", reference, reference, RegressionConfig(rel_tol=0.05)) == []
    current = pd.DataFrame({"caliber": [4.]}, index=["a"])
    assert compare_frames("sample", current, reference, RegressionConfig(rel_tol=0.05)) == [
        "sample :: caliber :: a :: ref=2 curr=4"
    ]


def test_five_percent_is_inclusive_for_floats_counts_and_negative_values():
    reference = pd.DataFrame({"x": [1., 100., -100., 100.]})
    current = pd.DataFrame({"x": [1.05, 105., -95., 95.]})
    assert compare_frames("sample", current, reference, RegressionConfig(rel_tol=0.05)) == []
    current.loc[0, "x"] = 1.05001
    assert len(compare_frames("sample", current, reference, RegressionConfig(rel_tol=0.05))) == 1


def test_individual_failures_cannot_cancel_in_the_mean():
    reference = pd.DataFrame({"x": [100., 100.]})
    current = pd.DataFrame({"x": [106., 94.]})
    assert len(compare_frames("sample", current, reference, RegressionConfig(rel_tol=0.05))) == 1


def test_zero_and_nonfinite_references():
    reference = pd.DataFrame({"x": [0., np.nan, np.inf, -np.inf]})
    assert compare_frames("sample", reference, reference, RegressionConfig(rel_tol=0.05)) == []
    for index, value in [(0, 1e-20), (1, 1.), (2, 1.), (3, np.inf)]:
        current = reference.copy()
        current.loc[index, "x"] = value
        assert len(compare_frames("sample", current, reference, RegressionConfig(rel_tol=0.05))) == 1


def test_test_time_threshold_overrides_old_yaml_tolerances(tmp_path, monkeypatch):
    from tests import regression_helpers as helpers

    monkeypatch.setattr(helpers, "REFERENCE_DIR", tmp_path)
    (tmp_path / "example.overrides.yaml").write_text("abs_tol: 100\nrel_tol: 1\n")
    reference = pd.DataFrame({"x": [100.]})
    current = pd.DataFrame({"x": [106.]})
    assert helpers.compare_frames("example", current, reference, helpers.load_regression_config("example", 5))
    assert not helpers.compare_frames("example", current, reference, helpers.load_regression_config("example", 10))
