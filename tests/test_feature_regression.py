from __future__ import annotations

import pytest

from tests.regression_helpers import (
    assert_matches_reference,
    discover_feature_set_names,
    extract_feature_frame,
    load_reference_frame,
    load_regression_config,
    write_reference_artifacts,
)

FEATURE_SET_NAMES = discover_feature_set_names()


@pytest.mark.reference
@pytest.mark.parametrize("feature_set_name", FEATURE_SET_NAMES, ids=FEATURE_SET_NAMES)
def test_feature_set_regression(
    feature_set_name: str, accept_vascx_reference: bool
) -> None:
    """Compare extracted biomarker outputs against stored references."""

    current = extract_feature_frame(feature_set_name)
    if accept_vascx_reference:
        write_reference_artifacts(feature_set_name, current)

    reference = load_reference_frame(feature_set_name)
    config = load_regression_config(feature_set_name)
    assert_matches_reference(feature_set_name, current, reference, config)
