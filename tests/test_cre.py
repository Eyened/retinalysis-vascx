from __future__ import annotations

import pytest

from vascx.fundus.features.cre import CRE, CREMode


def test_cre_resorts_diameters_after_each_reduction():
    # The first round leaves the smallest reduced diameter in the middle.
    calibers = [130, 50, 150, 80, 120, 60]

    result = CRE(CREMode.Full).recursive_cre(calibers, 0.88)

    assert result == pytest.approx(183.31407320953838)
