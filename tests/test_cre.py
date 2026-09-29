import numpy as np
import pytest

from vascx.fundus.features.cre import CRE, recursive_cre


@pytest.mark.parametrize('cte', [0.88, 0.95])
def test_resorts_aggregated_pairs(cte):
    # First round: sqrt(101), sqrt(29), sqrt(25), scaled by cte.
    # Re-sort so sqrt(29) is carried over, not sqrt(25).
    expected = cte * np.sqrt(cte**4 * 126 + cte**2 * 29)
    assert CRE().recursive_cre([1, 2, 3, 4, 5, 10], cte) == pytest.approx(expected)
    assert recursive_cre([10, 3, 5, 1, 4, 2], cte) == pytest.approx(expected)


@pytest.mark.parametrize('cte', [0.88, 0.95])
def test_four_vessels_unchanged(cte):
    assert recursive_cre([1, 2, 3, 10], cte) == pytest.approx(cte**2 * np.sqrt(114))


def test_empty_and_single_caliber():
    assert recursive_cre([], 0.88) is None
    assert recursive_cre([7], 0.88) == 7


def test_odd_round_carries_middle_caliber():
    cte = 0.88
    assert recursive_cre([1, 2, 10], cte) == pytest.approx(cte * np.sqrt(cte**2 * 101 + 4))
