from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom options for reference test maintenance."""

    parser.addoption(
        "--accept-vascx-reference",
        action="store_true",
        default=False,
        help="Refresh stored VascX regression references.",
    )


@pytest.fixture
def accept_vascx_reference(pytestconfig: pytest.Config) -> bool:
    """Expose whether the caller wants to refresh references."""

    return bool(pytestconfig.getoption("--accept-vascx-reference"))
