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
    parser.addoption(
        "--run-cli-e2e",
        action="store_true",
        default=False,
        help="Run slow end-to-end tests through the VascX CLI.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip opt-in CLI E2E tests unless explicitly requested."""

    if config.getoption("--run-cli-e2e"):
        return

    skip_cli_e2e = pytest.mark.skip(reason="need --run-cli-e2e option to run")
    for item in items:
        if "cli_e2e" in item.keywords:
            item.add_marker(skip_cli_e2e)


@pytest.fixture
def accept_vascx_reference(pytestconfig: pytest.Config) -> bool:
    """Expose whether the caller wants to refresh references."""

    return bool(pytestconfig.getoption("--accept-vascx-reference"))
