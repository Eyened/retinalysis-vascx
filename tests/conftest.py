from __future__ import annotations

import math
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

from tests import settings


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom options for reference test maintenance."""

    parser.addoption(
        "--vascx-max-percent-change",
        type=float,
        default=settings.BIOMARKER_MAX_PERCENT_CHANGE,
        help=f"Maximum biomarker change per image, relative to the reference (percent; default: {settings.BIOMARKER_MAX_PERCENT_CHANGE:g}).",
    )
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
        help="Compatibility option; inference tests now run by default.",
    )


def pytest_configure(config: pytest.Config) -> None:
    if not math.isfinite(settings.SEGMENTATION_MIN_DICE) or not 0 <= settings.SEGMENTATION_MIN_DICE <= 1:
        raise pytest.UsageError("settings.SEGMENTATION_MIN_DICE must be between 0 and 1")
    for name in (
        "FOVEA_ABS_TOL_PIXELS", "BIOMARKER_BOUNDARY_EPS_MULTIPLIER",
    ):
        value = getattr(settings, name)
        if not math.isfinite(value) or value < 0:
            raise pytest.UsageError(f"settings.{name} must be finite and non-negative")
    for name in ("PIPELINE_MAX_SECONDS_PER_CALL", "CLI_TIMEOUT_SECONDS"):
        value = getattr(settings, name)
        if not math.isfinite(value) or value <= 0:
            raise pytest.UsageError(f"settings.{name} must be finite and positive")
    threshold = config.getoption("--vascx-max-percent-change")
    if not math.isfinite(threshold) or threshold < 0:
        raise pytest.UsageError("--vascx-max-percent-change must be finite and non-negative")
    if os.environ.get("VASCX_PUBLIC_TEST") and config.getoption("--accept-vascx-reference"):
        raise pytest.UsageError("Public validation cannot accept new regression references")


@pytest.fixture
def accept_vascx_reference(pytestconfig: pytest.Config) -> bool:
    """Expose whether the caller wants to refresh references."""

    return bool(pytestconfig.getoption("--accept-vascx-reference"))


@pytest.fixture(scope="session")
def run_vascx():
    """Run the console entry point belonging to the test interpreter."""
    executable = Path(sysconfig.get_path("scripts")) / (
        "vascx.exe" if os.name == "nt" else "vascx"
    )
    assert executable.is_file(), f"VascX console entry point missing: {executable}"

    def run(*args, cwd):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.setdefault("MPLBACKEND", "Agg")
        # Small regression batches do not benefit from large BLAS thread pools.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        log_path = Path(cwd) / f"vascx-{args[0]}.log"
        with log_path.open("w") as log:
            result = subprocess.run(
                [str(executable), *map(str, args)], cwd=cwd, env=env,
                stdout=log, stderr=subprocess.STDOUT, text=True, timeout=settings.CLI_TIMEOUT_SECONDS,
            )
        assert result.returncode == 0, log_path.read_text()
        return result

    return run


@pytest.fixture(scope="session")
def biomarker_cli_input(tmp_path_factory):
    """Adapt stored regression inputs to the run-models output layout."""
    from tests.regression_helpers import SAMPLES_DIR

    folder = tmp_path_factory.mktemp("biomarker_inputs")
    for source, target in {
        "rgb": "preprocessed_rgb", "av": "artery_vein",
        "vessels": "vessels", "discs": "disc",
    }.items():
        shutil.copytree(SAMPLES_DIR / source, folder / target)
    shutil.copy2(SAMPLES_DIR / "meta.csv", folder / "bounds.csv")
    shutil.copy2(SAMPLES_DIR / "fovea.csv", folder / "fovea.csv")
    return folder
