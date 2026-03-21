from __future__ import annotations

import cProfile
import io
import pstats
import time

import pytest
import sknw
from vascx.fundus.loader import RetinaLoader
from vascx.utils.analysis import extract_one

from tests.regression_helpers import SAMPLES_DIR

FEATURE_SET_NAME = "full_v3"
MAX_SECONDS_PER_CALL = 2.0
WARMUP_INDEX = 1
PROFILE_ROWS = 20


def warm_up_jit(loader: RetinaLoader, warmup_index: int) -> None:
    """Compile the sknw path before measuring pipeline runtime."""
    retina = loader[warmup_index]
    sknw.build_sknw(retina.arteries.skeleton)


@pytest.mark.profile
def test_pipeline_profile_runtime_guard() -> None:
    """Fail if sample pipeline extraction exceeds the per-call time ceiling."""
    loader = RetinaLoader.from_folder(SAMPLES_DIR)
    assert len(loader) > 0, f"No retinas found in {SAMPLES_DIR}"

    warm_up_jit(loader, WARMUP_INDEX % len(loader))
    examples = [example.copy() for example in loader.to_dict()]

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    for example in examples:
        extract_one(example, FEATURE_SET_NAME)
    profiler.disable()
    elapsed = time.perf_counter() - start

    seconds_per_call = elapsed / len(examples)
    stats_buffer = io.StringIO()
    pstats.Stats(profiler, stream=stats_buffer).strip_dirs().sort_stats(
        "cumulative"
    ).print_stats(PROFILE_ROWS)

    assert seconds_per_call <= MAX_SECONDS_PER_CALL, (
        f"{FEATURE_SET_NAME} averaged {seconds_per_call:.3f}s per call across {len(examples)} "
        f"sample retinas, above the {MAX_SECONDS_PER_CALL:.3f}s ceiling.\n"
        f"Total elapsed: {elapsed:.3f}s\n"
        f"Top cumulative profile rows:\n{stats_buffer.getvalue()}"
    )
