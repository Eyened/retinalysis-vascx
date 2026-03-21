from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Iterable

import sknw

from vascx.fundus.loader import RetinaLoader
from vascx.utils.analysis import extract_one


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

cProfile = importlib.import_module("cProfile")
pstats = importlib.import_module("pstats")

DATASET_CANDIDATES: tuple[Path, ...] = (
    SCRIPT_DIR / "../../samples/fundus",
    SCRIPT_DIR / "../samples/fundus",
    SCRIPT_DIR / "samples/fundus",
)


def resolve_dataset_path(candidates: Iterable[Path]) -> Path:
    """Return the first dataset path that exists."""
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find a samples/fundus dataset folder.")


def warm_up_jit(loader: RetinaLoader, warmup_index: int) -> None:
    """Compile the Numba-backed sknw functions before profiling."""
    retina = loader[warmup_index]
    sknw.build_sknw(retina.arteries.skeleton)


def profile_extract_one(
    loader: RetinaLoader, target_index: int, feature_set_name: str
) -> cProfile.Profile:
    """Profile feature extraction for one example after JIT warmup."""
    example = loader.to_dict()[target_index].copy()
    profiler = cProfile.Profile()
    profiler.enable()
    _ = extract_one(example, feature_set_name)
    profiler.disable()
    return profiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile extract_one after sknw JIT warmup."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the fundus dataset root.",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=0,
        help="Example index to profile.",
    )
    parser.add_argument(
        "--warmup-index",
        type=int,
        default=1,
        help="Retina index used only for JIT warmup.",
    )
    parser.add_argument(
        "--feature-set-name",
        type=str,
        default="full_v3",
        help="Feature set name passed to extract_one.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Number of profile rows to print.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="cumulative",
        help="pstats sort key, for example cumulative or tottime.",
    )
    parser.add_argument(
        "--profile-out",
        type=Path,
        default=None,
        help="Optional path to save the .prof output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset or resolve_dataset_path(DATASET_CANDIDATES)
    loader = RetinaLoader.from_folder(dataset_path)

    if len(loader) == 0:
        raise ValueError(f"No retinas found in {dataset_path}")

    warmup_index = args.warmup_index % len(loader)
    target_index = args.target_index % len(loader)

    print(f"Dataset: {dataset_path.resolve()}")
    print(f"Warmup retina index: {warmup_index}")
    print(f"Target example index: {target_index}")
    print(f"Feature set: {args.feature_set_name}")

    warm_up_jit(loader, warmup_index)
    profiler = profile_extract_one(loader, target_index, args.feature_set_name)

    stats = pstats.Stats(profiler).strip_dirs().sort_stats(args.sort_by)
    stats.print_stats(args.top_n)

    if args.profile_out is not None:
        profiler.dump_stats(args.profile_out)
        print(f"Saved profile to {args.profile_out.resolve()}")


if __name__ == "__main__":
    main()
