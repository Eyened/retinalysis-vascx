from __future__ import annotations

import importlib
import json
import pkgutil
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from vascx.fundus.loader import RetinaLoader
from vascx.shared.features import FeatureSet
from vascx.utils.analysis import extract_in_parallel

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "samples" / "fundus"
REFERENCE_DIR = REPO_ROOT / "tests" / "reference"
FEATURE_SET_PACKAGE = "vascx.fundus.feature_sets"
DEFAULT_ABS_TOL = 1e-4
DEFAULT_REL_TOL = 1e-3
MAX_FAILURE_LINES = 25


@dataclass
class RegressionConfig:
    """Store comparison overrides for a feature set."""

    abs_tol: float = DEFAULT_ABS_TOL
    rel_tol: float = DEFAULT_REL_TOL
    rename_map: dict[str, str] = field(default_factory=dict)
    ignored_missing_features: set[str] = field(default_factory=set)
    ignored_new_features: set[str] = field(default_factory=set)
    per_feature_tolerances: dict[str, dict[str, float]] = field(default_factory=dict)


def discover_feature_set_names() -> list[str]:
    """Return all feature-set names defined in the fundus feature-set package."""

    package = importlib.import_module(FEATURE_SET_PACKAGE)
    feature_set_names: set[str] = set()
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{FEATURE_SET_PACKAGE}.{module_info.name}")
        for value in vars(module).values():
            if isinstance(value, FeatureSet):
                feature_set_names.add(value.name)

    if not feature_set_names:
        raise AssertionError("No fundus feature sets were discovered")

    return sorted(feature_set_names)


def reference_paths(feature_set_name: str) -> dict[str, Path]:
    """Return the artifact paths for one feature set."""

    return {
        "parquet": REFERENCE_DIR / f"{feature_set_name}.parquet",
        "meta": REFERENCE_DIR / f"{feature_set_name}.meta.json",
        "overrides": REFERENCE_DIR / f"{feature_set_name}.overrides.yaml",
    }


def load_regression_config(feature_set_name: str) -> RegressionConfig:
    """Load yaml overrides for one feature set."""

    overrides_path = reference_paths(feature_set_name)["overrides"]
    if not overrides_path.exists():
        return RegressionConfig()

    raw = yaml.safe_load(overrides_path.read_text(encoding="utf-8")) or {}
    return RegressionConfig(
        abs_tol=float(raw.get("abs_tol", DEFAULT_ABS_TOL)),
        rel_tol=float(raw.get("rel_tol", DEFAULT_REL_TOL)),
        rename_map=dict(raw.get("rename_map", {})),
        ignored_missing_features=set(raw.get("ignored_missing_features", [])),
        ignored_new_features=set(raw.get("ignored_new_features", [])),
        per_feature_tolerances=dict(raw.get("per_feature_tolerances", {})),
    )


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows and columns to make stored references deterministic."""

    return df.sort_index(axis=0).sort_index(axis=1)


def extract_feature_frame(feature_set_name: str) -> pd.DataFrame:
    """Run biomarker extraction on the packaged sample dataset."""

    try:
        loader = RetinaLoader.from_folder(SAMPLES_DIR)
        df = extract_in_parallel(
            loader.to_dict(),
            feature_set_name=feature_set_name,
            n_jobs=1,
            print_stack_trace=True,
        )
    except Exception as exc:  # pragma: no cover - exercised via pytest failure path
        raise AssertionError(
            f"{feature_set_name} :: <runtime> :: <suite> :: {exc}"
        ) from exc

    return normalize_frame(df)


def load_reference_frame(feature_set_name: str) -> pd.DataFrame:
    """Load the stored parquet baseline for one feature set."""

    parquet_path = reference_paths(feature_set_name)["parquet"]
    if not parquet_path.exists():
        raise AssertionError(
            f"{feature_set_name} :: <reference> :: <suite> :: missing {parquet_path.name}"
        )
    return normalize_frame(pd.read_parquet(parquet_path))


def write_reference_artifacts(feature_set_name: str, df: pd.DataFrame) -> None:
    """Persist the dataframe baseline and its metadata for one feature set."""

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    paths = reference_paths(feature_set_name)
    normalized = normalize_frame(df)
    normalized.to_parquet(paths["parquet"])

    if not paths["overrides"].exists():
        paths["overrides"].write_text(
            "\n".join(
                [
                    "abs_tol: 1.0e-4",
                    "rel_tol: 1.0e-3",
                    "rename_map: {}",
                    "ignored_missing_features: []",
                    "ignored_new_features: []",
                    "per_feature_tolerances: {}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    meta = {
        "feature_set": feature_set_name,
        "sample_dir": str(SAMPLES_DIR.relative_to(REPO_ROOT)),
        "image_ids": normalized.index.tolist(),
        "feature_count": int(normalized.shape[1]),
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "accept_reason": "accepted via --accept-vascx-reference",
        "python_version": platform.python_version(),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")


def compare_frames(
    feature_set_name: str,
    current: pd.DataFrame,
    reference: pd.DataFrame,
    config: RegressionConfig,
) -> list[str]:
    """Return concise regression failures for one feature set."""

    failures: list[str] = []
    current = normalize_frame(current).rename(columns={v: k for k, v in config.rename_map.items()})
    reference = normalize_frame(reference)

    current_images = set(current.index)
    reference_images = set(reference.index)

    for image_id in sorted(reference_images - current_images):
        failures.append(f"{feature_set_name} :: <image> :: {image_id} :: missing")
    for image_id in sorted(current_images - reference_images):
        failures.append(f"{feature_set_name} :: <image> :: {image_id} :: unexpected")

    shared_images = sorted(reference_images & current_images)
    current = current.loc[shared_images]
    reference = reference.loc[shared_images]

    current_columns = set(current.columns) - config.ignored_new_features
    reference_columns = set(reference.columns) - config.ignored_missing_features

    for feature_name in sorted(reference_columns - current_columns):
        failures.append(f"{feature_set_name} :: {feature_name} :: <schema> :: missing")
    for feature_name in sorted(current_columns - reference_columns):
        failures.append(f"{feature_set_name} :: {feature_name} :: <schema> :: unexpected")

    shared_columns = sorted(reference_columns & current_columns)
    current = current[shared_columns]
    reference = reference[shared_columns]

    for feature_name in shared_columns:
        tolerance = config.per_feature_tolerances.get(feature_name, {})
        abs_tol = float(tolerance.get("abs_tol", config.abs_tol))
        rel_tol = float(tolerance.get("rel_tol", config.rel_tol))

        reference_series = pd.to_numeric(reference[feature_name], errors="coerce")
        current_series = pd.to_numeric(current[feature_name], errors="coerce")

        if _is_integer_like(reference_series) and _is_integer_like(current_series):
            mismatch_mask = ~(
                (reference_series == current_series)
                | (reference_series.isna() & current_series.isna())
            )
        else:
            mismatch_mask = ~(
                np.isclose(
                    reference_series.to_numpy(dtype=float),
                    current_series.to_numpy(dtype=float),
                    rtol=rel_tol,
                    atol=abs_tol,
                    equal_nan=True,
                )
            )

        if not np.any(mismatch_mask):
            continue

        mismatch_index = current.index[np.asarray(mismatch_mask)]
        for image_id in mismatch_index:
            failures.append(
                f"{feature_set_name} :: {feature_name} :: {image_id} :: "
                f"ref={_format_value(reference.loc[image_id, feature_name])} "
                f"cur={_format_value(current.loc[image_id, feature_name])}"
            )

    return failures


def assert_matches_reference(
    feature_set_name: str,
    current: pd.DataFrame,
    reference: pd.DataFrame,
    config: RegressionConfig,
) -> None:
    """Raise an assertion with concise mismatch lines when drift is detected."""

    failures = compare_frames(feature_set_name, current, reference, config)
    if not failures:
        return

    shown_failures = failures[:MAX_FAILURE_LINES]
    remainder = len(failures) - len(shown_failures)
    lines = [f"{len(failures)} regression mismatches in {feature_set_name}", *shown_failures]
    if remainder > 0:
        lines.append(f"... and {remainder} more")
    raise AssertionError("\n".join(lines))


def _is_integer_like(series: pd.Series) -> bool:
    values = series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return False
    return bool(np.all(np.isclose(values, np.round(values), atol=0.0, rtol=0.0)))


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return format(float(value), ".6g")
    return str(value)
