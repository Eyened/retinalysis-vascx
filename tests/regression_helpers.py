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

from tests import settings

from vascx.fundus.loader import RetinaLoader
from vascx.shared.features import FeatureSet
from vascx.utils.analysis import extract_in_parallel

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "samples" / "fundus"
REFERENCE_DIR = REPO_ROOT / "tests" / "reference"
FEATURE_SET_PACKAGE = "vascx.fundus.feature_sets"
MAX_FAILURE_LINES = 100


@dataclass
class RegressionConfig:
    """Store comparison overrides for a feature set."""

    rel_tol: float = settings.BIOMARKER_MAX_PERCENT_CHANGE / 100
    rename_map: dict[str, str] = field(default_factory=dict)
    ignored_missing_features: set[str] = field(default_factory=set)
    ignored_new_features: set[str] = field(default_factory=set)


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


def load_regression_config(
    feature_set_name: str, max_percent_change: float = settings.BIOMARKER_MAX_PERCENT_CHANGE
) -> RegressionConfig:
    """Load schema overrides; the test-time percentage controls all value comparisons."""
    if not np.isfinite(max_percent_change) or max_percent_change < 0:
        raise ValueError("max_percent_change must be finite and non-negative")

    overrides_path = reference_paths(feature_set_name)["overrides"]
    if not overrides_path.exists():
        return RegressionConfig(rel_tol=max_percent_change / 100)

    raw = yaml.safe_load(overrides_path.read_text(encoding="utf-8")) or {}
    return RegressionConfig(
        rel_tol=max_percent_change / 100,
        rename_map=dict(raw.get("rename_map", {})),
        ignored_missing_features=set(raw.get("ignored_missing_features", [])),
        ignored_new_features=set(raw.get("ignored_new_features", [])),
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
            feature_set=feature_set_name,
            naming="canonical",
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
                    "rename_map: {}",
                    "ignored_missing_features: []",
                    "ignored_new_features: []",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    meta = {
        "feature_set": feature_set_name,
        "naming": "canonical",
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
        failures.append(f"{feature_set_name} :: <image> :: {image_id} :: present in reference, absent from current")
    for image_id in sorted(current_images - reference_images):
        failures.append(f"{feature_set_name} :: <image> :: {image_id} :: present in current, absent from reference")

    shared_images = sorted(reference_images & current_images)
    current = current.loc[shared_images]
    reference = reference.loc[shared_images]

    current_columns = set(current.columns) - config.ignored_new_features
    reference_columns = set(reference.columns) - config.ignored_missing_features

    for feature_name in sorted(reference_columns - current_columns):
        failures.append(f"{feature_set_name} :: {feature_name} :: <schema> :: variable present in reference, absent from current")
    for feature_name in sorted(current_columns - reference_columns):
        failures.append(f"{feature_set_name} :: {feature_name} :: <schema> :: variable present in current, absent from reference")

    shared_columns = sorted(reference_columns & current_columns)
    current = current[shared_columns]
    reference = reference[shared_columns]

    for feature_name in shared_columns:
        reference_series = pd.to_numeric(reference[feature_name], errors="coerce")
        current_series = pd.to_numeric(current[feature_name], errors="coerce")
        # Apply the same relative threshold to counts and floating-point biomarkers.
        reference_values = reference_series.to_numpy(dtype=float)
        current_values = current_series.to_numpy(dtype=float)
        with np.errstate(invalid="ignore", over="ignore"):
            differences = np.abs(current_values - reference_values)
            limits = config.rel_tol * np.abs(reference_values)
            # Allow rounding at the boundary (e.g. 1.05 - 1.0), not an absolute floor.
            within_threshold = (differences <= limits) | np.isclose(
                differences, limits, rtol=settings.BIOMARKER_BOUNDARY_EPS_MULTIPLIER * np.finfo(float).eps, atol=0,
            )
        finite = np.isfinite(reference_values) & np.isfinite(current_values)
        equal = (reference_values == current_values) | (
            np.isnan(reference_values) & np.isnan(current_values)
        )
        mismatch_mask = ~(equal | (finite & within_threshold))

        if not np.any(mismatch_mask):
            continue

        mismatch_index = current.index[np.asarray(mismatch_mask)]
        if len(mismatch_index) == 1:
            image_id = mismatch_index[0]
            failures.append(
                f"{feature_set_name} :: {feature_name} :: {image_id} :: "
                f"ref={_format_value(reference_series.loc[image_id])} "
                f"curr={_format_value(current_series.loc[image_id])}"
            )
            continue

        failed_reference = reference_series.loc[mismatch_index]
        failed_current = current_series.loc[mismatch_index]
        # A missing/non-finite value has no meaningful absolute difference.
        # Select the largest measurable difference and report undefined pairs separately.
        with np.errstate(invalid="ignore"):
            differences = (failed_current - failed_reference).abs()
        measurable = differences.dropna()
        if measurable.empty:
            worst = "largest absolute difference unavailable (all failing pairs contain NaN)"
        else:
            image_id = measurable.idxmax()
            worst = (
                f"largest absolute difference: image={image_id} "
                f"ref={_format_value(failed_reference.loc[image_id])} "
                f"curr={_format_value(failed_current.loc[image_id])} "
                f"abs_diff={_format_value(measurable.loc[image_id])}"
            )
        undefined = differences.index[differences.isna()]
        if len(undefined):
            image_id = undefined[0]
            worst += (
                f"; {len(undefined)} failing pair(s) with undefined difference, "
                f"example image={image_id} ref={_format_value(failed_reference.loc[image_id])} "
                f"curr={_format_value(failed_current.loc[image_id])}"
            )
        failures.append(
            f"{feature_set_name} :: {feature_name} :: "
            f"{len(mismatch_index)}/{len(shared_images)} images differ; "
            f"means over failing images (excluding NaN): "
            f"ref={_format_value(failed_reference.mean())} "
            f"curr={_format_value(failed_current.mean())}; {worst}"
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
    lines = [f"{len(failures)} regression issues in {feature_set_name} (allowed change: {config.rel_tol * 100:g}%)", *shown_failures]
    if remainder > 0:
        lines.append(f"... and {remainder} more")
    raise AssertionError("\n".join(lines))


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return format(float(value), ".6g")
    return str(value)
