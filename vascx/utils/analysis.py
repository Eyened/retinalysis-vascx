from __future__ import annotations

import shutil
import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd
from joblib import Parallel, delayed
from rtnls_enface.base import EnfaceImage
from rtnls_enface.bounds import make_roi_mask_from_bounds

from vascx.fundus.feature_sets import *
from vascx.fundus.retina import Retina
from vascx.shared.features import FeatureSet
from vascx.shared.inputs import InputContractError, prepare_retina, validate_inputs
from vascx.utils.biomarker_report import (
    render_biomarker_plots,
    write_report_manifest,
)
from vascx.utils.feature_docs import (
    write_biomarker_metadata,
    write_feature_set_readme,
)


def _resolve_feature_set(feature_set: FeatureSet | str) -> FeatureSet:
    """Return a FeatureSet instance from an object or registered name."""
    if isinstance(feature_set, FeatureSet):
        return feature_set
    resolved = FeatureSet.get_by_name(feature_set)
    if resolved is None:
        raise ValueError(f"Feature set '{feature_set}' not found.")
    return resolved


def _load_retina_input(ex, retina_cls=Retina, *, use_mm_per_pixel=False, use_fovea=True):
    """Return an existing retina or construct one from an extraction mapping."""
    if isinstance(ex, EnfaceImage):
        return prepare_retina(ex, use_mm_per_pixel=use_mm_per_pixel, use_fovea=use_fovea)

    retina_kwargs = dict(ex)
    validate_inputs(retina_kwargs.get("mm_per_pixel"), retina_kwargs.get("fovea_location"),
                    use_mm_per_pixel=use_mm_per_pixel, use_fovea=use_fovea)
    if not use_mm_per_pixel:
        retina_kwargs.pop("mm_per_pixel", None)
    if not use_fovea:
        retina_kwargs["fovea_location"] = None
    if "bounds" in retina_kwargs and retina_kwargs["bounds"] is not None:
        roi_mask = make_roi_mask_from_bounds(
            retina_kwargs["bounds"],
            target_diameter=1024,
            install_hint="pip install 'retinalysis-vascx[fundusprep]'",
        )
        if roi_mask is not None:
            retina_kwargs["roi_mask"] = roi_mask
            retina_kwargs["bounds"] = None
    return retina_cls.from_file(**retina_kwargs)


def extract_one(
    ex,
    feature_set: FeatureSet | str,
    retina_cls: EnfaceImage = Retina,
    print_stack_trace: bool = False,
    plots_folder: Optional[str] = None,
    naming: str = "resolved",
    *,
    use_mm_per_pixel: bool = False,
    use_fovea: bool = True,
):
    feature_set = _resolve_feature_set(feature_set)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        retina = _load_retina_input(ex, retina_cls, use_mm_per_pixel=use_mm_per_pixel, use_fovea=use_fovea)
        features = retina.calc_features(feature_set, plots_folder, naming=naming, use_mm_per_pixel=use_mm_per_pixel, use_fovea=use_fovea)

        warning_messages = [str(w.message) for w in caught_warnings]

    return features, warning_messages


def extract_multiple(
    examples: List[Dict],
    feature_set: FeatureSet | str,
    retina_cls: EnfaceImage = Retina,
    print_stack_trace: bool = False,
    plots_folder: Optional[str] = None,
    naming: str = "resolved",
    *,
    use_mm_per_pixel: bool = False,
    use_fovea: bool = True,
):
    """Extract features for multiple examples sequentially."""
    feature_set = _resolve_feature_set(feature_set)
    results = []
    for ex in examples:
        try:
            results.append(extract_one(
                ex, feature_set, retina_cls, print_stack_trace, plots_folder, naming,
                use_mm_per_pixel=use_mm_per_pixel, use_fovea=use_fovea))
        except InputContractError as exc:
            image_id = ex.get("id") if isinstance(ex, Mapping) else getattr(ex, "id", None)
            results.append((None, [f"Skipped image {image_id}: {exc}"]))
    return results


def extract_in_parallel(
    examples: List[Dict],
    feature_set: FeatureSet | str,
    retina_cls=Retina,
    n_jobs: int = 8,
    print_stack_trace: bool = False,
    logger=None,
    plots_folder: Optional[str] = None,
    naming: str = "resolved",
    *,
    use_mm_per_pixel: bool = False,
    use_fovea: bool = True,
):
    feature_set = _resolve_feature_set(feature_set)

    if len(examples) == 0:
        return pd.DataFrame()

    n_workers = min(n_jobs, len(examples))
    base_batch_size, remainder = divmod(len(examples), n_workers)
    example_batches: List[List[Dict]] = []
    start = 0
    for worker_idx in range(n_workers):
        batch_size = base_batch_size + (1 if worker_idx < remainder else 0)
        end = start + batch_size
        example_batches.append(examples[start:end])
        start = end

    batch_results = Parallel(n_jobs=n_workers, verbose=0)(
        delayed(extract_multiple)(
            batch, feature_set, retina_cls, print_stack_trace, plots_folder, naming,
            use_mm_per_pixel=use_mm_per_pixel, use_fovea=use_fovea
        )
        for batch in example_batches
    )
    res = [item for batch in batch_results for item in batch]

    successful = [i for i, r in enumerate(res) if r[0] is not None]
    features = [res[i][0] for i in successful]

    warning_counts: Dict[str, int] = defaultdict(int)
    for r in res:
        for w in r[1]:
            warning_counts[w] += 1

    for msg, count in warning_counts.items():
        if logger is not None or msg.startswith("Skipped image "):
            (logger or logging.getLogger(__name__)).warning(f"{msg} (x{count})" if count > 1 else msg)

    df = pd.DataFrame(features)
    if examples:
        identifiers = [
            ex.get("id") if isinstance(ex, Mapping) else getattr(ex, "id", None)
            for ex in examples
        ]
        if all(identifier is not None for identifier in identifiers):
            df.index = [identifiers[i] for i in successful]
        else:
            df.index = successful
    df.attrs["input_contract_failures"] = {i: res[i][1][0] for i in range(len(res)) if res[i][0] is None}
    return df


_MANAGED_OUTPUT_NAMES = (
    "biomarkers.csv",
    "biomarkers.metadata.csv",
    "biomarkers.names.json",
    "data_dictionary.csv",
    "README.md",
    "report.json",
    "plots",
)


def _prepare_output_folder(output_folder: Union[str, Path], overwrite: bool) -> Path:
    output_path = Path(output_folder)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"Output folder is an existing file: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    managed_paths = [output_path / name for name in _MANAGED_OUTPUT_NAMES]
    existing = [path for path in managed_paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output folder already contains report files: {names}. "
            "Pass overwrite=True to replace them."
        )

    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return output_path


def _input_identifier(retina_input, position: int, prefix: str = "sample") -> str:
    if isinstance(retina_input, Mapping):
        identifier = retina_input.get("id")
    else:
        identifier = getattr(retina_input, "id", None)
    if identifier is None or str(identifier).strip() == "":
        return f"{prefix}_{position + 1:03d}"
    return str(identifier)


def extract_biomarkers_to_folder(
    retinas: Sequence,
    feature_set: Union[FeatureSet, str],
    output_folder: Union[str, Path],
    *,
    retina_cls=Retina,
    n_jobs: int = 8,
    print_stack_trace: bool = False,
    logger=None,
    naming: str = "resolved",
    generate_report: bool = True,
    write_biomarker_values: bool = True,
    report_retinas: Optional[Sequence] = None,
    report_plot_groups: Optional[Mapping[str, str]] = None,
    report_sample_size: Optional[int] = None,
    overwrite: bool = False,
    use_mm_per_pixel: bool = False,
    use_fovea: bool = True,
) -> pd.DataFrame:
    """Extract biomarkers and write a self-contained output folder.

    Inputs may be the dictionaries accepted by `Retina.from_file` or already
    constructed retina objects. Metadata is always written. README, manifest,
    and plot generation can be disabled with `generate_report=False`.
    `report_plot_groups` optionally separates samples into distinct composite figures.
    By default, each group shows three samples for A/V features and four for
    other features. An explicit `report_sample_size` overrides both counts.
    """
    if report_sample_size is not None and report_sample_size < 0:
        raise ValueError("report_sample_size must be non-negative")

    inputs = list(retinas)
    resolved_feature_set = _resolve_feature_set(feature_set)
    identifiers = [
        _input_identifier(retina_input, position)
        for position, retina_input in enumerate(inputs)
    ]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Retina IDs must be unique when writing an output folder")

    output_path = _prepare_output_folder(output_folder, overwrite=overwrite)
    dataframe = extract_in_parallel(
        examples=inputs,
        feature_set=resolved_feature_set,
        retina_cls=retina_cls,
        n_jobs=n_jobs,
        print_stack_trace=print_stack_trace,
        logger=logger,
        naming=naming,
        use_mm_per_pixel=use_mm_per_pixel,
        use_fovea=use_fovea,
    )
    if dataframe.empty:
        raise ValueError("No images satisfied the run input contract")
    failed = dataframe.attrs.get("input_contract_failures", {})
    dataframe.index = [identifier for i, identifier in enumerate(identifiers) if i not in failed]
    if write_biomarker_values:
        dataframe.to_csv(output_path / "biomarkers.csv", index_label="id")

    write_biomarker_metadata(
        resolved_feature_set,
        output_path / "data_dictionary.csv",
        naming=naming,
        use_mm_per_pixel=use_mm_per_pixel,
        use_fovea=use_fovea,
    )

    if not generate_report:
        return dataframe

    if report_retinas is None:
        # Keep enough images per group for either grid layout.
        pool_size = 4 if report_sample_size is None else report_sample_size
        group_counts = {}
        sample_inputs, sample_ids = [], []
        for identifier, retina_input in zip(identifiers, inputs):
            if identifier not in dataframe.index:
                continue
            group = str((report_plot_groups or {}).get(identifier, ""))
            if group_counts.get(group, 0) >= pool_size:
                continue
            sample_inputs.append(retina_input)
            sample_ids.append(identifier)
            group_counts[group] = group_counts.get(group, 0) + 1
    else:
        sample_inputs = list(report_retinas)
        sample_ids = [
            _input_identifier(retina_input, position, prefix="report_sample")
            for position, retina_input in enumerate(sample_inputs)
        ]

    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Report retina IDs must be unique")

    loaded_samples = []
    for sample_id, retina_input in zip(sample_ids, sample_inputs):
        try:
            loaded_samples.append((sample_id, _load_retina_input(
                retina_input, retina_cls, use_mm_per_pixel=use_mm_per_pixel, use_fovea=use_fovea)))
        except InputContractError as exc:
            (logger or logging.getLogger(__name__)).warning("Skipped report image %s: %s", sample_id, exc)
    sample_ids = [sample_id for sample_id, _ in loaded_samples]
    plot_paths = render_biomarker_plots(
        loaded_samples,
        resolved_feature_set,
        output_path,
        dataframe=dataframe,
        naming=naming,
        use_mm_per_pixel=use_mm_per_pixel,
        use_fovea=use_fovea,
        plot_groups=report_plot_groups,
        sample_size=report_sample_size,
    )
    write_feature_set_readme(
        resolved_feature_set,
        output_path / "README.md",
        naming=naming,
        use_mm_per_pixel=use_mm_per_pixel,
        use_fovea=use_fovea,
        dataset_count=len(dataframe),
        plot_paths=plot_paths,
        include_biomarker_values=write_biomarker_values,
    )
    write_report_manifest(
        output_path / "report.json",
        feature_set=resolved_feature_set,
        naming=naming,
        use_mm_per_pixel=use_mm_per_pixel,
        use_fovea=use_fovea,
        dataset_count=len(dataframe),
        sample_ids=sample_ids,
        plot_paths=plot_paths,
        include_biomarker_values=write_biomarker_values,
    )
    return dataframe
