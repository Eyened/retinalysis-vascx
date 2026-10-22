from __future__ import annotations

import hashlib
import json
import re
import warnings
from textwrap import fill
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.text import Text

from vascx.fundus.retina import Retina
from vascx.shared.features import FeatureSet
from vascx.shared.naming import make_feature_names, make_output_names
from vascx.utils.feature_docs import get_biomarker_definitions

PlotPaths = Dict[str, List[Tuple[str, str]]]
_MISSING = object()


def _safe_path_component(value: object) -> str:
    raw = str(value)
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", raw).strip("._")
    cleaned = cleaned or "sample"
    if cleaned != raw:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned}-{digest}"
    return cleaned


def _dataframe_value(
    dataframe: Optional[pd.DataFrame],
    image_id: str,
    variable: str,
):
    if dataframe is None or variable not in dataframe.columns:
        return _MISSING

    index_by_string = {str(index): index for index in dataframe.index}
    dataframe_index = index_by_string.get(str(image_id), _MISSING)
    if dataframe_index is _MISSING:
        return _MISSING
    return dataframe.at[dataframe_index, variable]


def _is_missing_value(value: object) -> bool:
    if value is _MISSING or value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except ValueError:
        return False


def _clear_plot_axis(ax) -> None:
    """Remove titles, labels, ticks, and frames from a report panel."""
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()


def _render_feature_panel(
    ax,
    *,
    feature,
    target,
    image_id: str,
    variable: str,
    dataframe: Optional[pd.DataFrame],
    plot_caption: bool = True,
    component_variables: Optional[Sequence[str]] = None,
    use_mm_per_pixel: bool = False,
    use_fovea: bool = True,
    display_scale=None,
    display_schema=None,
) -> bool:
    """Render one feature target, returning whether the panel succeeded."""
    if component_variables is None:
        value = _dataframe_value(dataframe, image_id, variable)
    else:
        values = [_dataframe_value(dataframe, image_id, column) for column in component_variables]
        # Preserve stored NaNs, including partially available vectors.
        value = _MISSING if all(v is _MISSING for v in values) else np.array([
            np.nan if _is_missing_value(v) else v for v in values], dtype=float)
    # Missing measurements still deserve a visual panel. Passing None also
    # avoids recomputing a known missing result merely to draw the image.
    plot_kwargs = {} if value is _MISSING else {
        "computed_value": None if _is_missing_value(value) else value
    }
    plot_kwargs["plot_caption"] = plot_caption
    plot_kwargs.update(use_mm_per_pixel=use_mm_per_pixel, use_fovea=use_fovea)
    if display_scale is not None:
        plot_kwargs["display_scale"] = display_scale
    if display_schema is not None:
        plot_kwargs["display_schema"] = display_schema
    try:
        feature.plot(ax, target, **plot_kwargs)
    except Exception as exc:
        warnings.warn(
            f"Could not render report plot for {image_id}/{variable}: {exc}"
        )
        ax.clear()
        retina = getattr(target, "retina", target)
        image = getattr(retina, "image", None)
        if image is not None:
            ax.imshow(image)
        else:
            ax.set_facecolor("0.15")
        label = "N/A" if _is_missing_value(value) else feature._format_value(value)
        ax.text(0.01, 0.99, label + " — visualization unavailable",
                transform=ax.transAxes, ha="left", va="top", color="white",
                fontsize=8, bbox=dict(facecolor="black", alpha=0.7))
        if image is None:
            ax.text(0.5, 0.5, "Image unavailable", transform=ax.transAxes,
                    ha="center", va="center", color="black")
        # The failed overlay's caption would be misleading on a fallback image.
        _clear_plot_axis(ax)
        return True
    _clear_plot_axis(ax)
    return True


def _scale_report_fonts(fig):
    """Match text at equal displayed widths to the standard 8-inch OD report.

    A/V grids have three columns and some features request larger panels. Fixed
    point sizes would become much smaller when those grids are resized for docs.
    Apply once, after all overlays (including custom artist labels) are created.
    """
    scale = fig.get_figwidth() / 8
    for text in set(fig.findobj(match=Text)):
        text.set_fontsize(text.get_fontsize() * scale)
    return scale


def render_biomarker_plots(
    retinas: Sequence[Tuple[str, Retina]],
    feature_set: FeatureSet,
    output_folder: Union[str, Path],
    *,
    dataframe: Optional[pd.DataFrame] = None,
    naming: str = "resolved",
    plot_groups: Optional[Mapping[str, str]] = None,
    sample_size: Optional[int] = None,
    use_mm_per_pixel: bool = False,
    use_fovea: bool = True,
    display_scales: Optional[Mapping[str, float]] = None,
    display_schemas=None,
) -> PlotPaths:
    """Render sample grids: 2×3 A/V panels or 2×2 single-layer images by default.

    An explicit sample_size overrides both defaults, per plot group.
    """
    if sample_size is not None and sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    root = Path(output_folder)
    plots_folder = root / "plots"
    plots_folder.mkdir(parents=True, exist_ok=True)
    names = make_feature_names(
        feature_set, Retina._target_names_for_feature, naming=naming
    )
    outputs = make_output_names(feature_set, Retina._target_names_for_feature, naming=naming)
    grouped_retinas: Dict[str, List[Tuple[str, Retina]]] = {}
    from vascx.shared.inputs import InputContractError, prepare_retina
    for image_id, retina in retinas:
        try:
            retina = prepare_retina(retina, use_mm_per_pixel=use_mm_per_pixel, use_fovea=use_fovea)
        except InputContractError as exc:
            warnings.warn(f"Skipped report image {image_id}: {exc}")
            continue
        group = ""
        if plot_groups is not None:
            group = str(plot_groups.get(str(image_id), ""))
        grouped_retinas.setdefault(group, []).append((str(image_id), retina))

    plot_paths: PlotPaths = {}
    for feature_index, feature in enumerate(feature_set):
        if not getattr(feature, "plot_in_report", False):
            continue

        for group, samples in grouped_retinas.items():
            sample_targets = [
                (image_id, retina, list(retina.feature_targets(feature)))
                for image_id, retina in samples
            ]
            paired_layers = any(
                {target_name for target_name, _ in targets} >= {"arteries", "veins"}
                for _, _, targets in sample_targets
            )
            variables: List[str] = []
            rendered = 0
            plot_group_name = None

            limit = sample_size if sample_size is not None else (3 if paired_layers else 4)
            sample_targets = sample_targets[:limit]

            panel_width, panel_height = getattr(feature, "preferred_plot_size", (4, 4))
            if paired_layers:
                nrows, ncols = 2, max(1, len(sample_targets))
                fig, axes = plt.subplots(
                    nrows, ncols, figsize=(ncols * panel_width, panel_height * nrows), dpi=200, squeeze=False,
                    layout="constrained"
                )
                for column, (image_id, _retina, targets) in enumerate(sample_targets):
                    target_lookup = dict(targets)
                    for row, target_name in enumerate(("arteries", "veins")):
                        ax = axes[row, column]
                        target = target_lookup.get(target_name)
                        if target is None:
                            _clear_plot_axis(ax)
                            continue
                        item = names[(feature_index, target_name)]
                        plot_group_name = plot_group_name or item.name
                        columns = [outputs[(feature_index, target_name, o.key)].name for o in feature.output_schema()]
                        variables.extend(c for c in columns if c not in variables)
                        rendered += int(
                            _render_feature_panel(
                                ax,
                                feature=feature,
                                target=target,
                                image_id=image_id,
                                variable=item.name,
                                component_variables=columns if feature.is_vector else None,
                                dataframe=dataframe,
                                plot_caption=False,
                                use_mm_per_pixel=use_mm_per_pixel, use_fovea=use_fovea,
                                display_scale=(display_scales or {}).get(str(image_id)),
                                display_schema=(display_schemas or {}).get(feature_index),
                            )
                        )
            else:
                panels = [
                    (image_id, target_name, target)
                    for image_id, _retina, targets in sample_targets
                    for target_name, target in targets
                ]
                nrows = max(1, (len(panels) + 1) // 2)
                fig, axes = plt.subplots(
                    nrows, 2, figsize=(2 * panel_width, panel_height * nrows), dpi=200, squeeze=False,
                    layout="constrained"
                )
                for ax, panel in zip(axes.ravel(), panels):
                    image_id, target_name, target = panel
                    item = names[(feature_index, target_name)]
                    plot_group_name = plot_group_name or item.name
                    columns = [outputs[(feature_index, target_name, o.key)].name for o in feature.output_schema()]
                    variables.extend(c for c in columns if c not in variables)
                    rendered += int(
                        _render_feature_panel(
                            ax,
                            feature=feature,
                            target=target,
                            image_id=image_id,
                            variable=item.name,
                            component_variables=columns if feature.is_vector else None,
                            dataframe=dataframe,
                            plot_caption=False,
                            use_mm_per_pixel=use_mm_per_pixel, use_fovea=use_fovea,
                            display_scale=(display_scales or {}).get(str(image_id)),
                            display_schema=(display_schemas or {}).get(feature_index),
                        )
                    )

            # Do not clear colorbar axes: their scales explain the overlays.
            for ax in axes.ravel():
                _clear_plot_axis(ax)

            if rendered == 0 or not variables:
                plt.close(fig)
                continue

            suffix = f"_{_safe_path_component(group)}" if group else ""
            plot_path = plots_folder / f"{_safe_path_component(plot_group_name)}{suffix}.png"
            font_scale = _scale_report_fonts(fig)
            # One caption per composite, sourced from the same plotting
            # docstring used by standalone feature plots.
            caption = feature.plot_description()
            if paired_layers:
                column_caption = (
                    "The top row displays the explanatory plots for arteries, "
                    "and the bottom row for veins. Each column shows the same image."
                )
                caption = f"{caption} {column_caption}" if caption else column_caption
            if caption:
                wrapped = fill(" ".join(caption.split()), width=110)
                lines = wrapped.count("\n") + 1
                caption_height = (0.18 * lines + 0.25) * font_scale
                fig.set_figheight(panel_height * nrows + caption_height)
                caption_fraction = caption_height / fig.get_figheight()
                fig.text(0.02, 0.02, wrapped, ha="left", va="bottom",
                         fontsize=9 * font_scale, gid="biomarker-caption")
                fig.get_layout_engine().set(rect=(0, caption_fraction, 1, 1 - caption_fraction))
            fig.get_layout_engine().set(w_pad=0.08, h_pad=0.08, hspace=0.02)
            fig.savefig(plot_path, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            relative_path = plot_path.relative_to(root).as_posix()
            link_label = f"{group} ROI" if group else "sample grid"
            for variable in variables:
                plot_paths.setdefault(variable, []).append(
                    (link_label, relative_path)
                )

    # Derived measurements share their source explanations without new grids.
    from vascx.fundus.features.derived import DerivedFeature
    indices = {id(feature): i for i, feature in enumerate(feature_set)}
    for i, feature in enumerate(feature_set):
        if not isinstance(feature, DerivedFeature):
            continue
        links = []
        for dep in feature.dependencies.values():
            source_index = indices.get(id(dep.feature))
            if source_index is not None:
                source_name = names[(source_index, dep.target)].name
                links.extend(plot_paths.get(source_name, []))
        if links:
            plot_paths[names[(i, "retina")].name] = list(dict.fromkeys(links))
    return plot_paths


def _package_version() -> str:
    try:
        return version("retinalysis-vascx")
    except PackageNotFoundError:
        return "unknown"


def write_report_manifest(
    output_file: Union[str, Path],
    *,
    feature_set: FeatureSet,
    naming: str,
    dataset_count: int,
    sample_ids: Sequence[str],
    plot_paths: Mapping[str, Sequence[Tuple[str, str]]],
    include_biomarker_values: bool = True,
    use_mm_per_pixel: bool = False,
    use_fovea: bool = True,
) -> Path:
    """Write machine-readable report provenance."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    definitions = get_biomarker_definitions(feature_set, naming=naming)
    plot_count = len({path for paths in plot_paths.values() for _, path in paths})
    files = {
        "data_dictionary": "data_dictionary.csv",
        "readme": "README.md",
        "plots": "plots",
    }
    if include_biomarker_values:
        files = {"biomarkers": "biomarkers.csv", **files}
    manifest = {
        "schema_version": 2,
        "measurement_definition": "native-spatial-v2",
        "inputs": {"use_mm_per_pixel": use_mm_per_pixel, "use_fovea": use_fovea},
        "calibration": "mm_per_pixel" if use_mm_per_pixel else "pixels",
        "calibration_pixel_space": "original" if use_mm_per_pixel else None,
        "image_metadata_columns": ["crop_scale_factor"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vascx_version": _package_version(),
        "feature_set": feature_set.name,
        "feature_set_description": feature_set.description,
        "naming": naming,
        "dataset": {
            "image_count": dataset_count,
            "report_sample_ids": list(sample_ids),
        },
        "biomarker_count": len(definitions),
        "plot_count": plot_count,
        "files": files,
    }
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path
