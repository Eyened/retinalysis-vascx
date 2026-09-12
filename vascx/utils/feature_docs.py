import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from vascx.shared.features import FeatureSet
from vascx.shared.naming import make_feature_names


@dataclass(frozen=True)
class BiomarkerDefinition:
    """Describe one output variable produced by a configured feature set."""

    variable: str
    display_name: str
    description: str
    feature_index: int
    target: str


def _resolve_feature_set(feature_set: Union[FeatureSet, str]) -> FeatureSet:
    import vascx.fundus.feature_sets  # noqa: F401 - register feature sets

    if isinstance(feature_set, FeatureSet):
        return feature_set
    resolved = FeatureSet.get_by_name(feature_set)
    if resolved is None:
        raise ValueError(f"Feature set '{feature_set}' not found.")
    return resolved


def get_biomarker_definitions(
    feature_set: Union[FeatureSet, str],
    naming: str = "resolved",
) -> List[BiomarkerDefinition]:
    """Return ordered, variable-level metadata for a feature set."""
    from vascx.fundus.retina import Retina

    fs = _resolve_feature_set(feature_set)
    names = make_feature_names(fs, Retina._target_names_for_feature, naming=naming)
    definitions = []
    for (feature_index, target), item in names.items():
        feature = fs.features[feature_index]
        definitions.append(
            BiomarkerDefinition(
                variable=item.name,
                display_name=item.display_name,
                description=feature.description(layer_name=target),
                feature_index=feature_index,
                target=target,
            )
        )
    return definitions



def write_variable_display_mapping(
    feature_set: Union[FeatureSet, str],
    out_path: Union[str, Path],
    as_json: bool = False,
    naming: str = "resolved",
) -> Path:
    """Write legacy JSON names or descriptive CSV metadata for a feature set."""
    definitions = get_biomarker_definitions(feature_set, naming=naming)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if as_json:
        mapping: Dict[str, str] = {
            item.variable: item.display_name for item in definitions
        }
        out.write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return out

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["variable", "display_name", "description"]
        )
        writer.writeheader()
        for item in definitions:
            writer.writerow(
                {
                    "variable": item.variable,
                    "display_name": item.display_name,
                    "description": item.description,
                }
            )

    return out


def write_biomarker_metadata(
    feature_set: Union[FeatureSet, str],
    out_path: Union[str, Path],
    naming: str = "resolved",
) -> Path:
    """Write variable, display name, and description columns to CSV."""
    return write_variable_display_mapping(
        feature_set, out_path, as_json=False, naming=naming
    )


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def write_feature_set_readme(
    feature_set: Union[FeatureSet, str],
    output_file: Union[str, Path],
    *,
    naming: str = "resolved",
    dataset_count: Optional[int] = None,
    plot_paths: Optional[Mapping[str, Sequence[Tuple[str, str]]]] = None,
    include_biomarker_values: bool = True,
) -> Path:
    """Write Markdown documentation for a feature set and optional dataset report."""
    fs = _resolve_feature_set(feature_set)
    definitions = get_biomarker_definitions(fs, naming=naming)
    plots = plot_paths or {}
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Biomarker report: `{fs.name}`",
        "",
        fs.description or "No feature-set description available.",
        "",
        "## Extraction",
        "",
        f"- Feature set: `{fs.name}`",
        f"- Naming convention: `{naming}`",
        f"- Number of biomarkers: {len(definitions)}",
    ]
    if dataset_count is not None:
        lines.append(f"- Number of images: {dataset_count}")

    if dataset_count is not None:
        lines.extend(["", "## Output files", ""])
        if include_biomarker_values:
            lines.append("- `biomarkers.csv`: extracted values with one row per image.")
        lines.append(
            "- `data_dictionary.csv`: variable names, display names, and descriptions."
        )
        if plots:
            lines.append("- `plots/`: composite sample visualizations for selected biomarkers.")

    lines.extend(
        [
            "",
            "## Biomarkers",
            "",
            "| Variable | Display name | Description | Sample plots |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in definitions:
        links = ", ".join(
            f"[{_markdown_cell(image_id)}]({path})"
            for image_id, path in plots.get(item.variable, ())
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_markdown_cell(item.variable)}`",
                    _markdown_cell(item.display_name),
                    _markdown_cell(item.description),
                    links,
                ]
            )
            + " |"
        )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path
