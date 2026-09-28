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

    # Group from complete structured names, before feature-set resolution
    # removes constant parameters. Region and layer belong to each entry.
    groups = {}
    for item in definitions:
        feature = fs.features[item.feature_index]
        parts = tuple(feature.name_parts(layer_name=item.target))
        common = tuple(part for part in parts if part.key not in {
            "grid", "grid_parameters", "field", "field_parameters", "layer"
        })
        key = (type(feature), tuple((part.key, part.tokens) for part in common))
        groups.setdefault(key, []).append((item, feature, common, parts))

    lines.extend(["", "## Biomarkers"])
    for entries in groups.values():
        _, feature, common, _ = entries[0]
        title = " ".join(part.display for part in common
                         if part.display and not part.annotation).strip()
        parameters = "; ".join(part.display for part in common
                               if part.display and part.annotation)
        if parameters:
            title += f" ({parameters})"
        lines.extend(["", f"### {title or type(feature).__name__}", ""])
        # Describe the method once, without claiming that every entry is arterial
        # or belongs to the first region encountered in the feature set.
        description = " ".join(part.strip() for part in (
            feature.general_description,
            feature.implementation_description(layer_name="vessels"),
            feature.aggregation_description(layer_name="vessels"),
        ) if part and part.strip())
        # The neutral layer label is "retinal vessel"; some implementation
        # templates already append "vessels" or "vessel segments" themselves.
        description = (description.replace("retinal vessel vessels", "retinal vessels")
                       .replace("retinal vessel vessel segments", "retinal vessel segments")
                       .replace("retinal vessel resolved vessels", "resolved retinal vessels"))
        lines.extend([description or "No publication description available.", "",
                      "Computed combinations:", ""])
        regions = {}
        for item, instance, _, parts in entries:
            region = instance.region_description(layer_name=item.target)
            # Preserve geometric parameters even if the hand-written region
            # description omits a configured crop fraction or radius.
            parameters = "; ".join(part.display for part in parts
                                   if part.key in {"grid_parameters", "field_parameters"}
                                   and part.display)
            label = region + (f" Parameters: {parameters}." if parameters else "")
            regions.setdefault(label, []).append(item)
        for region, items in regions.items():
            combinations = "; ".join(
                f"{item.target}: `{item.variable}`" for item in items
            )
            lines.append(f"- {region} {combinations}.")
        # A composite artery/vein plot can be referenced by multiple variables.
        # Link it once within the section, with all matching variables named.
        section_plots = {}
        for item, _, _, _ in entries:
            for label, path in plots.get(item.variable, ()):
                section_plots.setdefault((label, path), []).append(item.variable)
        if section_plots:
            lines.extend(["", "Sample plots:", ""])
            for (label, path), variables in section_plots.items():
                names = ", ".join(f"`{variable}`" for variable in variables)
                lines.append(f"- [{_markdown_cell(label)}]({path}) — {names}")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path
