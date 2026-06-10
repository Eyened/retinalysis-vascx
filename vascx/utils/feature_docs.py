import csv
import inspect
import json
from pathlib import Path
from typing import Dict, Union

from vascx.shared.features import FeatureSet


def write_feature_descriptions(feature_set: str, desc_file: Union[str, Path]) -> Path:
    """Write per-feature metadata (repr + docstring) for a feature set to desc_file."""
    desc_path = Path(desc_file)
    desc_path.parent.mkdir(parents=True, exist_ok=True)

    fs = FeatureSet.get_by_name(feature_set)
    lines = []
    for idx, feat in enumerate(fs, start=1):
        doc = inspect.getdoc(feat.__class__) or "No description available."
        lines.extend([f"[feature_{idx:03d}]", repr(feat), doc.strip(), ""])

    desc_path.write_text("\n".join(lines), encoding="utf-8")
    return desc_path


def write_variable_display_mapping(
    feature_set_name: str, out_path: Union[str, Path], as_json: bool = False
) -> Path:
    """Write a mapping from canonical variable names to display names."""
    import vascx.fundus.feature_sets  # noqa: F401 — register FeatureSet instances

    from vascx.fundus.retina import Retina

    fs = FeatureSet.get_by_name(feature_set_name)
    if fs is None:
        raise ValueError(f"Feature set '{feature_set_name}' not found.")

    mapping: Dict[str, str] = Retina.make_feature_display_names(fs)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if as_json:
        out.write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return out

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "display_name"])
        writer.writerows(mapping.items())

    return out

