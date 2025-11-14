from pathlib import Path
from typing import Union
import inspect

from vascx.shared.features import FeatureSet


def write_feature_descriptions(feature_set: str, desc_file: Union[str, Path]) -> Path:
    """Write per-feature metadata (repr + docstring) for a feature set to desc_file."""
    desc_path = Path(desc_file)
    desc_path.parent.mkdir(parents=True, exist_ok=True)

    fs = FeatureSet.get_by_name(feature_set)
    lines = []
    for key, feat in fs.items():
        doc = inspect.getdoc(feat.__class__) or "No description available."
        lines.extend([f"[{key}]", repr(feat), doc.strip(), ""])  # blank line between entries

    desc_path.write_text("\n".join(lines), encoding="utf-8")
    return desc_path


