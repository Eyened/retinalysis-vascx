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
    for idx, feat in enumerate(fs, start=1):
        doc = inspect.getdoc(feat.__class__) or "No description available."
        lines.extend([f"[feature_{idx:03d}]", repr(feat), doc.strip(), ""])

    desc_path.write_text("\n".join(lines), encoding="utf-8")
    return desc_path


