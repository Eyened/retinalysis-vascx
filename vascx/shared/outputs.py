"""Declared scalar and fixed-length numeric outputs for VascX features."""
from dataclasses import dataclass
import re

import numpy as np


@dataclass(frozen=True)
class FeatureOutput:
    """One component; an empty key denotes an existing scalar feature."""

    key: str = ""
    display_name: str = ""
    description: str = ""
    units: str = "dimensionless"


def validate_schema(outputs):
    schema = tuple(outputs)
    if not schema or not all(isinstance(item, FeatureOutput) for item in schema):
        raise ValueError("outputs must contain FeatureOutput definitions")
    keys = [item.key for item in schema]
    if len(set(keys)) != len(keys):
        raise ValueError("Output component keys must be unique")
    if keys != [""] and any(not re.fullmatch(r"[a-z][a-z0-9_]*", key) for key in keys):
        raise ValueError("Vector components require nonempty snake_case keys")
    return schema


def output_values(outputs, value):
    """Validate vector shape; normalize missing components without changing scalars."""
    schema = validate_schema(outputs)
    if len(schema) == 1 and not schema[0].key:
        if value is not None and np.ndim(value) != 0:
            raise ValueError("Scalar feature returned a non-scalar value")
        return (value,)
    if value is None:
        return tuple(np.full(len(schema), np.nan))
    array = np.asarray(value)
    if array.shape != (len(schema),) or array.dtype.kind not in "biuf":
        raise ValueError(f"Expected a numeric vector of shape ({len(schema)},), got {array.shape} / {array.dtype}")
    if np.isinf(array).any():
        raise ValueError("Vector components must be finite or NaN")
    return tuple(array.astype(float))
