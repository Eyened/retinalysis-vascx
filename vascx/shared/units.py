"""Native output units and dimensional conversion, independent of feature names."""
from dataclasses import replace
from collections.abc import Mapping
from math import isfinite

PIXEL_POWERS = {"px": 1, "px²": 2, "px^2": 2, "px⁻¹": -1, "px^-1": -1}
MM_UNITS = {"px": "mm", "px²": "mm²", "px^2": "mm²", "px⁻¹": "mm⁻¹", "px^-1": "mm⁻¹"}

def crop_scale_from_bounds(bounds, target_diameter=1024):
    """Original pixels per processed pixel, using the preprocessing crop radius.

    ``bounds`` must describe the original input image, as saved by fundusprep.
    Missing or invalid bounds leave the scale unknown rather than assuming 1.
    """
    radius = bounds.get("radius") if isinstance(bounds, Mapping) else getattr(bounds, "radius", None)
    try:
        factor = 2 * float(radius) / target_diameter
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    return factor if isfinite(factor) and factor > 0 else None


def spatial_power(units):
    return PIXEL_POWERS.get(units, 0)

def effective_schema(schema, use_mm_per_pixel=False):
    return tuple(replace(o, units=MM_UNITS.get(o.units, o.units))
                 if use_mm_per_pixel else o for o in schema)

def scale_values(schema, values, scale=1.0):
    """Convert native spatial values, preserving missing and nonspatial values."""
    return tuple(v if v is None or not spatial_power(o.units)
                 else v * scale ** spatial_power(o.units)
                 for o, v in zip(schema, values))
