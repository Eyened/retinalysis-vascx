"""Native output units and dimensional conversion, independent of feature names."""
from dataclasses import replace

PIXEL_POWERS = {"px": 1, "px²": 2, "px^2": 2, "px⁻¹": -1, "px^-1": -1}
MM_UNITS = {"px": "mm", "px²": "mm²", "px^2": "mm²", "px⁻¹": "mm⁻¹", "px^-1": "mm⁻¹"}

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
