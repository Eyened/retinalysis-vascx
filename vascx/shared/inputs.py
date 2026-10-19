"""Per-image requirements of an explicitly configured extraction run."""
from copy import deepcopy
import numpy as np

class InputContractError(ValueError):
    """An image lacks a valid input required by this run."""

def validate_inputs(mm_per_pixel, fovea_location, *, use_mm_per_pixel=False, use_fovea=True):
    if use_mm_per_pixel:
        try:
            valid = (not isinstance(mm_per_pixel, (bool, np.bool_))
                     and np.ndim(mm_per_pixel) == 0 and mm_per_pixel is not None
                     and np.isfinite(float(mm_per_pixel)) and float(mm_per_pixel) > 0)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise InputContractError("mm_per_pixel must be a positive finite scalar when enabled")
    if use_fovea:
        try:
            point = ([fovea_location.x, fovea_location.y]
                     if hasattr(fovea_location, "x") else fovea_location)
            valid = np.shape(point) == (2,) and np.isfinite(np.asarray(point, dtype=float)).all()
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise InputContractError("fovea_location must contain two finite coordinates when enabled")

def prepare_retina(retina, *, use_mm_per_pixel=False, use_fovea=True):
    """Disable unused inputs on an isolated copy, including any dependent caches."""
    mm = getattr(retina, "mm_per_pixel", None)
    fovea = getattr(retina, "fovea_location", None)
    validate_inputs(mm, fovea, use_mm_per_pixel=use_mm_per_pixel, use_fovea=use_fovea)
    if (not use_mm_per_pixel and mm is not None) or (not use_fovea and fovea is not None):
        retina = deepcopy(retina)
        if not use_mm_per_pixel:
            retina.scaling_factor = None
        if not use_fovea:
            retina.fovea_location = None
        retina._invalidate_disc_caches()
    return retina
