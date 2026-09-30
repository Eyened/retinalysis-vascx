"""Coverage helpers for requested regions that may extend beyond the image."""
from __future__ import annotations

import numpy as np


def circle_region_in_bounds(circle, resolution, roi_mask, region_at) -> bool:
    """Test every integer pixel in the complete region inside a circle.

    ``region_at(y, x)`` applies the angular/hemifield constraints on arbitrary
    source coordinates. Image edges remain bounds even without a CFI mask.
    """
    cy, cx = circle.center.tuple
    radius = circle.r
    y0, y1 = int(np.floor(cy - radius)), int(np.ceil(cy + radius)) + 1
    x0, x1 = int(np.floor(cx - radius)), int(np.ceil(cx + radius)) + 1
    y, x = np.arange(y0, y1)[:, None], np.arange(x0, x1)[None, :]
    requested = region_at(y, x)
    rows, cols = np.nonzero(requested)
    if not len(rows):
        return False
    rows, cols = rows + y0, cols + x0
    h, w = resolution
    if np.any((rows < 0) | (rows >= h) | (cols < 0) | (cols >= w)):
        return False
    return roi_mask is None or bool(np.all(roi_mask[rows, cols]))
