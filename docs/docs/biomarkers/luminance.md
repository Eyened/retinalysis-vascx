---
title: Luminance
---

<!-- Authored page: edit directly. Initial description from the report. -->

Fundus luminance describes the overall brightness of the retinal image. Higher values indicate a brighter image. Calculated as 0.2989 R + 0.5870 G + 0.1140 B, retaining the input intensity scale (typically 0–255 for an 8-bit image). Reported as the mean across the visible retinal area.

Pass `Luminance(grid_field=field)` to restrict the mean to the intersection of
that grid field and the visible retina. Without a grid field, the full visible
retina is used. Grid visibility must meet `min_area_within_bounds`, resolved
from the feature, field, then grid specification, with a default of 0.80.
Insufficient visibility or an empty intersection yields a missing value.
Feature names and plots identify the selected region.

## Report examples

### Macula-centered

![Macula-centered luminance explanation from the Rotterdam Study report](/assets/biomarkers/macula-centered/luminance.jpg)

### OD-centered

![OD-centered luminance explanation from the Rotterdam Study report](/assets/biomarkers/od-centered/luminance.jpg)
