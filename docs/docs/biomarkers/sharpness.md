---
title: "Vessel-edge sharpness"
---

<!-- Authored page: edit directly. Neither documentation generator overwrites this text. -->

**Sharpness.** Estimates vessel-edge blur width (ESF sigma) from perpendicular
intensity profiles. By default, a segment of length `L` uses
`min(8, max(4, ceil(16 * L / D)))` candidate profiles, uniformly distributed in
the spline parameter from one endpoint to the other. `D` is the OD–fovea distance
in pixels; without a fovea, use 2.5 times the area-equivalent optic-disc diameter;
without a usable disc, use half the shorter image dimension. Profile rejection
for bounds, neighboring vessels, or invalid fits can leave fewer than four valid
measurements. `Sharpness(n_profiles=12)` retains the previous fixed-count sampling
for comparisons. Per-segment median sigma values are weighted by segment length.

## Report examples

### Macula-centered

![Macula-centered sharpness explanation from the Rotterdam Study report](/assets/biomarkers/macula-centered/sharpness_crcl.jpg)

### OD-centered

![OD-centered sharpness explanation from the Rotterdam Study report](/assets/biomarkers/od-centered/sharpness_crcl.jpg)

The measured ESF width is in pixels, or mm when calibration is enabled. Disc–fovea distance controls sampling density, not normalization of the measured width; the documented sampling fallback remains available.
