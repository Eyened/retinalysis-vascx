---
title: Biomarkers
---

These pages describe the measurements and formulas used by VascX.

VascX computes retinal vascular biomarkers from standardized representations (binary masks, undirected/directed graphs, resolved vessels). Below we describe each feature with the exact quantity being estimated and the equations used. Throughout, B denotes the stage‑1 binary vessel mask, S the set of eligible directed segments with lengths $\ell_i$, and R an analysis region of interest; cardinalities count pixels, and distances are in pixels unless noted.

## Reading the examples

The examples are the original explanation grids from the configured Rotterdam
Study macula-centered and OD-centered reports. The paired-image runner refreshes
these reports from a random sample of the evaluated cohort. The plot inventories
below identify the evaluated version and release. Paired artery/vein grids show
arteries in the top row and veins in the bottom row, with the same three
images across both rows. Single-layer features use a 2×2 grid. Features absent from the reports are
marked without substituting a different measurement.

The site uses high-quality JPEG copies of the report images, capped at 2048 pixels
on the longer side. Reusable
plot inventories are available for
[macula-centered images](pathname:///retinalysis-vascx/assets/biomarkers/macula-centered/plots.json)
and [OD-centered images](pathname:///retinalysis-vascx/assets/biomarkers/od-centered/plots.json).

## Units and missing measurements

Native spatial measurements use pixels of the processed image. Enabling
`use_mm_per_pixel` converts lengths to mm, areas to mm², and curvature to mm⁻¹
using each original image's calibration and its `crop_scale_factor`; angles, ratios, counts and Lab values do not
change. The data dictionary records the effective units. The Rotterdam Study
examples instead use pixels corrected to the report's reference device, and
present vascular density as a percentage.

Enabled inputs are required per image: `use_fovea=True` requires finite fovea
coordinates and `use_mm_per_pixel=True` requires positive finite calibration.
Contract failures raise an error for an individual image and skip it in bulk
extraction. An unavailable disc or insufficient region coverage produces
missing measurements rather than an input-contract error. Independent
measurements can still be available.

## Reliability results

The separate **Test-retest reliability** section lists the evaluated combinations
and device-pair ICC results for [macula-centered](../reliability/macula-centered.mdx)
and [OD-centered](../reliability/od-centered.mdx) images. Reliability scores use the normalized measurements from those exports.
