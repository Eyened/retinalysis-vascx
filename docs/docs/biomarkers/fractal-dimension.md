---
title: "Fractal dimension"
---

<!-- Authored page: edit directly. Neither documentation generator overwrites this text. -->

**Fractal dimension** describes how the retinal vessel pattern fills space at
different scales. Higher values generally indicate a more space-filling pattern.

VascX measures this by placing a grid over the vessel segmentation and counting
the squares that contain vessels. It repeats this with different square sizes.
The rate at which the count increases as the squares get smaller gives the
fractal dimension. This method is called **box counting**.

The `macula_centered` and `od_centered` feature sets report **D0**, separately for
arteries and veins within their circular analysis region. For example,
`fd_d0_crcl_arteries` is the arterial D0 measurement.

## D0, D1 and D2

VascX supports three related measures:

- **D0** counts every square containing vessels equally. This is the default.
- **D1** also takes into account how vessel pixels are distributed between squares.
- **D2** gives more weight to squares containing more vessel pixels.

All three are unitless. The `sparsity` feature set includes all three measures.
Keep the image resolution, analysis region and measurement settings consistent
when comparing results.

A result is missing if there are no vessels in the region, too little of the
region is visible, or the region is too small to measure at enough square sizes.

## Report examples

Cyan highlights the vessels being measured, and gold squares show which boxes
were counted at one size. The small graph shows the measurements across box
sizes and the fitted lines used to calculate the dimensions.

The plots display D0, D1 and D2, even when the feature set only exports D0.

### Macula-centered

![Macula-centered fractal dimension](/assets/biomarkers/macula-centered/fd_d0_crcl_arteries.jpg)

### OD-centered

![OD-centered fractal dimension](/assets/biomarkers/od-centered/fd_d0_crcl_arteries.jpg)
