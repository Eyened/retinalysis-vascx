---
title: "Vessel diameters"
---

<!-- Authored page: edit directly. Neither documentation generator overwrites this text. -->

**Caliber.** Diameters are sampled along a spline fitted to each vessel segment's
skeleton, using perpendicular distances to the vessel boundary. Each segment
contributes its median sampled diameter, $d_i$. Eligibility requires more than
`min_numpoints` skeleton points (default 10), plus the configured region and
visibility criteria.

The feature's default aggregator is the median across eligible segments. The
macula-centered and OD-centered feature sets instead use the length-weighted
mean, with spline arc length $\ell_i$ as the weight:

$$
C = \frac{\sum_{i\in S}\ell_i d_i}{\sum_{i\in S}\ell_i}.
$$

Values are in pixels, or mm when calibration is enabled. The plots color the
segments by their individual median widths and display the aggregated value.

## Report examples

### Macula-centered

![Macula-centered caliber explanation from the Rotterdam Study report](/assets/biomarkers/macula-centered/diam_crcl_arteries.jpg)

### OD-centered

![OD-centered caliber explanation from the Rotterdam Study report](/assets/biomarkers/od-centered/diam_crcl_arteries.jpg)
