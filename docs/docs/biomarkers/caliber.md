---
title: "Vessel calibers"
---

<!-- Authored page: edit directly. Neither documentation generator overwrites this text. -->

VascX implements general diameter measurements, meant to capture arterial and venular retinal caliber.
Diameters are sampled along a spline fitted to each vessel segment's skeleton, using perpendicular distances to the vessel boundary. Each segment
contributes its median sampled diameter, $d_i$. Segments with less than
`min_numpoints` skeleton points (default 10) are discarded to prevent

:::tip Difference with CRE
In practice, calibers are different from [CREs](./cre.md) in that calibers measured all eligible vessel segments, while CREs measure only the largest vessels in Zone B. This may make calibers more sensitive to image quality differences.
:::

By default, VascX aggregates across segments using the length-weighted mean, with spline arc length $\ell_i$ as the weight:

$$
C = \frac{\sum_{i\in S}\ell_i d_i}{\sum_{i\in S}\ell_i}.
$$

Output values are measured in pixels.

## Report examples

### Macula-centered

![Macula-centered caliber explanation from the Rotterdam Study report](/assets/biomarkers/macula-centered/diam_crcl_arteries.jpg)

### OD-centered

![OD-centered caliber explanation from the Rotterdam Study report](/assets/biomarkers/od-centered/diam_crcl_arteries.jpg)
