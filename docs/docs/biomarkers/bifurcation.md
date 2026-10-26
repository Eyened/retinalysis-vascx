---
title: "Bifurcation features"
---

<!-- Authored page: edit directly. -->

One compound feature measures the opening angle, branching coefficient,
absolute cubic junctional exponent deviation (JED), and bifurcation count,
separately for arteries and veins in the configured region.

Daughter directions are obtained from their splines at a distance $\delta$ from the
junction. The opening angle is the angle between those two direction vectors.
A bifurcation is eligible when its position lies in the region, both daughters
are longer than $\delta$, and its angle is finite and at most `max_angle`.
Defaults are `delta=20` pixels and `max_angle=135` degrees.

Let $d_0$ be the parent segment's mean diameter and $d_1,d_2$ the daughter
segments' mean diameters. These are **whole-segment means**, using the existing
VascX diameter machinery, not local widths measured at $\delta$.

$$
BC = \frac{d_1^2+d_2^2}{d_0^2}, \qquad
JED = \frac{\sqrt[3]{\left|d_0^3-d_1^3-d_2^3\right|}}{d_0}.
$$

JED is nonnegative, dimensionless, and zero when the diameters satisfy Murray's
cubic relation. It is not the deviation of a numerically fitted junction exponent.

| Output feature | Meaning |
| --- | --- |
| `bif_angle_mean` | Arithmetic mean of eligible opening angles, in degrees. |
| `bif_branching_coefficient_mean` | Arithmetic mean of the individual branching coefficients. |
| `bif_jed_mean` | Arithmetic mean of the individual absolute cubic JED values. |
| `bif_count` | Number of bifurcations passing the region, daughter-length, and angle filters. |

:::caution
Some of the bifurcation features, especially JED and branching coefficients have low reproducibility scores in [our analysis](../reliability/overview.md).
:::

## Report examples

### Macula-centered

![Macula-centered bifurcation measurements](/assets/biomarkers/macula-centered/bif_crcl_arteries.jpg)

### OD-centered

![OD-centered bifurcation measurements](/assets/biomarkers/od-centered/bif_crcl_arteries.jpg)
