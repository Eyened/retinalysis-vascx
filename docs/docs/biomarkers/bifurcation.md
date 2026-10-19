---
title: "Bifurcation"
---

<!-- Authored page: edit directly. -->

**Bifurcation.** One compound feature measures the opening angle, branching coefficient,
absolute cubic junctional exponent deviation (JED), and filtered bifurcation count,
separately for arteries and veins in the configured region.

Daughter directions are obtained from their splines at distance $\delta$ from the
junction. The opening angle is the angle between those two direction vectors.
A bifurcation is eligible when its position lies in the region, both daughters
are longer than $\delta$, and its angle is finite and at most `max_angle`.
Defaults are `delta=20` pixels, `max_angle=135` degrees,
`spline_error_fraction=0.05`, and `min_bifurcations=3`.
Standard grid visibility rules apply, with a default minimum visible fraction of 0.95.

Let $d_0$ be the parent segment's mean diameter and $d_1,d_2$ the daughter
segments' mean diameters. These are **whole-segment means**, using the existing
VascX diameter machinery, not local widths measured at $\delta$.

$$
BC = \frac{d_1^2+d_2^2}{d_0^2}, \qquad
JED = \frac{\sqrt[3]{\left|d_0^3-d_1^3-d_2^3\right|}}{d_0}.
$$

JED is nonnegative, dimensionless, and zero when the diameters satisfy Murray's
cubic relation. It is not the deviation of a numerically fitted junction exponent.

| Output component | Meaning |
| --- | --- |
| `bif_angle_mean` | Arithmetic mean of eligible opening angles, in degrees. |
| `bif_branching_coefficient_mean` | Arithmetic mean of the individual branching coefficients. |
| `bif_jed_mean` | Arithmetic mean of the individual absolute cubic JED values. |
| `bif_count` | Number passing the region, daughter-length, and angle filters. |

Parameter, region, and layer tokens follow these prefixes under the normal naming
conventions. Coefficients are calculated per bifurcation before averaging.
Each mean needs at least `min_bifurcations` observations in its own valid sample.
Unusable diameter triplets do not remove valid angles or reduce the count.
An adequately visible empty region returns count zero and missing means; an
unavailable or insufficiently visible region returns four missing values.

The visualization shows one full fundus image with two short daughter-direction
rays and an arc marking each eligible bifurcation angle. The markers are unfilled,
and an outline shows the measurement region. Compact labels
beside each marker show `angle° | coefficient | JED`: integer angles and two decimal
places for coefficient and JED. Unavailable diameter measurements display `N/A`.
A semi-transparent box in the top-left corner reports all four values and sample sizes; the fundus fills the panel.

This feature replaces the fundus `BifurcationAngles` and `BifurcationCount` APIs.
Its count is filtered, unlike the former count of every regional branch point.
`full` and `full_v2` now report mean angles rather than medians. The `bergmann`
set no longer includes a bifurcation measurement. The separate FAZ count is unchanged.


## Report examples

### Macula-centered

![Macula-centered bifurcation measurements](/assets/biomarkers/macula-centered/bif_crcl_arteries.jpg)

### OD-centered

![OD-centered bifurcation measurements](/assets/biomarkers/od-centered/bif_crcl_arteries.jpg)
