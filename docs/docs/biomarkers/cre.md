---
title: "Central retinal equivalents and AVR"
---

**CRE** approximates the caliber of the major arteries (CRAE) or veins (CRVE) from the calibers of the retinal vessels that exit the optic disc.
VascX implements two algorithms. The default is a method that calculates a CRE at several distances of the optic disc boundary, and then takes the median measurement for robustness. The second is an automated version of the protocol in the [ARIC vessel measurement system](https://aric.cscc.unc.edu/aric9/sites/default/files/public/visitdocuments/v5/14b1%20ARIC%20NCS%20VM%20Protocol%2011.23.10.pdf), which performs a single measurement. Both use the same segment caliber calculation and the same Knudtson reduction formula:

$$
d \leftarrow c\sqrt{d_1^2+d_2^2}, \qquad
c=0.88\ \text{for arteries},\quad c=0.95\ \text{for veins}.
$$

At each reduction round, the smallest and largest calibers are paired, followed
by the next smallest and largest. An unpaired middle caliber is carried forward;
the results are re-sorted and reduced until one equivalent caliber remains.
CRE is measured in pixels. AVR is the ratio between the artery and vein caliber estimates, and is unitless.

## Default: concentric circles

`CRE(measure=CREMeasure.Circles)` samples five circles by default, with radii
from 1 to 1.5 optic-disc diameters. On each circle it selects the largest crossing
segments: up to six for the full region, or four for a temporal or nasal region.
Their **whole-segment median widths** are combined into one equivalent caliber.
The final value is the median across valid circles. At least two valid circles
are required by default; circles outside the visible retinal mask are discarded.

This samples the vessel network at several distances from the disc. A segment
can contribute at multiple radii; each circle is combined separately.

## ARIC-style: select trunks once

`CRE(measure=CREMeasure.ARIC)` selects the largest vessel trunks in the Zone B annulus (0.5–1 disc diameters beyond the disc margin). 
In a bifurcation, it selects only the parent when possible, but a parent vessel that cannot be measured
may be replaced by usable daughters. The largest eligible vessels are combined **once** using the Knudtson formula.

The default implementation, designed for optic-disc-centered images measures six vessels, with at least four required.

This is an automated adaptation of the [ARIC protocol](https://aric.cscc.unc.edu/aric9/sites/default/files/public/visitdocuments/v5/14b1%20ARIC%20NCS%20VM%20Protocol%2011.23.10.pdf), not an exact reproduction of
manual grading: segment widths are the median over measurements taken along the entire segment, so
measurements can extend beyond Zone B and assume limited tapering. 

:::tip Hemifield extensions
VascX implements variants of CRE designed for robust caliber measurements on macula-centered images and over retinal hemifields. These apply the reduction formula using vessels from the temporal, nasal, superior or inferior hemifields only. These variants target three vessels and require at least two. By default, the requested annulus must be fully visible. `max_vessels`, `min_vessels`, and `min_area_within_bounds` configure these choices. We note that these variants are not an approximation of the central retinal artery and vein calibers.
:::

## Artery–vein ratio

AVR uses matching arterial and venous CRE measurements from the **same algorithm
and region**:

$$
\mathrm{AVR}=\frac{\mathrm{CRE}_{\mathrm{arteries}}}{\mathrm{CRE}_{\mathrm{veins}}}.
$$

Missing inputs or a zero denominator give a missing ratio. The built-in feature
sets include AVR for the default-count circle and ARIC configurations.

## Report examples

### Macula-centered

![Macula-centered default CRE measurements](/assets/biomarkers/macula-centered/cre_arteries.jpg)

### OD-centered

![OD-centered default CRE measurements](/assets/biomarkers/od-centered/cre_arteries.jpg)

### ARIC-style, OD-centered

![OD-centered ARIC-style CRE measurements](/assets/biomarkers/od-centered/cre_aric_arteries.jpg)
