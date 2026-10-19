---
title: "Central retinal equivalents and AVR"
---

**CRE** summarizes the width of the major arteries (CRAE) or veins (CRVE).
VascX offers two algorithms: the default circle-based method and an automated
ARIC-style method. Both use the existing per-segment median widths and the same
Knudtson combination formula:

$$
d \leftarrow c\sqrt{d_1^2+d_2^2}, \qquad
c=0.88\ \text{for arteries},\quad c=0.95\ \text{for veins}.
$$

At each reduction round, the smallest and largest calibers are paired, followed
by the next smallest and largest. An unpaired middle caliber is carried forward;
the results are re-sorted and reduced until one equivalent caliber remains.
CRE is measured in pixels, or mm when calibration is enabled. AVR is unitless.

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

`CRE(measure=CREMeasure.ARIC)` selects eligible vessel trunks in the annulus
between those radii (Zone B: 0.5–1 disc diameters beyond the disc margin).
It avoids selecting a parent together with its daughters. An ungradable parent
may be replaced by usable daughters, which is flagged in the diagnostic result.
The largest eligible vessels are combined **once**, without a median over circles.

The default full-field target is six vessels, with at least four required;
regional variants target three and require at least two. Fewer than the target
can still give a measurement, with a quality flag. By default, the requested
annulus must be fully visible. `max_vessels`, `min_vessels`, and
`min_area_within_bounds` configure these choices.

This is an automated ARIC-style implementation, not an exact reproduction of
manual grading: widths come from the safeguarded **whole-segment medians**, so
measurements can extend beyond Zone B and assume limited tapering. Temporal,
nasal and other regional variants extend the full-field protocol.

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
