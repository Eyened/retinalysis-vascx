---
title: "Murray equivalent diameter"
---

**Murray equivalent diameter (MED)** expresses a collection of vessels as the caliber of one hypothetical vessel under a chosen branching power. It is calculated separately for arteries and veins. The defaults are **`p_arteries=1.5`** and **`p_veins=1.25`**, based on empirical Zone B bifurcation closure. These separate constructor arguments accept finite positive numbers and apply to the corresponding vessel type in both standalone use and the hemifields feature sets.

For each measurement radius, VascX sums the powers of the segment diameters:

$$
S_p(r)=\sum_{i\in C(r)} d_i^p,
\qquad
\mathrm{MED}_p=\left[\operatorname{median}_{r}S_p(r)\right]^{1/p}.
$$

The median is taken **before** the root, including when an even number of circles is available. MED has length units: processed-image pixels by default, or physical units through the usual scale conversion. A common diameter scale factor multiplies MED by the same factor.

## Measurement region

Five circles are sampled at 1, 1.125, 1.25, 1.375, and 1.5 optic-disc diameters from the fitted disc center. This spans **Zone B**, 0.5–1 disc diameters beyond the equivalent circular disc margin.

The supplied region selects crossings on these circles. It must geometrically span the entire Zone B radial interval within its angular sector. This check uses an unclipped mask with a one-pixel raster tolerance; an undersized region gives a warning and a missing result. Circle visibility is checked separately against image bounds and the retinal ROI: only circles whose **requested arc is fully visible** contribute. At least two valid arcs are required. Visibility outside the selected sector is irrelevant.

Each segment contributes once per circle, even if it crosses that circle multiple times. All eligible crossing segments contribute, with no largest-vessel cutoff or vessel orientation filter. An empty selection or a selected segment with an invalid diameter makes that circle unavailable.

Diameters are the existing **whole-segment median widths**, as in circle CRE. The crossing must be in the region, but diameter samples may extend outside the region and Zone B. No clipping or new diameter estimator is applied.

## Configuration

```python
from rtnls_enface.grids.disc_centered import DiscCenteredHemifield, DiscCenteredQuadrant
from rtnls_enface.grids.specifications import DiscCenteredGridSpecification, GridFieldSpecification
from vascx.fundus.features.murray import MurrayEquivalentDiameter

full = MurrayEquivalentDiameter()  # p_arteries=1.5, p_veins=1.25
superior = MurrayEquivalentDiameter(
    p_arteries=1.5,
    p_veins=1.25,
    region=GridFieldSpecification(
        DiscCenteredGridSpecification(), DiscCenteredHemifield.Superior),
    plot=True,
)
superior_temporal = MurrayEquivalentDiameter(
    p_arteries=2.8,
    p_veins=2.4,
    region=GridFieldSpecification(
        DiscCenteredGridSpecification(), DiscCenteredQuadrant.SuperiorTemporal),
)
```

Existing superior/inferior/nasal/temporal quadrant sectors retain their original geometry. Anatomical `SuperiorTemporal`, `SuperiorNasal`, `InferiorTemporal`, and `InferiorNasal` quadrants follow the OD–fovea axis and its perpendicular. Other grid-field specifications work when their geometry contains the requested Zone B radial interval.

Both `hemifields` and `hemifields_rs` include full, superior, and inferior MED at power 1.5 for arteries and 1.25 for veins: six additional outputs. MED uses its own OD-centered regions; the other features retain their macula-centered regions. Canonical examples are `med_disc_superior_arteries` and `med_disc_superior_veins`. Quadrant names retain `quadrant_` to distinguish them from hemifields, such as `med_power_2p4_disc_quadrant_superior_temporal_veins`. Only a non-default power for the measured vessel type is included in its output name. For cubic branching, set `p_arteries=3, p_veins=3`.

## Interpretation and calibration

Unlike CRE, MED combines every selected segment with a direct power sum. If a parent and its daughters satisfy the selected power law, their equivalent diameters agree. This conservation is approximate in real vessel trees. Repeated crossings of a tortuous vessel across different segments and segmentation errors can also affect the result.

MED is an equivalent caliber under a branching model, **not a measurement of blood flow**. Diameter alone does not determine velocity. Ratios or asymmetry intended to describe modeled flow capacity should use the power sums, rather than the rooted diameters. Independently taking medians across radii does not guarantee that regional power sums add to the full-region median.

The internal `docs_vascx/calibrate_murray.ipynb` notebook fits separate arterial and venous powers from Zone B bifurcations in the saved documentation image pairs. It uses segment medians directly, without calling MED, and balances the fitting loss across images. Its printed powers characterize empirical bifurcation closure; they do not automatically replace the feature-set default.

Explanation plots show contributing arcs in white, unavailable arcs as gray dashed lines, colored contributing segments, and dots at selected crossings.
