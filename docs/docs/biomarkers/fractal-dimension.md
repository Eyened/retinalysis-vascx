---
title: "Fractal dimension"
---

<!-- Authored page: edit directly. Neither documentation generator overwrites this text. -->

**FractalDimension.** Generalized dimensions `q=0`, `q=1`, and `q=2` describe
how vessel coverage and vessel-pixel distribution scale with box size. For box
masses $m_i > 0$, let $p_i = m_i / \sum_j m_j$. D0, D1, and D2 are the slopes
against $-\log(s)$ of $\log(N)$, $-\sum_i p_i\log(p_i)$, and
$-\log(\sum_i p_i^2)$, respectively. The dimensions are unitless finite-scale
estimates; no OD-fovea normalization is applied.

Defaults use the filled binary vessel mask, an image-origin-aligned grid, and
powers-of-two box sizes starting at 4 pixels and ending at half the shorter
bounding-box extent of the visible analysis region. At least four scales are
required. All nonempty boxes count, including fully occupied boxes and boxes
crossing the region or image boundary; only vessel pixels inside the region
contribute mass. Explicit grid fields require at least 80% coverage by default; a feature or
grid `min_area_within_bounds` setting can override this threshold. Empty masks, inadequate coverage,
and insufficient scales return missing values. No rotations, occupancy filtering,
or post-fit ordering/clamping are applied. These choices draw on AutoMorph's
dyadic box counting and PVBM's generalized dimensions but reproduce neither
package exactly. The scale range, mask representation and region must be kept
consistent when comparing estimates.

`FractalDimension(q=0, grid_field=...)` selects one scalar output.
`representation="skeleton"` instead measures centerline pixels (skeletonized
before region clipping); `min_box_size` and `max_box_fraction` configure the
scale range. Nondefault settings appear in canonical names. `get_result(layer)`
returns the three dimensions, box sizes/counts, scale curves, fit coefficients,
R-squared values and any missing-value reason. Results are shared between q
selectors on the same immutable layer. Coarse box sums reuse finer sums; plotting
uses the same cached fits.

The macula-centered and OD-centered feature sets include D0 for arteries and
veins in their configured circular region. The **sparsity** set additionally
includes D0/D1/D2 over multiple regions. The D0 panel displays all three values
and fits as diagnostics, alongside a cyan vessel-mask overlay and gold counted
boxes at one scale. Showing D1 and D2 in the panel does not add them to a feature
set that exports only D0.

## Report examples

### Macula-centered

![Macula-centered fractal dimension](/assets/biomarkers/macula-centered/fd_d0_crcl_arteries.jpg)

### OD-centered

![OD-centered fractal dimension](/assets/biomarkers/od-centered/fd_d0_crcl_arteries.jpg)
