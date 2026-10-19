---
title: "Background Lab and pigmentation score"
---

`BackgroundLab` measures median CIELAB lightness (L*), green–red (a*) and
blue–yellow (b*) coordinates over the retinal background. It returns one vector,
exported as `background_lab_l`, `background_lab_a`, and `background_lab_b`.
Unenhanced RGB pixels are converted to Lab **before** taking each channel's median.

The background mask excludes vessels, the optic disc and the darkest pixels,
dilates these exclusions, and clips to the visible retina. It covers the full
visible background independently of the vascular measurement grid. Unavailable
inputs or an empty background yield missing values. These colour coordinates
are unaffected by spatial calibration.

The retinal pigmentation score (RPS) is a separate, dataset-level postprocessing
step: `vascx.utils.rps.calculate_rps(df, a_column, b_column)` fits PCA to the finite
median a* and b* values without standardizing them. RPS is the centered first
component score; its sign is flipped if both loadings are positive, following
the original implementation. L* is not used. Missing input rows retain NaN scores.

Filter for image quality before fitting. Scores from independently fitted datasets
or batches are not necessarily comparable. The reports below show the Lab
measurements used as inputs, rather than a fitted RPS.

## RPS reproducibility

The [macula-centered](../reliability/macula-centered.mdx) and
[OD-centered](../reliability/od-centered.mdx) reliability results include `rps`
under **Other**. One PCA is fitted to the unique comparison images pooled across
F1, F2 and all included devices, after the common pair-level quality filters.
Only finite a* and b* measurements enter the fit, and each image counts once even
when it appears in several comparisons. The same PCA scores are used for every
device comparison; ICCs additionally require both images to have an RPS value.

This measures reproducibility of a **Rotterdam-fitted RPS**. It does not evaluate
transfer of the paper's UK Biobank PCA. The PCA is refitted whenever the ICC assets
are regenerated, so changing the eligible cohort can change the score definition.
RPS is derived from the existing Lab measurements during comparison analysis;
extraction exports and reports remain unchanged.

## Report examples

### Macula-centered

![Macula-centered retinal background Lab measurements](/assets/biomarkers/macula-centered/background_lab.jpg)

### OD-centered

![OD-centered retinal background Lab measurements](/assets/biomarkers/od-centered/background_lab.jpg)
