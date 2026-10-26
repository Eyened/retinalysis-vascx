---
title: "Sharpness"
---

<!-- Authored page: edit directly. Neither documentation generator overwrites this text. -->

**Sharpness** captures how sharp or blurry an image is, particularly along vessel edges. Because image sharpness can potentially affect the computation of other biomarkers, sharpness is meant to be an explainable quality control variable, or quality covariate. VascX measures sharpness by capturing how steep the gradient of a vessel edge is. The process is the following:

1. For each vessel segment in the measurement region:

    1. VascX fits a Gaussian-blurred rectangular pulse to green-channel intensity profiles sampled perpendicular to vessel centerlines. The rectangular pulse represents a vessel of uniform intensity and finite width; convolution with a Gaussian produces smooth transitions at its two boundaries. A linear background accommodates local illumination gradients. 
    2. The fitted `sigma` value (standard deviation of the fitted Gaussian blurring kernel), expressed in pixels, represents the sharpness / steepness of the pulse. Larger values indicate broader edge transitions, and therefore higher bluriness.

    By default, a segment of length `L` uses `min(8, max(4, ceil(16 * L / D)))` candidate profiles, uniformly distributed in the spline parameter from one endpoint to the other. `D` is the OD–fovea distance in pixels. Without a fovea, VascX uses 2.5 times the area-equivalent optic-disc diameter. Without a usable disc, it uses half the shorter image dimension.

2. Per-segment median sigma values are weighted by segment length to obtain an aggregate measure of image sharpness. 

:::caution
Despite its name, higher sharpness values indicate a more blurry image. 
:::

## Report examples

### Macula-centered

![Macula-centered sharpness explanation from the Rotterdam Study report](/assets/biomarkers/macula-centered/sharpness_crcl.jpg)

### OD-centered

![OD-centered sharpness explanation from the Rotterdam Study report](/assets/biomarkers/od-centered/sharpness_crcl.jpg)
