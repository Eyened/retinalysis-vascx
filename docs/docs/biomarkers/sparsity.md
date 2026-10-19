---
title: "Sparsity"
---

<!-- Authored page: edit directly. Neither documentation generator overwrites this text. -->

**Sparsity.** Let $\mathrm{DT}(x)$ represent the distance transform over $R$, ie. the Euclidean distance to the nearest vessel pixel, measured in image pixels. Over pixels in R we report either the mean or the largest local maximum:

$$
S_{\text{mean}} = \frac{1}{|R|} \sum_{x \in R} \mathrm{DT}(x), \qquad
S_{\max} = \max_{x \in R \cap \text{local maxima}} \mathrm{DT}(x).
$$

## Report examples

### Macula-centered

The report illustrates mean sparsity.

![Macula-centered sparsity explanation from the Rotterdam Study report](/assets/biomarkers/macula-centered/sparsity.jpg)

### OD-centered

The report illustrates mean sparsity.

![OD-centered sparsity explanation from the Rotterdam Study report](/assets/biomarkers/od-centered/sparsity.jpg)

Sparsity is measured in pixels, or mm when calibration is enabled. It is no longer divided by optic-disc–fovea distance. Without fovea input the optional foveal exclusion mask is absent.
