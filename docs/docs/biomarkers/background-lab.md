---
title: "Background Lab and pigmentation score"
---

**BackgroundLab** measures median CIELAB lightness (L*), green–red (a*) and
blue–yellow (b*) coordinates over the retinal background, excluding vessels and the optic disc. It returns one vector,
exported as `background_lab_l`, `background_lab_a`, and `background_lab_b`.
RGB pixels are converted to Lab **before** taking each channel's median.

The **retinal pigmentation score (RPS)** is a separate, dataset-level postprocessing
step. Because the [RPS is calculated relative to a cohort](https://www.nature.com/articles/s41467-024-55198-7), the RPS is not output directly by VascX. Instead, VascX includes a utility to facilitate computation: `vascx.utils.rps.calculate_rps(df, a_column, b_column)` accepts a dataframe and column names for the **a** and **b** Lab components. It uses PCA to compute the RPS following the [original implementation](https://github.com/uw-biomedical-ml/retinal-pigmentation-score). 

:::tip Attribution
If you use retinal pigment score in VascX, don't forget to cite the [original authors](https://www.nature.com/articles/s41467-024-55198-7):

    Rajesh, A.E., Olvera-Barrios, A., Warwick, A.N. et al. Machine learning derived retinal pigment score from ophthalmic imaging shows ethnicity is not biology. Nat Commun 16, 60 (2025). https://doi.org/10.1038/s41467-024-55198-7
:::

:::caution
Filter for image quality before fitting. Scores from independently fitted datasets or batches are not necessarily comparable.
:::

## Report examples

### Macula-centered

![Macula-centered retinal background Lab measurements](/assets/biomarkers/macula-centered/background_lab.jpg)

### OD-centered

![OD-centered retinal background Lab measurements](/assets/biomarkers/od-centered/background_lab.jpg)
