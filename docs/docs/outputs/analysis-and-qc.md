---
title: Analysis and QC
---

VascX provides measurements that help assess image quality, without imposing a single universal image-level pass/fail rule. Appropriate quality control (QC) depends on the research question, the images and the biomarkers being analysed.

:::warning Before applying VascX
VascX was developed and evaluated on images with no vascular alterations, or *common* vascular alterations due to, for example, aging and hypertension. We **do not recommend its application on images from subjects with severe or rare abnormalities** in their retinal vasculature.
:::

## Recommended approach

For fully automatic analyses, we recommend a multi-step approach to image quality control in VascX analyses:
1. **Global image filter.** Exclude images with severe quality issues using a quality classifier, such as the model trained on EyePACS, which is included in VascX. Exclude images classified as `bad`, both `bad` and `useable`, or use a calibrated threshold, depending on the analysis.
2. **Run VascX** on the accepted images. VascX implements region-aware measurements, which means that a feature will produce a null / NaN output if the anatomical measurement region falls out of bounds. VascX implements sensible default thresholds for rejection. These thresholds can be configured in the feature set definitions, or by modifying the code itself.
3. **Filtering based on VascX features**. VascX computes region-specific features useful additional quality filtering before downstream analysis, and potentially as analysis covariates. The [vessel-edge or sharpness feature](../biomarkers/sharpness.md), for example, computes blur along vessel edges from perpendicular
intensity profiles. We recommend at least the sharpness feature because blur, or lack of sharpness is a common quality issue that is not fully captured by the global filter from step 1.


| Output | How it can inform QC |
| --- | --- |
| [Vessel-edge blur](../biomarkers/sharpness.md), e.g. `sharpness_crcl` | Estimates vessel-edge blur width and provides a useful indication of overall image blur. **Higher values mean more blur**, despite the column name. Calibrate thresholds carefully for the dataset, device, image scale and analysis. |
| [Luminance](../biomarkers/luminance.md), `luminance` | Summarises brightness; unusually dark or bright images may need inspection. |
| [Optic-disc variables](../biomarkers/optic-disc.md), especially `od_valid` and `od_fraction_outside_bounds` | Support OD-dependent QC: require an accepted fit when a measurement relies on the optic disc, and consider whether the disc must be fully visible. Disc position, size, shape and fovea distance provide further plausibility checks. |

:::tip QC threshold calibration
Most of these variables are continuous measures of quality, not binary decisions. We recommend to manually validate examples around candidate cutoffs and assess how filtering changes the sample. Avoid transferring numerical thresholds between datasets without checking their units and calibration. 
:::