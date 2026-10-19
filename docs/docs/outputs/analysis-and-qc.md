---
title: Analysis and QC
---

VascX provides measurements that help you assess quality, without imposing a
universal image-level pass/fail rule. Appropriate quality control (QC) depends
on the research question, the images and the biomarkers being analysed.
Individual measurements can still be unavailable when their required inputs or
anatomical fits are invalid.

## Recommended approach

Start with **global image QC**, then apply **variable-specific QC**. For example,
use an EyePACS-trained image-quality model and exclude images labelled `bad`.
Then check the measurements and anatomical regions required for your analysis;
an image suitable for one biomarker may be unsuitable for another.

| Output | How it can inform QC |
| --- | --- |
| [Vessel-edge blur](../biomarkers/sharpness.md), e.g. `sharpness_crcl` | Estimates vessel-edge blur width and provides a useful indication of overall image blur. **Higher values mean more blur**, despite the column name. Calibrate thresholds carefully for the dataset, device, image scale and analysis. |
| [Luminance](../biomarkers/luminance.md), `luminance` | Summarises brightness; unusually dark or bright images may need inspection. |
| [Sparsity](../biomarkers/sparsity.md) | Describes gaps in the vessel segmentation. It can help flag sparse or incomplete segmentations, but also reflects anatomy and is not a standalone quality label. |
| [Optic-disc variables](../biomarkers/optic-disc.md), especially `od_valid` and `od_fraction_outside_bounds` | Support OD-dependent QC: require an accepted fit when a measurement relies on the disc, and consider whether the disc must be fully visible. Disc position, size, shape and fovea distance provide further plausibility checks. |
| Missing measurements | Check availability of the particular variables needed for the analysis. A missing value does not automatically invalidate every other measurement from the image. |

Inspect examples around candidate cutoffs and assess how filtering changes the
sample. Avoid transferring numerical thresholds between datasets without checking
their units and calibration. The blur cutoff in the
[test–retest evaluation](../reliability/overview.md) is specific to that experiment,
not a VascX-wide recommendation.
