---
title: Overview & methods
mdx:
  format: mdx
---

import DevicePairCounts from '@site/static/assets/reliability/device-pair-counts.mdx';

The **Rotterdam Study** contains thousands of cases of repeat imaging, taken using two different cameras during the same visit, and capturing the same eye. We used these images to evaluate the test-retest reproducibility of VascX biomarkers across fundus cameras. We evaluated reproducibility separately for **F1 (OD-centered)** and **F2 (macula-centered)**.

:::info About generalisation
The images used in these test-retest comparisons were not used for training the segmentation models. However, our segmentation models were trained on other images from the Rotterdam Study. For that reason, the ICC scores presented here should be considered internal validation.
:::

## Image selection and quality control

We performed image selection and [quality-control](../outputs/analysis-and-qc.md) for the test-retest validation as follows:
1. **Sampling.** Up to **500 pairs per device pair per field** were sampled without replacement (seed 0).
2. **Global QC.** We excluded images classified as `bad` by the EyePACS-trained quality model.
3. **Bluriness filter.** We applied an additional filter on `sharpness_crcl` ≤ 3 to filter out blurry images.

:::info 
For more information about the test-retest reproducibility analysis, see [our manuscript](https://tvst.arvojournals.org/article.aspx?articleid=2818595).
:::

## Scaling and ICC

To adjust for differences in device FoV / scale, distance biomarkers were scaled [as described here](../outputs/scaling.md). We used [retinalysis-registration](https://github.com/Eyened/retinalysis-registration) to register the image pairs, and used the scaling factors of the registration transforms to calculate a single canonical scaling factor per device. These were then used to rescale the biomarkers in pixel units to a canonical scale.

**ICC(2,1)** is the two-way random-effects, absolute-agreement, single-measurement intraclass correlation. Each matched pair is a target and the two devices are raters. Results include a **95% confidence interval** and the complete-pair count; at least **60 pairs per biomarker** are required. Missing measurements are excluded pairwise. 

## Image pairs per device comparison

Counts are matched image pairs within each device comparison and field. **Sampled** counts include pairs present in the saved export, after global QC; **analysed** counts include pairs contributing to at least one estimable ICC after blur QC and excluding uncomputable biomarkers. Biomarkers are uncomputable when VascX cannot produce an output, due to, for example, the RoI being out of bounds. 


<DevicePairCounts />




### Paired image samples

Each device comparison shows **8 sample pairs** horizontally: the first device above, the second below. These images come from pairs contributing to at least one reported ICC.

#### F1: OD-centered

![F1 paired images by device comparison](/assets/reliability/od-centered/paired-images.jpg)

#### F2: Macula-centered

![F2 paired images by device comparison](/assets/reliability/macula-centered/paired-images.jpg)

