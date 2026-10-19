---
title: Test-retest reliability
---

We evaluated agreement between fundus cameras using **Rotterdam Study** images of the same eye at the same study visit, separately for **F1 (OD-centered)** and **F2 (macula-centered)**. One image per device was selected within each visit/eye/field group. These comparisons measure reproducibility across devices.

## Cohort and quality control

Both members of a pair must pass the EyePACS-trained quality model (`bad` is excluded), with required metadata, calibration and cached inputs available. Missing quality labels are excluded. Up to **200 pairs per device comparison per field** were sampled without replacement (seed 0). Both images must also have finite scaled `sharpness_crcl` ≤ 3; higher values indicate more blur. This study-specific filter is applied after sampling; rejected pairs are not replaced. No sparsity cutoff was applied.

Excluded comparisons (devices are too similar): Triton plus–Triton plus - Fundus; Triton–Triton plus - Fundus.

### Images per device

Counts are unique images within each device and field. **Sampled** counts include pairs present in the saved export, after initial QC; **analysed** counts include images contributing to at least one estimable ICC after blur QC and measurement availability checks. Images shared between device comparisons are counted once. Excluded comparisons are omitted.

| Device | F1 sampled | F1 analysed | F2 sampled | F2 analysed |
| --- | ---: | ---: | ---: | ---: |
| TOPCON 3D OCT-1000 | 200 | 181 | 200 | 191 |
| TOPCON 3D OCT-1000 MARK II | 200 | 140 | 200 | 145 |
| TOPCON 3D OCT-2000 FA Plus | 643 | 476 | 650 | 507 |
| TOPCON Analog Device #2 | 0 | 0 | 200 | 173 |
| TOPCON DRI OCT Triton plus | 219 | 132 | 227 | 144 |
| TOPCON DRI OCT Triton plus - Fundus | 89 | 60 | 147 | 104 |
| TOPCON TRC-50DX + D7100 | 422 | 306 | 470 | 361 |
| TOPCON TRC-50EX + DXC-950P | 604 | 493 | 1011 | 841 |
| TOPCON TRC-50IX + D7100 | 206 | 147 | 212 | 163 |
| TOPCON TRC-50VT - Analog Fundus Camera | 0 | 0 | 200 | 151 |

[Download image counts](pathname:///retinalysis-vascx/assets/reliability/device-image-counts.csv)

## Scaling and ICC

Device- and field-specific geometric factors, estimated from registered image pairs, express spatial measurements in **TOPCON TRC-50EX + DXC-950P reference pixels**. Lengths are divided by the factor, areas by its square, and inverse lengths multiplied by it; ratios, angles and counts are unchanged. This corrects scale, not anatomical position.

**ICC(2,1)** is the two-way random-effects, absolute-agreement, single-measurement intraclass correlation, computed with Pingouin. Each matched pair is a target and the two devices are raters. Results include a **95% confidence interval** and the complete-pair count; at least **60 pairs per biomarker** are required. Missing measurements are excluded pairwise. Undefined estimates are omitted from plots but retained in the downloadable results.

## Paired examples

Each device comparison shows up to **8 matched pairs** horizontally: the first device above, the second below. These images come from pairs contributing to at least one reported ICC. Comparisons with no estimable ICC are omitted. The reproducible figure sample uses seed **17**, independently of cohort selection. Click a figure to expand it.

### F1: OD-centered

![F1 paired images by device comparison](/assets/reliability/od-centered/paired-images.jpg)

[View od-centered ICC results](./od-centered.mdx)

### F2: Macula-centered

![F2 paired images by device comparison](/assets/reliability/macula-centered/paired-images.jpg)

[View macula-centered ICC results](./macula-centered.mdx)
