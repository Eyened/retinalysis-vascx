---
title: "Optic disc features"
---

<!-- Authored page: edit directly. Neither documentation generator overwrites this text. -->

`ODMorphology` describes optic disc (OD) visibility, position, size, and shape. VascX models the optic disc by fitting a **partial ellipse** to the optic disc segmentation (see examples below). The fitting is designed to work when part of the OD is not visible / out of bounds. This allows VascX to estimate the position of the OD that is out of bounds, and calculate more accurate centroid coordinates and optic-disc fovea distance. Several OD biomarkers (ovality index, circularity, area, torsion and ellipse axes) are additionally computed when the OD is fully within bounds.

| Component / variable | Availability and meaning |
| --- | --- |
| `valid` | Normally, `valid=1`. Zero indicates that the OD was likely not segmented correctly or is fully out of bounds |
| `fraction_outside_bounds` | Reconstructed ellipse fraction outside the image or fundus ROI. |
| `centroid_x`, `centroid_y` | Fitted ellipse center in pixels, x = column and y = row. |
| `fovea_distance` | Distance between OD centroid and fovea in pixels. |
| `ellipse_a`, `ellipse_b` | Fully visible discs only: semi-major and semi-minor axis lengths in pixels. |
| `ovality_index` | Fully visible discs only: a/b, with 1 indicating a circle. |
| `circularity` | Fully visible discs only: 4πA/P² using segmentation pixel area and a four-direction Crofton perimeter. Approximately 1 for a circle. |
| `area` | Fully visible discs only: optic disc segmentation area in pixels². |
| `area_norm` | Fully visible discs only: `area` divided by `fovea_distance`. |
| `torsion` | Fully visible discs only: unsigned acute angle (0–90°) between the major axis and the perpendicular to the disc–fovea line. Missing for equal axes; unstable near a circular shape. |

Unavailable components are NaN. Fully visible means the fitted ellipse has no area outside the CFI bounds. Centroids are measured from the processed image’s top-left origin; they do not align anatomy between images. Area and circularity use the binary segmentation; the axes, centroid, ovality,
and torsion use the fitted ellipse. The torsion reference follows [the disc–fovea perpendicular definition](https://doi.org/10.1038/s41598-019-38960-6).

:::info Scaling biomarkers in pixels
Measurements in pixels are relative to the preprocessed image that is input to VascX. See [Biomarker scaling](../outputs/scaling.md) to learn how to map these measurements back to the original image, and how to harmonize them if necessary.
:::


## Report examples

### Macula-centered

![Macula-centered optic disc features](/assets/biomarkers/macula-centered/od.jpg)

### OD-centered

![OD-centered optic disc features](/assets/biomarkers/od-centered/od.jpg)
