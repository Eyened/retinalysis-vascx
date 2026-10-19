---
title: "Optic disc features"
---

<!-- Authored page: edit directly. Neither documentation generator overwrites this text. -->

`ODMorphology` describes optic disc visibility, position, size, and shape.
It replaces the standalone `DiscFoveaDistance` feature in built-in feature sets.
Its fixed vector is flattened into columns with the `od_` prefix, including
`od_fovea_distance`. With fitted disc centroid $c_{OD}$ and fovea position $p_f$,

$$
d_{ODF} = \lVert c_{OD} - p_f \rVert_2.
$$

| Component / variable | Availability and meaning |
| --- | --- |
| `valid` | Always 0 or 1; 1 means VascX accepted a fitted optic-disc ellipse, including a clipped disc. |
| `fraction_outside_bounds` | For valid discs: reconstructed ellipse fraction outside the image or fundus ROI. |
| `centroid_x`, `centroid_y` | Fitted ellipse center in pixels, x = column and y = row. Estimated from the visible contour when the outside fraction is greater than zero. |
| `fovea_distance` | Centroid–fovea distance in pixels, when the fovea is available. Also estimated for clipped discs. |
| `ellipse_a`, `ellipse_b` | Fully visible discs only: semi-major and semi-minor axis lengths in pixels. |
| `ovality_index` | Fully visible discs only: a/b, with 1 indicating a circle. |
| `circularity` | Fully visible discs only: 4πA/P² using segmentation pixel area and a four-direction Crofton perimeter. Approximately 1 for a circle; pixel discretization can produce values slightly above 1. |
| `area` | Fully visible discs only: cleaned segmentation area in pixels². |
| `area_norm` | Fully visible discs with a finite, nonzero fovea distance: area / distance². |
| `torsion` | Fully visible discs with a distinct fovea: unsigned acute angle (0–90°) between the major axis and the perpendicular to the disc–fovea line. Missing for equal axes; unstable near a circular shape. |

Unavailable components are NaN. Fully visible means the reconstructed ellipse
has no area outside the visible bounds. Pixel coordinates and sizes refer to the
loaded/preprocessed image (normally 1024×1024). With `use_mm_per_pixel=True`, lengths and coordinates are converted to mm and areas to mm². Centroids are measured from the processed image’s top-left origin; they do not align anatomy between images.
Area and circularity use the cleaned segmentation; the axes, centroid, ovality,
and torsion use the fitted ellipse. The torsion reference follows
[the disc–fovea perpendicular definition](https://doi.org/10.1038/s41598-019-38960-6).

The plot shows the full fundus with segmentation, ellipse, semi-axes and torsion
reference. All twelve measurements are overlaid near the bottom in a full-width,
semi-transparent box. Clipped
centroids and distances are marked as estimates; unavailable measurements remain
visible as N/A.


## Report examples

### Macula-centered

![Macula-centered optic disc features](/assets/biomarkers/macula-centered/od.jpg)

### OD-centered

![OD-centered optic disc features](/assets/biomarkers/od-centered/od.jpg)

An accepted fit requires finite visible ellipse–segmentation IoU of at least 0.80
and must not reach the optimizer's extrapolation limits. Ovality (major/minor
semi-axis ratio) must be strictly below 1.5, including for clipped discs. No area
or area/ODFD² acceptance threshold is applied. Overlap is evaluated
only inside the image and fundus ROI, so a clipped disc can still be accepted.
Low contour error on a short arc does not override poor overlap. Rejected fits
set the disc to `None` and produce `od_valid = 0`; other OD components are NaN.
