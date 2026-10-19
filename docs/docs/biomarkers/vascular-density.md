---
title: "Vascular density"
---

<!-- Authored page: edit directly. Neither documentation generator overwrites this text. -->

**VascularDensity.** The fraction of retinal area occupied by vessels in R, computed on the binary mask B, with R restricted to the visible retinal area:

$$
D = \frac{|B \cap R|}{|R|}.
$$

The pipeline returns a fraction (0–1). The Rotterdam Study reports and reliability
exports present it as a percentage (100 × D). Neither calibration nor device
spatial correction changes this fraction.

## Report examples

### Macula-centered

![Macula-centered vascular density explanation from the Rotterdam Study report](/assets/biomarkers/macula-centered/vd_crcl_arteries.jpg)

### OD-centered

![OD-centered vascular density explanation from the Rotterdam Study report](/assets/biomarkers/od-centered/vd_crcl_arteries.jpg)
