---
title: OD-centered
---

# OD-centered

| Metadata | Evaluated configuration |
| --- | --- |
| VascX version | `1.1.2.dev2+gf9ff26462` |
| Feature set | `od_centered_rs` |
| Export | Rotterdam Study, 2026_06, F1 |
| Naming | `resolved` |
| Normalization reference | TOPCON TRC-50EX + DXC-950P |
| Reliability | ICC(2,1), 95% CI, ≥60 complete pairs |

These results describe this version and export; they do not automatically apply to newer VascX releases.

[Source fingerprints and provenance](pathname:///retinalysis-vascx/assets/reliability/od-centered/provenance.json) · [All ICC results](pathname:///retinalysis-vascx/assets/reliability/od-centered/icc.csv) · [Pairing attrition](pathname:///retinalysis-vascx/assets/reliability/od-centered/pairing-attrition.csv)

## Angles

### Median Temporal Angle

Temporal vessel angle describes the angular spread of the major temporal retinal vessels. Calculated as the angle between the dominant superior and inferior temporal retinal vessels on concentric circles centered on the optic disc. Reported as the median across valid measurement circles.

Computed combinations:

- Computed on the full image. arteries: `temporal_angle_arteries`; veins: `temporal_angle_veins`.

Sample plots:

- ![sample grid](/assets/reliability/od-centered/explanations/temporal_angle_arteries.png) — `temporal_angle_arteries`, `temporal_angle_veins`
### Mean Bifurcation Angle

Retinal vessel bifurcation angle describes the opening between two daughter vessels at a branch point. Calculated as the angle between the outgoing branches of each eligible retinal vessel bifurcation. Reported as the mean across eligible bifurcations.

Computed combinations:

- Computed in the full region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `bifangle_crcl_full_arteries`; veins: `bifangle_crcl_full_veins`.
- Computed in the superior region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `bifangle_crcl_superior_arteries`; veins: `bifangle_crcl_superior_veins`.
- Computed in the inferior region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `bifangle_crcl_inferior_arteries`; veins: `bifangle_crcl_inferior_veins`.
- Computed in the temporal region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `bifangle_crcl_temporal_arteries`; veins: `bifangle_crcl_temporal_veins`.
- Computed in the nasal region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `bifangle_crcl_nasal_arteries`; veins: `bifangle_crcl_nasal_veins`.

Sample plots:

- ![sample grid](/assets/reliability/od-centered/explanations/bifangle_crcl_full_arteries.png) — `bifangle_crcl_full_arteries`, `bifangle_crcl_full_veins`

### Reliability

![Angles: device-pair ICCs](/assets/reliability/od-centered/icc-angles.png)

Cells show ICC(2,1), complete-pair count in parentheses, and 95% confidence interval. Rows and device comparisons without an estimable ICC remain in the downloadable table.

[Open full-size ICC plot](pathname:///retinalysis-vascx/assets/reliability/od-centered/icc-angles.png) · [Download Angles ICC table](pathname:///retinalysis-vascx/assets/reliability/od-centered/icc-angles.csv)

## CREs, diameters and AVR

### Length-Weighted Caliber

Retinal vessel caliber describes the width of the retinal vasculature. Calculated from the median vessel width measured along each eligible retinal vessel segment. Reported as the mean across eligible vessel segments, weighted by vessel length.

Computed combinations:

- Computed in the full region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `diam_crcl_full_arteries`; veins: `diam_crcl_full_veins`.
- Computed in the superior region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `diam_crcl_superior_arteries`; veins: `diam_crcl_superior_veins`.
- Computed in the inferior region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `diam_crcl_inferior_arteries`; veins: `diam_crcl_inferior_veins`.
- Computed in the temporal region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `diam_crcl_temporal_arteries`; veins: `diam_crcl_temporal_veins`.
- Computed in the nasal region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `diam_crcl_nasal_arteries`; veins: `diam_crcl_nasal_veins`.

Sample plots:

- ![sample grid](/assets/reliability/od-centered/explanations/diam_crcl_full_arteries.png) — `diam_crcl_full_arteries`, `diam_crcl_full_veins`
### Full CRE

Central retinal equivalent caliber summarizes the width of the major retinal vessels. Calculated by combining the largest retinal vessels crossing concentric circles centered on the optic disc using an established equivalent-caliber formula. Reported as the median across valid measurement circles.

Computed combinations:

- Computed on the full image. arteries: `full_cre_arteries`; veins: `full_cre_veins`.

Sample plots:

- ![sample grid](/assets/reliability/od-centered/explanations/full_cre_arteries.png) — `full_cre_arteries`, `full_cre_veins`
### Nasal CRE

Central retinal equivalent caliber summarizes the width of the major retinal vessels. Calculated by combining the largest nasal retinal vessels crossing concentric circles centered on the optic disc using an established equivalent-caliber formula. Reported as the median across valid measurement circles.

Computed combinations:

- Computed on the full image. arteries: `nasal_cre_arteries`; veins: `nasal_cre_veins`.
### Temporal CRE

Central retinal equivalent caliber summarizes the width of the major retinal vessels. Calculated by combining the largest temporal retinal vessels crossing concentric circles centered on the optic disc using an established equivalent-caliber formula. Reported as the median across valid measurement circles.

Computed combinations:

- Computed on the full image. arteries: `temporal_cre_arteries`; veins: `temporal_cre_veins`.
### Artery-vein ratio of Full CRE

Artery-vein ratio compares the arterial and venous values of the same measurement.

Calculated as the arterial measurement divided by the corresponding venous measurement. Missing inputs or a zero denominator give a missing result.

Computed combinations:

- Computed on the full image. `avr_full_cre` = `full_cre_arteries` / `full_cre_veins`.

This export contains historical post-processing ratios; the native VascX definitions are equivalent.

[Source CRE explanation](pathname:///retinalysis-vascx/assets/reliability/od-centered/explanations/full_cre_arteries.png)
### Artery-vein ratio of Nasal CRE

Artery-vein ratio compares the arterial and venous values of the same measurement.

Calculated as the arterial measurement divided by the corresponding venous measurement. Missing inputs or a zero denominator give a missing result.

Computed combinations:

- Computed on the full image. `avr_nasal_cre` = `nasal_cre_arteries` / `nasal_cre_veins`.

This export contains historical post-processing ratios; the native VascX definitions are equivalent.
### Artery-vein ratio of Temporal CRE

Artery-vein ratio compares the arterial and venous values of the same measurement.

Calculated as the arterial measurement divided by the corresponding venous measurement. Missing inputs or a zero denominator give a missing result.

Computed combinations:

- Computed on the full image. `avr_temporal_cre` = `temporal_cre_arteries` / `temporal_cre_veins`.

This export contains historical post-processing ratios; the native VascX definitions are equivalent.

### Reliability

![CREs, diameters and AVR: device-pair ICCs](/assets/reliability/od-centered/icc-caliber.png)

Cells show ICC(2,1), complete-pair count in parentheses, and 95% confidence interval. Rows and device comparisons without an estimable ICC remain in the downloadable table.

[Open full-size ICC plot](pathname:///retinalysis-vascx/assets/reliability/od-centered/icc-caliber.png) · [Download CREs, diameters and AVR ICC table](pathname:///retinalysis-vascx/assets/reliability/od-centered/icc-caliber.csv)

## Vascular densities

### Vessel Density

Retinal vascular density is the proportion of the evaluated retinal area occupied by blood vessels. Higher values indicate greater vessel coverage. Calculated as the area occupied by retinal vessels divided by the evaluated retinal area.

Computed combinations:

- Computed in the full region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `vd_crcl_full_arteries`; veins: `vd_crcl_full_veins`.
- Computed in the superior region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `vd_crcl_superior_arteries`; veins: `vd_crcl_superior_veins`.
- Computed in the inferior region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `vd_crcl_inferior_arteries`; veins: `vd_crcl_inferior_veins`.
- Computed in the temporal region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `vd_crcl_temporal_arteries`; veins: `vd_crcl_temporal_veins`.
- Computed in the nasal region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `vd_crcl_nasal_arteries`; veins: `vd_crcl_nasal_veins`.

Sample plots:

- ![sample grid](/assets/reliability/od-centered/explanations/vd_crcl_full_arteries.png) — `vd_crcl_full_arteries`, `vd_crcl_full_veins`

### Reliability

![Vascular densities: device-pair ICCs](/assets/reliability/od-centered/icc-density.png)

Cells show ICC(2,1), complete-pair count in parentheses, and 95% confidence interval. Rows and device comparisons without an estimable ICC remain in the downloadable table.

[Open full-size ICC plot](pathname:///retinalysis-vascx/assets/reliability/od-centered/icc-density.png) · [Download Vascular densities ICC table](pathname:///retinalysis-vascx/assets/reliability/od-centered/icc-density.csv)

## Tortuosity

### Length-Weighted Distance Tortuosity (max segment len=0.15)

Retinal vessel tortuosity describes the degree to which vessels deviate from a straight course. Higher values indicate more tortuous vessels. Calculated for retinal vessel segments as path length divided by the straight-line distance between endpoints. Reported as the mean across eligible vessel segments, weighted by vessel length.

Computed combinations:

- Computed in the full region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `tort_dist_max_segment_len_0p15_crcl_full_arteries`; veins: `tort_dist_max_segment_len_0p15_crcl_full_veins`.

Sample plots:

- ![sample grid](/assets/reliability/od-centered/explanations/tort_dist_max_segment_len_0p15_crcl_full_arteries.png) — `tort_dist_max_segment_len_0p15_crcl_full_arteries`, `tort_dist_max_segment_len_0p15_crcl_full_veins`
### Length-Weighted Distance Tortuosity (max segment len=0.2)

Retinal vessel tortuosity describes the degree to which vessels deviate from a straight course. Higher values indicate more tortuous vessels. Calculated for retinal vessel segments as path length divided by the straight-line distance between endpoints. Reported as the mean across eligible vessel segments, weighted by vessel length.

Computed combinations:

- Computed in the full region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `tort_dist_max_segment_len_0p2_crcl_full_arteries`; veins: `tort_dist_max_segment_len_0p2_crcl_full_veins`.
- Computed in the superior region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `tort_dist_max_segment_len_0p2_crcl_superior_arteries`; veins: `tort_dist_max_segment_len_0p2_crcl_superior_veins`.
- Computed in the inferior region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `tort_dist_max_segment_len_0p2_crcl_inferior_arteries`; veins: `tort_dist_max_segment_len_0p2_crcl_inferior_veins`.
- Computed in the temporal region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `tort_dist_max_segment_len_0p2_crcl_temporal_arteries`; veins: `tort_dist_max_segment_len_0p2_crcl_temporal_veins`.
- Computed in the nasal region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `tort_dist_max_segment_len_0p2_crcl_nasal_arteries`; veins: `tort_dist_max_segment_len_0p2_crcl_nasal_veins`.
### Length-Weighted Distance Tortuosity (max segment len=0.25)

Retinal vessel tortuosity describes the degree to which vessels deviate from a straight course. Higher values indicate more tortuous vessels. Calculated for retinal vessel segments as path length divided by the straight-line distance between endpoints. Reported as the mean across eligible vessel segments, weighted by vessel length.

Computed combinations:

- Computed in the full region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `tort_dist_max_segment_len_0p25_crcl_full_arteries`; veins: `tort_dist_max_segment_len_0p25_crcl_full_veins`.
### Length-Weighted Curvature Tortuosity

Retinal vessel tortuosity describes the degree to which vessels deviate from a straight course. Higher values indicate more tortuous vessels. Calculated by integrating curvature along retinal vessel segments. Reported as the mean across eligible vessel segments, weighted by vessel length.

Computed combinations:

- Computed in the full region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `tort_curv_crcl_full_arteries`; veins: `tort_curv_crcl_full_veins`.
- Computed in the superior region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `tort_curv_crcl_superior_arteries`; veins: `tort_curv_crcl_superior_veins`.
- Computed in the inferior region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `tort_curv_crcl_inferior_arteries`; veins: `tort_curv_crcl_inferior_veins`.
- Computed in the temporal region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `tort_curv_crcl_temporal_arteries`; veins: `tort_curv_crcl_temporal_veins`.
- Computed in the nasal region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. arteries: `tort_curv_crcl_nasal_arteries`; veins: `tort_curv_crcl_nasal_veins`.

### Reliability

![Tortuosity: device-pair ICCs](/assets/reliability/od-centered/icc-tortuosity.png)

Cells show ICC(2,1), complete-pair count in parentheses, and 95% confidence interval. Rows and device comparisons without an estimable ICC remain in the downloadable table.

[Open full-size ICC plot](pathname:///retinalysis-vascx/assets/reliability/od-centered/icc-tortuosity.png) · [Download Tortuosity ICC table](pathname:///retinalysis-vascx/assets/reliability/od-centered/icc-tortuosity.csv)

## Other

### Disc-Fovea Distance (Center)

Disc–fovea distance describes the anatomical separation between the optic disc and fovea. Calculated from the optic-disc center to the foveal center.

Computed combinations:

- Computed on the full image. retina: `disc_fovea_distance_center`.

Sample plots:

- ![sample grid](/assets/reliability/od-centered/explanations/disc_fovea_distance_center.png) — `disc_fovea_distance_center`
### Disc-Fovea Distance

Disc–fovea distance describes the anatomical separation between the optic disc and fovea. Calculated from the nearest optic-disc margin to the foveal center.

Computed combinations:

- Computed on the full image. retina: `disc_fovea_distance`.
### Mean Sparsity

Retinal vessel sparsity describes the extent of vessel-free space in the retina. Higher values indicate larger vessel-free spaces. Calculated from the distance between each retinal location and the nearest vessel. Reported as the mean across evaluated retinal locations.

Computed combinations:

- Computed on the full image. vessels: `sparsity`.
- Computed in the full region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. vessels: `sparsity_crcl_full`.

Sample plots:

- ![sample grid](/assets/reliability/od-centered/explanations/sparsity.png) — `sparsity`
### Sharpness (ESF σ)

Vessel sharpness describes the clarity of vessel borders in the fundus image. Lower values indicate sharper vessel borders. Calculated from the width of the intensity transition across retinal vessel vessel edges. Reported as the mean across eligible vessel segments, weighted by vessel length.

Computed combinations:

- Computed in the full region of a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images. Parameters: multiplier=1.16666666667; band crop fraction=0.06. vessels: `sharpness_crcl`.

Sample plots:

- ![sample grid](/assets/reliability/od-centered/explanations/sharpness_crcl.png) — `sharpness_crcl`

ICC unavailable: `sharpness_crcl` is not present in this export.
### Luminance

Fundus luminance describes the overall brightness of the retinal image. Higher values indicate a brighter image. Calculated by converting the fundus image to luminance. Reported as the mean across the visible retinal area.

Computed combinations:

- Computed on the full image. retina: `luminance`.

Sample plots:

- ![sample grid](/assets/reliability/od-centered/explanations/luminance.png) — `luminance`

ICC unavailable: `luminance` is not present in this export.

### Reliability

![Other: device-pair ICCs](/assets/reliability/od-centered/icc-other.png)

Cells show ICC(2,1), complete-pair count in parentheses, and 95% confidence interval. Rows and device comparisons without an estimable ICC remain in the downloadable table.

[Open full-size ICC plot](pathname:///retinalysis-vascx/assets/reliability/od-centered/icc-other.png) · [Download Other ICC table](pathname:///retinalysis-vascx/assets/reliability/od-centered/icc-other.csv)
