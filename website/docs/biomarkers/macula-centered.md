---
title: Macula-centered
---

# Macula-centered

| Metadata | Evaluated configuration |
| --- | --- |
| VascX version | `1.1.2.dev2+gf9ff26462` |
| Feature set | `macula_centered_rs` |
| Export | Rotterdam Study, 2026_06, F2 |
| Naming | `resolved` |
| Normalization reference | TOPCON TRC-50EX + DXC-950P |
| Reliability | ICC(2,1), 95% CI, ≥60 complete pairs |

These results describe this version and export; they do not automatically apply to newer VascX releases.

[Source fingerprints and provenance](pathname:///retinalysis-vascx/assets/reliability/macula-centered/provenance.json) · [All ICC results](pathname:///retinalysis-vascx/assets/reliability/macula-centered/icc.csv) · [Pairing attrition](pathname:///retinalysis-vascx/assets/reliability/macula-centered/pairing-attrition.csv)

## Angles

### Median Temporal Angle

Temporal vessel angle describes the angular spread of the major temporal retinal vessels. Calculated as the angle between the dominant superior and inferior temporal retinal vessels on concentric circles centered on the optic disc. Reported as the median across valid measurement circles.

Computed combinations:

- Computed on the full image. arteries: `temporal_angle_arteries`; veins: `temporal_angle_veins`.

Sample plots:

- ![sample grid](/assets/reliability/macula-centered/explanations/temporal_angle_arteries.png) — `temporal_angle_arteries`, `temporal_angle_veins`
### Mean Bifurcation Angle

Retinal vessel bifurcation angle describes the opening between two daughter vessels at a branch point. Calculated as the angle between the outgoing branches of each eligible retinal vessel bifurcation. Reported as the mean across eligible bifurcations.

Computed combinations:

- Computed in the full region of a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius. Parameters: center=0.9; radius multiplier=0.9; band crop fraction=0.12. arteries: `bifangle_crcl_full_arteries`; veins: `bifangle_crcl_full_veins`.
- Computed in the superior region of a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius. Parameters: center=0.9; radius multiplier=0.9; band crop fraction=0.12. arteries: `bifangle_crcl_superior_arteries`; veins: `bifangle_crcl_superior_veins`.
- Computed in the inferior region of a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius. Parameters: center=0.9; radius multiplier=0.9; band crop fraction=0.12. arteries: `bifangle_crcl_inferior_arteries`; veins: `bifangle_crcl_inferior_veins`.

Sample plots:

- ![sample grid](/assets/reliability/macula-centered/explanations/bifangle_crcl_full_arteries.png) — `bifangle_crcl_full_arteries`, `bifangle_crcl_full_veins`

### Reliability

![Angles: device-pair ICCs](/assets/reliability/macula-centered/icc-angles.png)

Cells show ICC(2,1), complete-pair count in parentheses, and 95% confidence interval. Rows and device comparisons without an estimable ICC remain in the downloadable table.

[Open full-size ICC plot](pathname:///retinalysis-vascx/assets/reliability/macula-centered/icc-angles.png) · [Download Angles ICC table](pathname:///retinalysis-vascx/assets/reliability/macula-centered/icc-angles.csv)

## CREs, diameters and AVR

### Length-Weighted Caliber

Retinal vessel caliber describes the width of the retinal vasculature. Calculated from the median vessel width measured along each eligible retinal vessel segment. Reported as the mean across eligible vessel segments, weighted by vessel length.

Computed combinations:

- Computed in the full region of a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius. Parameters: center=0.9; radius multiplier=0.9; band crop fraction=0.12. arteries: `diam_crcl_full_arteries`; veins: `diam_crcl_full_veins`.
- Computed in the superior region of a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius. Parameters: center=0.9; radius multiplier=0.9; band crop fraction=0.12. arteries: `diam_crcl_superior_arteries`; veins: `diam_crcl_superior_veins`.
- Computed in the inferior region of a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius. Parameters: center=0.9; radius multiplier=0.9; band crop fraction=0.12. arteries: `diam_crcl_inferior_arteries`; veins: `diam_crcl_inferior_veins`.
- Computed in the full region of a fovea-centered ETDRS grid with ring radii of 0.5, 1.5, and 3.0 mm. arteries: `diam_etdrs_full_arteries`; veins: `diam_etdrs_full_veins`.

Sample plots:

- ![sample grid](/assets/reliability/macula-centered/explanations/diam_crcl_full_arteries.png) — `diam_crcl_full_arteries`, `diam_crcl_full_veins`
### Temporal CRE

Central retinal equivalent caliber summarizes the width of the major retinal vessels. Calculated by combining the largest temporal retinal vessels crossing concentric circles centered on the optic disc using an established equivalent-caliber formula. Reported as the median across valid measurement circles.

Computed combinations:

- Computed on the full image. arteries: `cre_arteries`; veins: `cre_veins`.
- Computed in the superior region of the visible retina divided into superior and inferior regions by the optic-disc–fovea axis. arteries: `cre_hf_superior_arteries`; veins: `cre_hf_superior_veins`.
- Computed in the inferior region of the visible retina divided into superior and inferior regions by the optic-disc–fovea axis. arteries: `cre_hf_inferior_arteries`; veins: `cre_hf_inferior_veins`.

Sample plots:

- ![sample grid](/assets/reliability/macula-centered/explanations/cre_arteries.png) — `cre_arteries`, `cre_veins`
### Temporal CRE (max vessels=3)

Central retinal equivalent caliber summarizes the width of the major retinal vessels. Calculated by combining the largest temporal retinal vessels crossing concentric circles centered on the optic disc using an established equivalent-caliber formula. Reported as the median across valid measurement circles.

Computed combinations:

- Computed on the full image. arteries: `cre_max_vessels_3_arteries`; veins: `cre_max_vessels_3_veins`.
### Temporal CRE (max vessels=6)

Central retinal equivalent caliber summarizes the width of the major retinal vessels. Calculated by combining the largest temporal retinal vessels crossing concentric circles centered on the optic disc using an established equivalent-caliber formula. Reported as the median across valid measurement circles.

Computed combinations:

- Computed on the full image. arteries: `cre_max_vessels_6_arteries`; veins: `cre_max_vessels_6_veins`.
### Artery-vein ratio of Temporal CRE

Artery-vein ratio compares the arterial and venous values of the same measurement.

Calculated as the arterial measurement divided by the corresponding venous measurement. Missing inputs or a zero denominator give a missing result.

Computed combinations:

- Computed on the full image. `avr_cre` = `cre_arteries` / `cre_veins`.
- Computed in the superior region of the visible retina divided into superior and inferior regions by the optic-disc–fovea axis. `avr_cre_hf_superior` = `cre_hf_superior_arteries` / `cre_hf_superior_veins`.
- Computed in the inferior region of the visible retina divided into superior and inferior regions by the optic-disc–fovea axis. `avr_cre_hf_inferior` = `cre_hf_inferior_arteries` / `cre_hf_inferior_veins`.

This export contains historical post-processing ratios; the native VascX definitions are equivalent.

[Source CRE explanation](pathname:///retinalysis-vascx/assets/reliability/macula-centered/explanations/cre_arteries.png)

### Reliability

![CREs, diameters and AVR: device-pair ICCs](/assets/reliability/macula-centered/icc-caliber.png)

Cells show ICC(2,1), complete-pair count in parentheses, and 95% confidence interval. Rows and device comparisons without an estimable ICC remain in the downloadable table.

[Open full-size ICC plot](pathname:///retinalysis-vascx/assets/reliability/macula-centered/icc-caliber.png) · [Download CREs, diameters and AVR ICC table](pathname:///retinalysis-vascx/assets/reliability/macula-centered/icc-caliber.csv)

## Vascular densities

### Vessel Density

Retinal vascular density is the proportion of the evaluated retinal area occupied by blood vessels. Higher values indicate greater vessel coverage. Calculated as the area occupied by retinal vessels divided by the evaluated retinal area.

Computed combinations:

- Computed in the full region of a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius. Parameters: center=0.9; radius multiplier=0.9; band crop fraction=0.12. arteries: `vd_crcl_full_arteries`; veins: `vd_crcl_full_veins`.
- Computed in the superior region of a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius. Parameters: center=0.9; radius multiplier=0.9; band crop fraction=0.12. arteries: `vd_crcl_superior_arteries`; veins: `vd_crcl_superior_veins`.
- Computed in the inferior region of a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius. Parameters: center=0.9; radius multiplier=0.9; band crop fraction=0.12. arteries: `vd_crcl_inferior_arteries`; veins: `vd_crcl_inferior_veins`.

Sample plots:

- ![sample grid](/assets/reliability/macula-centered/explanations/vd_crcl_full_arteries.png) — `vd_crcl_full_arteries`, `vd_crcl_full_veins`

### Reliability

![Vascular densities: device-pair ICCs](/assets/reliability/macula-centered/icc-density.png)

Cells show ICC(2,1), complete-pair count in parentheses, and 95% confidence interval. Rows and device comparisons without an estimable ICC remain in the downloadable table.

[Open full-size ICC plot](pathname:///retinalysis-vascx/assets/reliability/macula-centered/icc-density.png) · [Download Vascular densities ICC table](pathname:///retinalysis-vascx/assets/reliability/macula-centered/icc-density.csv)

## Tortuosity

### Length-Weighted Distance Tortuosity (max segment len=0.2)

Retinal vessel tortuosity describes the degree to which vessels deviate from a straight course. Higher values indicate more tortuous vessels. Calculated for retinal vessel segments as path length divided by the straight-line distance between endpoints. Reported as the mean across eligible vessel segments, weighted by vessel length.

Computed combinations:

- Computed in the full region of a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius. Parameters: center=0.9; radius multiplier=0.9; band crop fraction=0.12. arteries: `tort_dist_max_segment_len_0p2_crcl_arteries`; veins: `tort_dist_max_segment_len_0p2_crcl_veins`.
- Computed in the full region of a fovea-centered ETDRS grid with ring radii of 0.5, 1.5, and 3.0 mm. arteries: `tort_dist_max_segment_len_0p2_etdrs_arteries`; veins: `tort_dist_max_segment_len_0p2_etdrs_veins`.

Sample plots:

- ![sample grid](/assets/reliability/macula-centered/explanations/tort_dist_max_segment_len_0p2_crcl_arteries.png) — `tort_dist_max_segment_len_0p2_crcl_arteries`, `tort_dist_max_segment_len_0p2_crcl_veins`
### Length-Weighted Curvature Tortuosity

Retinal vessel tortuosity describes the degree to which vessels deviate from a straight course. Higher values indicate more tortuous vessels. Calculated by integrating curvature along retinal vessel segments. Reported as the mean across eligible vessel segments, weighted by vessel length.

Computed combinations:

- Computed in the full region of a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius. Parameters: center=0.9; radius multiplier=0.9; band crop fraction=0.12. arteries: `tort_curv_crcl_arteries`; veins: `tort_curv_crcl_veins`.
- Computed in the full region of a fovea-centered ETDRS grid with ring radii of 0.5, 1.5, and 3.0 mm. arteries: `tort_curv_etdrs_arteries`; veins: `tort_curv_etdrs_veins`.
### Length-Weighted Distance Tortuosity (max segment len=0.15)

Retinal vessel tortuosity describes the degree to which vessels deviate from a straight course. Higher values indicate more tortuous vessels. Calculated for retinal vessel segments as path length divided by the straight-line distance between endpoints. Reported as the mean across eligible vessel segments, weighted by vessel length.

Computed combinations:

- Computed in the full region of a fovea-centered ETDRS grid with ring radii of 0.5, 1.5, and 3.0 mm. arteries: `tort_dist_max_segment_len_0p15_etdrs_arteries`; veins: `tort_dist_max_segment_len_0p15_etdrs_veins`.
### Length-Weighted Distance Tortuosity (max segment len=0.25)

Retinal vessel tortuosity describes the degree to which vessels deviate from a straight course. Higher values indicate more tortuous vessels. Calculated for retinal vessel segments as path length divided by the straight-line distance between endpoints. Reported as the mean across eligible vessel segments, weighted by vessel length.

Computed combinations:

- Computed in the full region of a fovea-centered ETDRS grid with ring radii of 0.5, 1.5, and 3.0 mm. arteries: `tort_dist_max_segment_len_0p25_etdrs_arteries`; veins: `tort_dist_max_segment_len_0p25_etdrs_veins`.

### Reliability

![Tortuosity: device-pair ICCs](/assets/reliability/macula-centered/icc-tortuosity.png)

Cells show ICC(2,1), complete-pair count in parentheses, and 95% confidence interval. Rows and device comparisons without an estimable ICC remain in the downloadable table.

[Open full-size ICC plot](pathname:///retinalysis-vascx/assets/reliability/macula-centered/icc-tortuosity.png) · [Download Tortuosity ICC table](pathname:///retinalysis-vascx/assets/reliability/macula-centered/icc-tortuosity.csv)

## Other

### Disc-Fovea Distance (Center)

Disc–fovea distance describes the anatomical separation between the optic disc and fovea. Calculated from the optic-disc center to the foveal center.

Computed combinations:

- Computed on the full image. retina: `disc_fovea_distance_center`.

Sample plots:

- ![sample grid](/assets/reliability/macula-centered/explanations/disc_fovea_distance_center.png) — `disc_fovea_distance_center`
### Disc-Fovea Distance

Disc–fovea distance describes the anatomical separation between the optic disc and fovea. Calculated from the nearest optic-disc margin to the foveal center.

Computed combinations:

- Computed on the full image. retina: `disc_fovea_distance`.
### Mean Sparsity

Retinal vessel sparsity describes the extent of vessel-free space in the retina. Higher values indicate larger vessel-free spaces. Calculated from the distance between each retinal location and the nearest vessel. Reported as the mean across evaluated retinal locations.

Computed combinations:

- Computed on the full image. vessels: `sparsity`.
- Computed in the full region of a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius. Parameters: center=0.9; radius multiplier=0.9; band crop fraction=0.12. vessels: `sparsity_crcl_full`.
- Computed in the full region of a fovea-centered ETDRS grid with ring radii of 0.5, 1.5, and 3.0 mm. vessels: `sparsity_etdrs_full`.

Sample plots:

- ![sample grid](/assets/reliability/macula-centered/explanations/sparsity.png) — `sparsity`
### Sharpness (ESF σ)

Vessel sharpness describes the clarity of vessel borders in the fundus image. Lower values indicate sharper vessel borders. Calculated from the width of the intensity transition across retinal vessel vessel edges. Reported as the mean across eligible vessel segments, weighted by vessel length.

Computed combinations:

- Computed in the full region of a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius. Parameters: center=0.9; radius multiplier=0.9; band crop fraction=0.12. vessels: `sharpness_crcl`.
- Computed in the full region of a fovea-centered ETDRS grid with ring radii of 0.5, 1.5, and 3.0 mm. vessels: `sharpness_etdrs`.

Sample plots:

- ![sample grid](/assets/reliability/macula-centered/explanations/sharpness_crcl.png) — `sharpness_crcl`
### Luminance

Fundus luminance describes the overall brightness of the retinal image. Higher values indicate a brighter image. Calculated by converting the fundus image to luminance. Reported as the mean across the visible retinal area.

Computed combinations:

- Computed on the full image. retina: `luminance`.

Sample plots:

- ![sample grid](/assets/reliability/macula-centered/explanations/luminance.png) — `luminance`

ICC unavailable: `luminance` is not present in this export.

### Reliability

![Other: device-pair ICCs](/assets/reliability/macula-centered/icc-other.png)

Cells show ICC(2,1), complete-pair count in parentheses, and 95% confidence interval. Rows and device comparisons without an estimable ICC remain in the downloadable table.

[Open full-size ICC plot](pathname:///retinalysis-vascx/assets/reliability/macula-centered/icc-other.png) · [Download Other ICC table](pathname:///retinalysis-vascx/assets/reliability/macula-centered/icc-other.csv)
