---
title: "CSV outputs and naming"
---

By default, CLI output uses feature-set-aware `resolved` names; pass `--naming canonical` to preserve the full canonical convention.
VascX parameter names typically follow the format:

```
[AGGREGATION]_[BIOMARKER]_[PARAMETERS]_[REGION]_[LAYER]
```

For example, for `median_tort_dist_etdrs_arteries`:

```
AGGREGATION = median - the vessel segments are aggregated using the median
BIOMARKER = tort (Tortuosity)
PARAMETERS = dist - this is the distance implementation of tortuosity
REGION = measured over the full ETDRS grid
LAYER = arteries - arterial biomarker - as opposed to veins or vessels (combined)
```

Some of the tokens may be ommited if they match defaults or if they do not apply. A CSV mapping between the selected biomarker names in the output CSV and publication-friendly display names can be generated with the same `--naming` option:

```
vascx write-mapping --feature_set full_v3 PATH_TO_MAPPING_CSV --naming resolved
```


`vascx run-models` will write segmentations and other model predictions to `PATH_TO_FOLDER_FOR_SEGMENTATIONS`, with the following structure:

```
/path/to/segmentations
  - preprocessed_rgb/ - preprocessed fundus images
  - artery_vein/ - artery-vein model segmentations
  - vessels/ - vessel model segmentations
  - disc/ - optic disc model segmentations
  - overlays/ - optional overlays showing the segmentations
  - bounds.csv - contains the bounds of the fundus image
  - fovea.csv - model predictions of the fovea locations for each image
  - quality.csv - model estimations of CFI quality
```

The folders above will contain images with matching filenames.

> [!IMPORTANT]
> ### Biomarker post-processing and normalisation
> While several VascX biomarkers such as tortuosity are unitless, others such as vessel calibers or CREs measure distances. By default, VascX reports these distance measurements in pixels. If an explicit per-image `mm_per_pixel` value is provided, VascX uses it for ETDRS grid scaling and multiplies caliber and CRE biomarkers into millimeters. Some derived biomarkers such as artery-vein ratio (AVR) may also be computed in a post-processing stage as the ratio between caliber or CRE (recommended) measurements. The [sample post-processing notebook](https://github.com/Eyened/retinalysis-vascx/blob/main/notebooks/3_post_process.ipynb) shows how to post-process and optionally normalise a VascX export.

VascX localises feature computations using anatomical references and predefined grids:

- **Anatomical anchoring**
  - The optic disc mask and fovea position orient geometry (e.g., OD–fovea axis) and define a retinal mask.
  - All region masks are intersected with the retinal mask; features operate only on visible retina.

- **Predefined grids** (`rtnls_enface/rtnls_enface/grids`)
  - `EllipseGrid`: ellipse centered midway between disc and fovea, major axis along OD–fovea. Fields: `FullGrid`, `Superior`, `Inferior`.
  - `CircleGrid`: circle along the OD–fovea axis (`center` places it between disc and fovea; radius is `radius_multiplier × OD–fovea distance + disc radius`). Fields: `FullGrid`, `Superior`, `Inferior`.
  - `ETDRSGrid`: classic ETDRS layout with rings (`Center`, `Inner`, `Outer`), quadrants (`Superior`, `Inferior`, `Nasal`, `Temporal`, plus `Left`/`Right`), and subfields (`CSF`, `SIM`, `NIM`, `TIM`, `IIM`, `SOM`, `NOM`, `TOM`, `IOM`).
  - `HemifieldGrid`: superior/inferior half-planes split relative to the OD–fovea axis. Fields: `FullGrid`, `Superior`, `Inferior`.
  - `DiscCenteredGrid`: disc-anchored rings (`inner`, `center`, `outer`) and quadrants (`superior`, `inferior`, `nasal`, `temporal`, plus `left`/`right`), taking laterality into account.

- **Bounds and visibility (CFI bounds)**
  - For a chosen field, the platform evaluates the fraction within bounds using `grid_field_fraction_in_bounds` (and `grid_field_masks_and_fraction`).
  - If the fraction in-bounds is too small (typically < 0.5), many features skip computation and return `None` to avoid out-of-frame bias.
  - Visualizers plot the requested field overlayed on the image; computations always respect in-bounds masking.
