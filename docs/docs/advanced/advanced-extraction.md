---
title: "Advanced extraction"
---

For most datasets, the standard [feature extraction notebook](https://github.com/Eyened/retinalysis-vascx/blob/main/notebooks/2_feature_extraction.ipynb) and `vascx calc-biomarkers` command are the simplest way to compute biomarkers. Advanced users can create `Retina` objects manually when they need to pass custom parameters per image, such as an explicit scaling factor `mm_per_pixel` from an acquisition platform. The [advanced feature extraction notebook](https://github.com/Eyened/retinalysis-vascx/blob/main/notebooks/advanced/2_feature_extraction_advanced.ipynb) is an alternative to `2_feature_extraction.ipynb` that demonstrates this using the sample folder.

`mm_per_pixel` is millimetres per pixel of the **original image supplied to VascX**. VascX adjusts it for preprocessing using `crop_scale_factor` (original pixels per processed pixel). The standard loaders recover this factor from the saved preprocessing bounds, including for existing runs. When constructing a `Retina` manually, pass `crop_scale_factor` explicitly; use `1` only if the image was not resized. Unknown crop scale prevents calibrated extraction rather than assuming that the scales match. Extraction defaults to `use_mm_per_pixel=False, use_fovea=True`. When enabled, calibration must be finite and positive for each image; enabled fovea coordinates must both be finite (outside-image coordinates are allowed). Disabled inputs are ignored. Validation happens during each image’s execution: single-image calls raise `InputContractError`; bulk calls log and skip that image. Missing discs remain measurement failures, with `od_valid=0` and unavailable measurements. With fovea disabled, fovea-dependent measurements are unavailable while independent measurements continue.

Feature `compute()` methods always return native units. Shared output conversion scales caliber, CRE, sparsity, sharpness, and optic-disc lengths by `mm_per_pixel × crop_scale_factor`, optic-disc area by the square of this combined factor, and curvature by its inverse. The existing dictionary and report describe the effective units; names do not change. Explicitly enabled calibration also controls physical grids such as ETDRS. Otherwise their existing 4.75 mm / disc–fovea-distance fallback remains, while measurements stay in pixels.

```python
frame = extract_biomarkers_to_folder(
    images, "macula_centered_rs", "report", use_mm_per_pixel=True, use_fovea=True,
)
```

For the CLI, use `--use-mm-per-pixel --mm-per-pixel-csv calibration.csv` with columns `id,mm_per_pixel`; use `--no-use-fovea` to disable landmarks. The `write-readme` and `write-mapping` commands accept the same unit/landmark switches.

**Measurement definition change (`native-spatial-v2`):** sparsity is now a pixel distance and curvature is mean sampled absolute curvature in inverse pixels. Neither is normalized by disc–fovea distance. Curvature sampling is uniform in the spline parameter, not an arc-length integral. Remove the former `Sparsity(normalize=...)` argument. Regenerate outputs in a fresh release; do not relabel old normalized measurements as pixels. Reports record this definition and the enabled inputs.


