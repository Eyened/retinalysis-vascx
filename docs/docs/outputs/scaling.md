---
title: "Biomarker scaling and normalisation"
---

**VascX does not normalise biomarkers by default**. Spatial measurements, such as vessel widths and distances, are normally reported in pixels of the **processed image**. If your analysis needs normalisation, apply it to the exported results as a **post-processing step**.

This is normally not a problem in single-camera studies, but becomes an issue when combining images from different devices. The same vessel can occupy a different number of pixels in images from different cameras. There are several options:

## Which measurements may need spatial normalisation?

Only measurements with spatial units, such as pixels, square pixels or inverse pixels, require this kind of scale correction. Use the `units` column in your report's `data_dictionary.csv` to identify them:

| Units | Examples |
| --- | --- | 
| `px` | Caliber, CRE, optic-disc–fovea distance, sparsity and vessel-edge blur width | 
| `px²` | Optic-disc area | 
| `px⁻¹` | Curvature-based tortuosity | 

Measurements without spatial units can still differ between devices because of image quality, colour or the retinal area captured. Spatial normalisation does not address those differences.

## Spatial normalisation

Apply known mm-per-pixel values or device conversion factors manually to the exported biomarker table. Choose a common target scale for the analysis: millimetres, or the pixel scale of a reference device.

### First restore the original image scale

Device conversion factors usually describe the **original image**, while VascX measurements use the processed image. Start with the **Restore the original image scale** section of the [post-processing notebook](https://github.com/Eyened/retinalysis-vascx/blob/main/notebooks/3_post_process.ipynb). It uses each row's `crop_scale_factor` and the units in `data_dictionary.csv` to multiply lengths by the factor, areas by its square, and divide curvature by it. You can then apply your calibration or device factors to that table.

The factor refers to the image supplied to VascX, including any resizing you did beforehand. A blank factor means the scale is unknown; re-extract with the saved preprocessing metadata to recover it. Measurements already exported in `mm`, `mm²` or `mm⁻¹` have already had both corrections applied and should not be rescaled again.

### If you have a known pixel-to-millimetre calibration

After restoring the original image scale, use the **mm-per-pixel value for each original image** as the length conversion factor:

```text
Length in mm = length in pixels × mm per pixel
Area in mm² = area in square pixels × (mm per pixel)²
Curvature in mm⁻¹ = curvature in inverse pixels / mm per pixel
```

The resulting units are `mm`, `mm²` and `mm⁻¹`. 


### Relative device conversion factors

After restoring the original image scale, choose one device as the reference for your relative device conversion factors. Here, a factor means **the multiplier that converts a length from the source device to the reference-device scale**. The reference device's factor is `1`.

For example, suppose lengths from device B occupy 1.5 times as many pixels as equivalent lengths from device A. To use A as the reference, multiply lengths from B by `1 / 1.5`, or approximately `0.667`. A width of 30 pixels from B then becomes 20 reference pixels, matching a width of 20 pixels from A. These results are in **reference-device pixels**, square reference pixels or inverse reference pixels. 

