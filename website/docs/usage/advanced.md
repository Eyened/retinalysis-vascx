---
title: "Advanced usage"
---

## Advanced Usage

### Offline Usage

By default, `vascx run-models` loads model weights from the [Hugging Face model repository](https://huggingface.co/Eyened/vascx). This is convenient on connected workstations, but it will fail on offline clusters unless the model files are provided locally.

For offline use, download the VascX model files on a machine with internet access, copy them to the cluster, and arrange them in this directory layout:

```
/path/to/vascx-models/
  - quality/
    - quality.pt
  - artery_vein/
    - av_wsoft_patches_02_finetune.pt
  - vessels/
    - vessels_may26.pt
  - disc/
    - disc_may26.pt
  - fovea/
    - fovea_may26.pt
```

Then pass that directory to `run-models`:

```
vascx run-models /path/to/images /path/to/segmentations --model-dir /path/to/vascx-models
```

You can also set the model directory once in the environment:

```
export VASCX_MODEL_DIR=/path/to/vascx-models
vascx run-models /path/to/images /path/to/segmentations
```

For more control, provide individual model files directly with options `--quality-model`, `--av-model`, `--vessels-model`, `--disc-model` and `--fovea-model`. Per-model options take precedence over `--model-dir` and `VASCX_MODEL_DIR`.


### Skipping steps during re-execution

The `run-models` command has several `--no-*` flags that are intended for re-executing part of a pipeline after some outputs already exist. These flags skip work; they do not change where downstream steps look for inputs. If a later step needs an output from a skipped step, that output must already be present in the same `OUTPUT_PATH` layout.

For example, to reuse images that were already preprocessed, place them in the output folder before running `run-models` and pass `--no-preprocess`:

```
/path/to/segmentations
  - preprocessed_rgb/
    - image_001.png
    - image_002.png
```

```
vascx run-models ./samples/fundus/original/ /path/to/segmentations --no-preprocess
```

In `--no-preprocess` mode, model inputs are read from `/path/to/segmentations/preprocessed_rgb/*.png`. Files must be RGB PNG images named `<id>.png`; the `<id>` filename stem is used for all downstream outputs, such as `artery_vein/<id>.png`, `vessels/<id>.png`, `disc/<id>.png`, and rows in `fovea.csv` and `quality.csv`. These images should have the same preprocessed format produced by VascX/fundusprep, namely a square, model-ready fundus crop, normally 1024x1024 pixels.

`DATA_PATH` is still required to be an existing path for CLI compatibility, but when `--no-preprocess` is used its contents are not used as model inputs. If you plan to run `calc-biomarkers` on the output folder, keep or provide the matching `bounds.csv` from preprocessing as well.

The other skip flags follow the same idea:

- `--no-vessels` skips vessel and artery-vein segmentation. Reuse this only when `vessels/<id>.png` and `artery_vein/<id>.png` already exist for the IDs you need downstream.
- `--no-disc` skips optic disc segmentation. Reuse this only when `disc/<id>.png` already exists for the IDs you need downstream.
- `--no-fovea` skips fovea detection. Reuse this only when `fovea.csv` already exists if you will create overlays or run biomarker extraction.
- `--no-quality` skips image quality estimation. Reuse this when `quality.csv` already exists or when you do not need quality predictions.
- `--no-overlay` skips overlay creation. Reuse this when overlays already exist or when visualization overlays are not needed.

Only models for enabled steps are required. For example, a run with `--no-quality --no-fovea` does not need `quality/quality.pt` or `fovea/fovea_may26.pt`. If no local model options are provided, VascX keeps using the default Hugging Face model locations.

### Advanced extraction

For most datasets, the standard [feature extraction notebook](https://github.com/Eyened/retinalysis-vascx/blob/main/notebooks/2_feature_extraction.ipynb) and `vascx calc-biomarkers` command are the simplest way to compute biomarkers. Advanced users can create `Retina` objects manually when they need to pass custom parameters per image, such as an explicit scaling factor `mm_per_pixel` from an acquisition platform. The [advanced feature extraction notebook](https://github.com/Eyened/retinalysis-vascx/blob/main/notebooks/advanced/2_feature_extraction_advanced.ipynb) is an alternative to `2_feature_extraction.ipynb` that demonstrates this using the sample folder.

`mm_per_pixel` is interpreted as millimeters per pixel. When provided, VascX uses it to scale some grids defined in physical units (eg. ETDRS grid), and to convert caliber and CRE outputs from pixels to millimeters. Unitless features, densities, angles, and tortuosity values are not multiplied. If no `mm_per_pixel` value is provided, VascX keeps the legacy behavior: grid scaling is derived from the optic-disc to fovea distance using the 4.75 mm assumption, and caliber/CRE outputs remain in pixels. For full feature sets, also provide the ROI mask or preprocessing bounds; the advanced notebook shows how to convert VascX preprocessing bounds into an ROI mask.

<a id="code-changes"></a>
### Code changes

We welcome advanced users who want to do one of:
- Implement new biomarkers or variants of them
- Develop custom feature sets
- Contribute to the code in any other way

For that we recommend to clone VascX repo and install in development mode (-e flag) in a clean environment:

```
git clone git@github.com:Eyened/retinalysis-vascx.git rtnls_vascx
cd rtnls_vascx
pip install -e .
```

After that, you are able to locally:
- Change or add biomarker implementations by working on the [features folder](https://github.com/Eyened/retinalysis-vascx/blob/main/vascx/fundus/features). Biomarkers must 
- Change or add "feature sets", or sets of biomarkers to extract from an image by working on the [feature_sets folder](https://github.com/Eyened/retinalysis-vascx/blob/main/vascx/fundus/feature_sets).
- Keep using the extraction notebooks and the `vascx calc-biomarkers` command to run experiments.

Ready-to-run feature sets are available under `vascx/fundus/feature_sets` and can be selected by name. Use `extract_in_parallel` when only an in-memory DataFrame is needed:

```python
from vascx.utils.analysis import extract_in_parallel

df = extract_in_parallel(examples, "full_v3", n_jobs=8)
```
