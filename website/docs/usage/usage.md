---
title: "Usage"
---

## Usage

To run the two stages of VascX on a folder with input CFIs:

```
vascx run-models <PATH_TO_IMAGES> <PATH_TO_FOLDER_FOR_SEGMENTATIONS>
vascx calc-biomarkers <PATH_TO_FOLDER_FOR_SEGMENTATIONS> <PATH_TO_BIOMARKER_REPORT> --feature-set full_v3 --n-jobs 8
```

> [!TIP]
> Use `vascx run-models --help` or `vascx calc-biomarkers --help` to get help.

- `PATH_TO_IMAGES` should point to a folder with input CFIs in standard imaging formats. DICOM is supported via `pydicom`. Alternatively `PATH_TO_IMAGES` may be the path to a CSV file with `id` and `path` columns, and one image per row. If provided in this way, each `path` should be an absolute path to a valid image file and the corresponding unique `id` will be used to name the output and intermediate results for that image. This is useful when the source images are not all in a single folder.

  <details>
  <summary>Sample CSV file with inputs</summary>
  
  If you chose to use a CSV file as input to `run-models`, the format should be the following:
  ```
  id,path
  drive_left_22,/path/to/samples/fundus/original/DRIVE_22.png
  drive_right_40,/path/to/samples/fundus/original/DRIVE_40.png
  chasedb_left_08,/path/to/samples/fundus/original/CHASEDB1_08L.png
  chasedb_right_12,/path/to/samples/fundus/original/CHASEDB1_12R.png
  hrf_green_04,/path/to/samples/fundus/original/HRF_04_g.jpg
  hrf_dr_07,/path/to/samples/fundus/original/HRF_07_dr.jpg
  ```
  
  Note that paths should be absolute and IDs should be unique.
  </details>

- The folder `PATH_TO_FOLDER_FOR_SEGMENTATIONS` should initially be an empty or non-existent folder. The first stage (`run-models`) will write intermediate segmentations and other AI outputs to this folder.
- The same folder should be passed to the second stage as input as shown above. 
- `PATH_TO_BIOMARKER_REPORT` is an output folder containing the biomarker values, metadata, README, manifest, and sample plots.
- The `--feature-set` is used to specify a set of biomarkers to extract. The underscore spelling `--feature_set` remains available as an alias.
- The `--n-jobs` option indicates how many CPU jobs to start. For optimal performance, we recommend to set to the number of CPU threads available divided by two, as a rule of thumb.
- By default, `run-models` picks the first available CUDA GPU, Apple MPS acceleration, or CPU. Override this with `--device`, passing any string accepted by `torch.device` (for example `cuda:0`, `mps`, or `cpu`).
- An optional `--logfile PATH_TO_LOGFILE_TXT` can be added to store warnings.

### Example 

For example, to run VascX on the provided samples folder in our git repository:

```
git clone git@github.com:Eyened/retinalysis-vascx.git rtnls_vascx
cd rtnls_vascx
vascx run-models ./samples/fundus/original/ ./samples/fundus/segmentations
vascx calc-biomarkers ./samples/fundus/segmentations ./samples/fundus/biomarker_report --feature-set full_v3 --n-jobs 8 --naming resolved
```

As an alternative, we also provide notebooks for running the three stages:

1. Preprocessing. See [this notebook](https://github.com/Eyened/retinalysis-vascx/blob/main/notebooks/0_preprocess.ipynb). This step is CPU-heavy and benefits from parallelization (see notebook).

2. Inference. See [this notebook](https://github.com/Eyened/retinalysis-vascx/blob/main/notebooks/1_segment_preprocessed.ipynb). All models can be ran in a single GPU with >10GB VRAM.

3. Feature extraction. See [this notebook](https://github.com/Eyened/retinalysis-vascx/blob/main/notebooks/2_feature_extraction.ipynb). This step is CPU-heavy again and benefits from parallelization (see notebook). Advanced users who need to pass custom parameters per image can use the alternative [advanced feature extraction notebook](https://github.com/Eyened/retinalysis-vascx/blob/main/notebooks/advanced/2_feature_extraction_advanced.ipynb).

4. Post-processing and normalisation. See our [sample post-processing notebook](https://github.com/Eyened/retinalysis-vascx/blob/main/notebooks/3_post_process.ipynb) showing how to post-process, and optionally normalise VascX exports.
