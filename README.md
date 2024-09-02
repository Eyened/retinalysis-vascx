## VascX retinal vascular analysis

VascX was created for the extraction of vascular features from fundus image segmentations.

### Installation

To install the entire fundus analysis pipeline including fundus preprocessing, model inference code and vascular biomarker extraction:

1. Create a conda or virtualenv virtual environment, or otherwise ensure a clean environment.

2. Install the [rtnls_inference package](https://github.com/Eyened/retinalysis-inference).

3. Download the vascx models and place them in the folder pointed by `$RTNLS_MODEL_RELEASES`.

4. Install retinalysis-enface and retinalysis-fundusprep:
```
git clone git@github.com:Eyened/retinalysis-enface.git rtnls_enface
cd rtnls_enface
pip install -e .

git clone git@github.com:Eyened/retinalysis-fundusprep.git rtnls_fundusprep
cd rtnls_fundusprep
pip install -e .
```

5. Install VascX:

```
git clone git@github.com:Eyened/retinalysis-vascx.git rtnls_vascx
cd rtnls_vascx
pip install -e .
```

### Usage

To speed up re-execution of vascx we recommend to run the preprocessing, segmentation and feature extraction steps separately:

1. Preprocessing. See [this notebook](./notebooks/0_preprocess.ipynb). This step is CPU-heavy and benefits from parallelization (see notebook).

2. Inference. See [this notebook](./notebooks/1_segment_preprocessed.ipynb). All models can be ran in a single GPU with >10GB VRAM.

3. Feature extraction. See [this notebook](./notebooks/2_feature_extraction.ipynb). This step is CPU-heavy again and benefits from parallelization (see notebook).
