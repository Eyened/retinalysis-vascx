---
title: "📦 Installation"
---

To install the entire fundus analysis pipeline including fundus preprocessing, model inference code and vascular biomarker extraction:

1. Create a Python virtual environment, or otherwise ensure a clean environment.

2. If necessary, install a version of torch and torchvision that matches your CUDA environment.

:::tip[Installing torch]
Depending on your system, one of the following lines may install torch + torchvision:
```
pip install torch torchvision torchaudio  # pip and CUDA 12
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia # conda and CUDA 12
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # pip and CUDA 11
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # conda and CUDA 11
```
:::

3. Install VascX:

```
pip install retinalysis-vascx
```

:::info[Development install]
To be able to directly modify VascX code to eg. customise biomarkers or develop new ones, clone and install the Python package in development mode instead:
```
git clone git@github.com:Eyened/retinalysis-vascx.git rtnls_vascx
cd rtnls_vascx
pip install -e .
```
:::
