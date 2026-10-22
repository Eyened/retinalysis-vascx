---
title: "About"
slug: /
mdx:
  format: mdx
---

import BiomarkerBanner from '@site/src/components/BiomarkerBanner';

<BiomarkerBanner />

## VascX retinal vascular analysis

VascX was created to facilitate the extraction of **retinal vascular biomarkers from color fundus images**. The pipeline segments the **arteries, veins and optic disc**; and locates the **fovea**. From these intermediate outputs, it computes **region-aware** vascular biomarkers like vessel density, central retinal equivalents (CREs) or tortuosity. **Explanation plots** can be generate for any biomarker, showing the region of interest and the anatomy used. The instructions in this repository explain how to run the entire pipeline. VascX is open source and available as `retinalysis-vacx` in PyPI, and is part of the [retinalysis packages for ophthalmic research](https://eyened.github.io/retinalysis/)  .


## Features

- Currently supported biomarkers / features:
  - Bifurcation features
  - Central retinal equivalents(CREs), diameters, artery-vein ratio (AVR)
  - Tortuosity
  - Optic disc morphology
  - Image quality: sharpness, luminance
  - Other: sparsity, fractal dimension, vascular density, background Lab, temporal angles
- Many of these biomarkers support multiple implementations and configuration arguments as detailed in [our manuscript](https://arxiv.org/abs/2602.08580). The code for each biomarker specificies the supported options.
- Region-aware measurements, relative to the optic disc - fovea axis. If the region is not visible, the biomarker is not computed.
- VascX can generate visualisations for every biomarker.


## Attribution

If you use VascX, please consider citing our open access papers / preprints:
- [VascX Models: Deep Ensembles for Retinal Vascular Analysis From Color Fundus Images](https://tvst.arvojournals.org/article.aspx?articleid=2810436)
Vargas Quiros, J. D., Liefers, B., van Garderen, K. A., Vermeulen, J. P., & Klaver, C. C. W. (2025). VascX Models: Deep Ensembles for Retinal Vascular Analysis From Color Fundus Images. Translational Vision Science & Technology, 14(7), 19-19.

- [retinalysis-vascx: An explainable software toolbox for the extraction of retinal vascular biomarkers](https://arxiv.org/abs/2602.08580)
Vargas Quiros, J. V., Beyeler, M. J., Vela, S. O., Center, E. R., Bergmann, S., Klaver, C. C. W., & Liefers, B. (2026). retinalysis-vascx: An explainable software toolbox for the extraction of retinal vascular biomarkers. arXiv preprint arXiv:2602.08580.
