---
title: "Biomarker explanations"
---

VascX can draw biomarker measurements directly on fundus images, showing the retinal region, vessels and measurements used to obtain a value. These **visual explanations** help you understand a biomarker and check how it behaves on your images.

:::tip[Example: understanding a CRE measurement]

Central retinal equivalent (CRE) summarises the width of the major retinal vessels. The arterial value is often called **CRAE**, and the venous value **CRVE**.

![CRE explanation: measurement circles and selected vessels on fundus images, with arteries in the top row and veins in the bottom row](/assets/biomarkers/od-centered/cre_arteries.jpg)
:::

There are several ways to generate such visual explanations on your images.

## Start with a small sample

To see the measurements before processing a whole dataset, add `--sample N`, for example:

```bash
vascx calc-biomarkers <SEGMENTATIONS> <OUTPUT_FOLDER> --feature-set od_centered --sample 3
```

When `--sample N` is provided, VascX will randomly sample `N` images from the input folder for extraction, and will plot their visual explanations. Plots contain at most **16 images per figure**, so `N > 16` has no effect on the plots. The figures cover the biomarkers in the feature set. Related biomarkers can share a figure, so there is not necessarily a separate plot for every column.

Here, `<SEGMENTATIONS>` is the output folder from `vascx run-models`, and `<OUTPUT_FOLDER>` is a new folder for the results. The command reuses the existing images and segmentations from `vascx run-models`, so it does not rerun model inference.

The default random seed is `0`, so repeating the command with the same available images and sample size selects the same images. To choose another sample, add `--seed` with a different integer.

Open `<OUTPUT_FOLDER>/plots/` to view the figures. The folder also contains `README.md` with biomarker descriptions, `biomarkers.csv` with measurements for the sample and `data_dictionary.csv` with their definitions and units.

## Generate explanations for specific images

For more flexibility, use the [explanation plotting notebook](https://github.com/Eyened/retinalysis-vascx/blob/main/notebooks/4_plot_explanations.ipynb) to plot a chosen biomarker on your own selection of images.

Open the notebook in Jupyter and set:

- `INPUT_DIR`: the folder containing your prepared images and segmentations, such as the output of `vascx run-models`.
- `FEATURE_SET_NAME`: the feature set used for your analysis.
- `FEATURE_NAME`: the resolved column name from your biomarker table, such as `cre_arteries` or `diam_crcl_veins`.
- `IMAGE_IDS`: the image IDs to display, in your preferred order.
- `OUTPUT_PATH`: where to save the figure. You can also adjust the number of columns with `N_COLS`.

Run the cells to view the explanations together and save the figure. The notebook uses `RetinaLoader` to load the selected images and the feature set's measurement settings to draw the plots. It computes only the selected measurement and any measurements it depends on; it does not rerun model inference or extract the full feature set for the folder.

Some columns share a plot: selecting `od_area`, for example, shows the optic-disc explanation. Selecting an AVR column shows the underlying arterial and venous measurements and their ratio. The defaults use prepared sample images bundled with the repository so you can try it before choosing your own data.
