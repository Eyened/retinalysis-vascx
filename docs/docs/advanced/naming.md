---
title: "Understanding biomarker names"
---

After running `vascx calc-biomarkers`, open `biomarkers.csv` in your report folder. Each row represents an image, and each biomarker column contains one measurement. **By default, these columns use resolved names**: short names that identify the measurements in your chosen feature set.

Keep `data_dictionary.csv` alongside the results. It lists each column's readable name, description and units, so you do not have to work out every detail from the column heading. The report's `README.md` gives further explanations and links to example plots; see [Biomarker explanations](../outputs/explanations.md).

## Reading a column name

A name tells you what was measured and, where relevant, which vessels and retinal region were used. For example, in the `macula_centered` feature set:

| Column | How to read it |
| --- | --- |
| `diam_crcl_arteries` | Vessel caliber (diameter), measured in the circular region, for arteries. |
| `diam_etdrs_veins` | Vessel caliber, measured in the ETDRS region around the fovea, for veins. |
| `cre_arteries` | Central retinal artery equivalent (CRAE): a summary of the widths of the major arteries. |
| `bif_angle_mean_crcl_arteries` | Mean branching angle at arterial bifurcations in the circular region. |
| `od_fovea_distance` | Distance between the optic disc and the fovea. |
| `background_lab_l` | Background lightness, the L* component of the Lab colour measurements. |

Some useful parts to recognise:

| Part of the name | Meaning |
| --- | --- |
| `diam`, `cre`, `tort`, `vd`, `fd`, `bif`, `od` | Caliber, central retinal equivalent, tortuosity, vascular density, fractal dimension, bifurcation measurements and optic-disc measurements. |
| `arteries`, `veins` | Which vessel type was measured. A final `vessels` refers to the combined vessel layer. |
| `crcl`, `disc`, `etdrs`, `elps`, `hf` | The measurement region: a circular/disc-centered region, disc-centered grid, ETDRS grid, ellipse or hemifield. Check the report for its exact position and size. |
| `superior`, `inferior`, `nasal`, `temporal` | The upper, lower, nasal or temporal part of the retinal region. |
| `full` | The full configured region, which need not cover the whole image. |
| `md`, `lw`, `std` | How individual measurements were summarised: median, length-weighted mean or standard deviation. |
| `dist`, `curv` after `tort` | Distance-based tortuosity (path length relative to a straight line) or curvature-based tortuosity (how sharply vessels bend). |

Longer names usually identify a particular variant. For example, `cre_max_vessels_3_arteries` uses at most three arteries. In `tort_dist_max_segment_len_0p2_crcl_arteries`, `0p2` means **0.2**: long vessel segments are split into pieces no longer than 20% of the optic-disc–fovea distance before measuring distance-based tortuosity. A `p` replaces the decimal point in numerical settings.

## Why some details are missing from the name

Resolved names leave out settings that are shared by the relevant measurements in a feature set. The measurement itself is unchanged.

For example, `macula_centered` uses length-weighted caliber measurements. Its resolved column `diam_crcl_arteries` therefore leaves out `lw`. If a custom feature set includes both median and length-weighted caliber, its names retain `md` and `lw` to tell those measurements apart. Similarly, `od_fovea_distance` leaves out the longer name's `_retina` ending.

**Read a resolved name together with its feature set.** For example, `cre_arteries` uses the temporal vessels in `macula_centered`, but the full region around the disc in `od_centered`. Both are arterial CRE measurements, but their definitions differ. Keep the report with the table, especially when comparing exports from different feature sets.

## Getting the longer, canonical names

Canonical names retain the naming details that resolved names shorten. They generally combine the summary method, biomarker, component or settings, region and vessel type. For the `macula_centered` caliber example, the canonical name is `lw_diam_crcl_full_arteries`.

:::tip[Export canonical names]
Add `--naming canonical` when extracting biomarkers:

```bash
vascx calc-biomarkers segmentations report_canonical --feature-set macula_centered --naming canonical
```

To inspect the canonical names and descriptions without running extraction, write a separate dictionary:

```bash
vascx write-mapping canonical_dictionary.csv --feature-set macula_centered --naming canonical
```

Both commands leave the measurement definitions unchanged. The first writes results into a new report folder; the second only writes metadata.
:::

## Using readable names in figures and tables

The `display_name` column of `data_dictionary.csv` provides labels for presentation. For example, `diam_crcl_arteries` maps to **Caliber Circle - A** in `macula_centered`; `A` and `V` denote arteries and veins.

You can also export this mapping separately:

```bash
vascx write-mapping biomarker_names.csv --feature-set macula_centered
vascx write-mapping biomarker_names.json --feature-set macula_centered --json
```

The CSV contains `variable`, `display_name`, `description` and `units`. The JSON contains just the column-name-to-display-name pairs. Use the same feature set and naming convention as your results; add `--naming canonical` for canonical exports.

For example, in Python you can use the dictionary already in your report to label a presentation copy of the results:

```python
import pandas as pd

values = pd.read_csv("report/biomarkers.csv", index_col=0)
dictionary = pd.read_csv("report/data_dictionary.csv")
labels = dictionary.set_index("variable")["display_name"].to_dict()
values_for_display = values.rename(columns=labels)
```

Use the original column names for analysis and the display names for figure axes or table headings. Include units from the dictionary where appropriate.
