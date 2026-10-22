---
title: "Custom feature sets"
---

A feature set is a named list of biomarkers and their measurement settings. Create one when you want a smaller selection of biomarkers, different retinal regions, or several variants of the same measurement. You can reuse the existing biomarkers without changing how they are implemented.

## 1. Install VascX in development mode

Follow the **Development install** instructions on the [Installation page](../installation.md). This gives you a local copy of the package whose Python files you can edit:

```bash
git clone git@github.com:Eyened/retinalysis-vascx.git rtnls_vascx
cd rtnls_vascx
pip install -e .
```

Use the environment where you made this installation for the commands below. Run them from the repository root.

## 2. Create a file and choose your measurement regions

Create `vascx/fundus/feature_sets/my_features.py`. It will hold both the regions you want to measure and the feature set that uses them.

A **grid** defines the measurement area's shape, size and position relative to the optic disc and fovea. A **field** selects part of that grid: for example, the entire circle or just its superior half. Pass that field to a biomarker using `grid_field=`.

Useful grid choices include:

| Grid specification | Measurement area |
| --- | --- |
| `CircleGridSpecification` | A circle positioned along the optic-disc–fovea axis. |
| `DiscCenteredGridSpecification` | An annulus around the optic disc, with rings and directional subdivisions. |
| `EllipseGridSpecification` | An ellipse between the optic disc and fovea. |
| `ETDRSGridSpecification` | The ETDRS rings and subfields around the fovea. |
| `HemifieldGridSpecification` | Superior and inferior regions divided by the optic-disc–fovea axis. |

For the example below, use a circle centered on the fovea. In `CircleGridSpecification`, `center=0` places the center on the optic disc and `center=1` places it on the fovea. The radius is `radius_multiplier × disc–fovea distance + disc radius`. Setting `radius_multiplier=0.5` makes a smaller circle than the default value of `1.0`.

`CircleField.FullGrid` selects the whole circle, while `CircleField.Superior` selects its upper half relative to the disc–fovea axis. Match fields to their grid type; for example, use `CircleField` with a circle grid.

## 3. Define a simple feature set

Put this code in `my_features.py`:

```python
from rtnls_enface.grids.circle import CircleField
from rtnls_enface.grids.specifications import (
    CircleGridSpecification,
    GridFieldSpecification,
)

from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.od_morphology import ODMorphology
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import median
from vascx.shared.features import FeatureSet

# Use the same circle for all regional measurements.
circle = CircleGridSpecification(
    center=1.0,
    radius_multiplier=0.5,
    min_area_within_bounds=0.9,
    description=(
        "a fovea-centered circle with radius equal to half the "
        "optic-disc–fovea distance plus the optic-disc radius"
    ),
)
full_circle = GridFieldSpecification(circle, CircleField.FullGrid)
superior_circle = GridFieldSpecification(circle, CircleField.Superior)

fs_my_features = FeatureSet(
    "my_features",
    [
        Caliber(grid_field=full_circle, aggregator=median, plot=True),
        VascularDensity(grid_field=full_circle, plot=True),
        VascularDensity(grid_field=superior_circle, plot=True),
        ODMorphology(plot=True),
    ],
    description=(
        "Median caliber in a fovea-centered circle, vascular density "
        "in that circle and its superior half, and optic-disc morphology."
    ),
)
```

This computes median caliber and two regional density measurements separately for arteries and veins, plus the optic-disc measurements. `plot=True` includes sample explanation figures for that measurement in the output report; it does not change the biomarker value.

The grid's `min_area_within_bounds=0.9` requires at least 90% of the requested field to be visible for these caliber and density measurements. Otherwise, the measurement is missing. Measurements use the visible retinal area within the chosen field. This helps avoid comparing a complete region in one image with a small visible fragment in another.

The same grid can be reused across many biomarkers. Some biomarkers define their own measurement regions: for example, CRE has its own circles around the optic disc, so you configure it through its CRE settings instead of passing `grid_field=`. The existing files in `vascx/fundus/feature_sets/` provide more examples.

## 4. Make the feature set available to the command line

Add this import to `vascx/fundus/feature_sets/__init__.py`:

```python
from .my_features import fs_my_features
```

This loads the file when VascX starts, making the set available under the name `my_features`. Use a unique name in `FeatureSet(...)`; it is the name you pass to `--feature-set`. The Python variable name, `fs_my_features`, is used by the import.

You can define more than one `FeatureSet` in a file. Give each a distinct name and import each one in `__init__.py`. With the development install, you do not need to reinstall after editing these files; start a new command or restart a notebook's kernel to load the changes.

## 5. Check the definitions and run it

First, check that VascX can find the set and that its descriptions match your intent:

```bash
vascx write-readme my_features_guide.md --feature-set my_features
vascx write-mapping my_features_dictionary.csv --feature-set my_features
```

Then run it on a folder previously produced by `vascx run-models`:

```bash
vascx calc-biomarkers segmentations report_my_features --feature-set my_features --n-jobs 8
```

For a quick check on up to three images, use a separate output folder and add `--sample 3`. The same images are used in the explanation plots. Larger samples show at most sixteen images per figure. Open the report's README, dictionary and plots to confirm the regions and measurements before extracting the full dataset.

Output columns use [resolved names](naming.md) by default. Adding another measurement variant can make a column name more explicit, so regenerate the dictionary whenever you change the set.

You can also pass your feature set directly to the Python extraction helper:

```python
from pathlib import Path

from vascx.cli import make_examples
from vascx.fundus.feature_sets.my_features import fs_my_features
from vascx.utils.analysis import extract_biomarkers_to_folder

# Read images and predictions from the run-models output folder.
images = make_examples(Path("segmentations"))
values = extract_biomarkers_to_folder(
    images, fs_my_features, "report_my_features_python", n_jobs=8,
)
```

Keep your feature-set file with the analysis so you can reproduce the exact biomarker selection and grid settings later.
