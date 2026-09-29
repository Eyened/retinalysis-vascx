---
title: "Explanations"
---

`vascx calc-biomarkers` writes a self-contained output folder:

```text
PATH_TO_BIOMARKER_REPORT/
├── biomarkers.csv
├── biomarkers.metadata.csv
├── biomarkers.names.json
├── README.md
├── report.json
└── plots/
    └── <variable>/
        └── <image-id>.png
```

`biomarkers.metadata.csv` contains the machine variable, human-friendly display name, and description for every biomarker. `biomarkers.names.json` preserves the existing variable-to-display-name JSON format. By default, the report visualizes up to three input images; use `--report-samples` or repeated `--report-image` options to select them. Use `--no-report` to skip README, manifest, and plot generation while still writing biomarkers and metadata. Existing report-owned outputs are protected unless `--overwrite` is passed.

Use `extract_biomarkers_to_folder` to write the same report bundle as the CLI:

```python
from vascx.utils.analysis import extract_biomarkers_to_folder

df = extract_biomarkers_to_folder(
    examples,
    "full_v3",
    "biomarker_report",
    n_jobs=8,
    naming="resolved",
    report_sample_size=3,
)
```

Inputs may be dictionaries accepted by `Retina.from_file` or already constructed `Retina` objects. Pass `report_retinas` to visualize a separate, explicitly selected set of representative images, or `generate_report=False` to skip README, manifest, and plot generation.
