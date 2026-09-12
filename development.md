# Development

`rtnls_vascx` supports Python `3.10`, `3.11`, and `3.12`.

Edit [`tests/settings.py`](tests/settings.py) to configure all numerical test
tolerances: biomarker percentage changes, segmentation Dice, fovea coordinate
errors, extraction timing, CLI timeouts, and floating-point boundary allowance.
Each setting has a comment explaining its meaning and units. Local and public
runs use the same file. The biomarker pytest option overrides its configured
default for that run. Per-feature-set YAML files contain schema exceptions only.

`test_biomarker_report.py` generates real reports for `macula_centered` and
`od_centered`, with light checks for report files and plots. `test_feature_plotting.py`
renders the individual feature plots for the same two sets. Outputs are saved in
pytest's temporary directories; use `--basetemp=/tmp/vascx-test-output` to choose a
predictable location (pytest clears that directory on each run).
The inference test is `tests/test_inference.py::test_inference`; its selection
marker remains `cli_e2e`.

The committed `samples/fundus` and `tests/reference` fixtures are intentionally
included so the tests work from a clean checkout and from the source release.

## Local tests

```bash
pip install -e ".[test]"
pytest
```

Useful variants:

```bash
pytest -m reference
pytest -m plotting
pytest -m profile
pytest -m cli_e2e
pytest -m "reference or cli_e2e"
pytest --accept-vascx-reference
```

## What the regression tests cover

The two stages are tested independently through the installed `vascx` executable:

- **Biomarker computation** (`reference`): `calc-biomarkers` reads the fixed sample
  segmentations and metadata. Each feature set's canonical CSV columns are compared
  against `tests/reference/*.parquet`, allowing up to 5% change per image by default.
  Report files are checked too. This test does not run inference.
  Multiple value mismatches are summarized per biomarker with means over failing
  images (excluding NaN), plus the image with the largest absolute difference.
  Schema messages explicitly identify variables present only in the reference
  or only in the current output.
- **AI inference** (`cli_e2e`): `run-models` reads `samples/fundus/original` and
  compares vessel, artery/vein and disc masks against the stored sample masks
  (Dice >= 0.99 for each foreground class), and fovea coordinates within 2 pixels.
  It also checks output IDs, files and finite quality logits. Quality scores do
  not yet have a numerical reference baseline. Model weights are downloaded from
  Hugging Face or reused from its cache; the CLI selects the available device.
  `VASCX_MODEL_DIR` can select a fixed local model release in both modes.

Inference runs by default in both local and public tests and loads real models.
To exclude it, use `pytest -m "not cli_e2e"` or
`tox -e public -- -m "reference and not cli_e2e"`.
The old `--run-cli-e2e` option remains accepted for compatibility but is unnecessary.

When a reviewed change intentionally updates biomarker outputs, refresh the local
references explicitly (never as part of public-package validation):

```bash
pytest --accept-vascx-reference -m reference
```

Inference uses the existing sample masks and fovea CSV as references and does not
rewrite them. Changes to model weights or preprocessing may require a separately
reviewed update of those inputs and the corresponding biomarker references.

Regression tests explicitly request `canonical` names, and reference metadata records
`naming: canonical`. VascX CLI commands and Python extraction/reporting APIs default
to `resolved` names; pass `naming="canonical"` in Python or `--naming canonical`
on the CLI when canonical output is needed. Reference column renaming preserves
all stored numerical values, including existing schema and numerical mismatches.

The biomarker regression threshold is configurable at test time:

```bash
pytest -m reference --vascx-max-percent-change 5
tox -e public -- -m reference --vascx-max-percent-change 5
# Include inference as well:
tox -e public -- -m "reference or cli_e2e" --vascx-max-percent-change 5
```

For each biomarker on each image, a numerical comparison fails when
`abs(current - reference) > (threshold / 100) * abs(reference)`.
Exactly 5% passes at the default threshold, including integer-valued biomarkers.
The threshold must be finite and non-negative; use `0` for exact numerical equality.
Zero references require zero current values. Matching NaNs pass, but transitions
between missing and measured values fail, as do changed infinities. Missing images,
variables, and reference files still fail independently of the numerical threshold.
YAML overrides control schema compatibility; historical numeric tolerance entries
are no longer used. The percentage option does not change inference mask/landmark tolerances.

## Tox

Run the full tox matrix:

```bash
tox
```

Run a single environment:

```bash
tox -e py312
tox -e pkg
```

The ordinary tox environments, including `pkg`, still use the local editable
package.

## Published-package regression checks

`tox -e public` runs the same inference and biomarker CLI regression tests against
PyPI packages, with no editable installations.

```bash
pip install tox
tox -e public
# Select a particular published VascX version:
VASCX_VERSION=1.2.3 tox -r -e public
```

`public` installs VascX from PyPI with its published dependencies, checks dependency
consistency, and prints package versions and import paths. It copies the tests and
sample data into an isolated working directory without the local package source.
It runs both regression groups by default. Tests and reference values are identical
in local and public modes; only the installed target changes. The version above is
an example; select an existing release. To validate historical releases, use tests
and references from the corresponding source tag when their CLI/schema differs.
Public testing uses the tests and references from this checkout.

```bash
tox -e public -- -m reference
tox -e public -- -m cli_e2e
```

Public runs deliberately do not add missing runtime
dependencies separately: incomplete published dependency metadata should fail.
